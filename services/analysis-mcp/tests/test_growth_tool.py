"""The `growth` tool end to end: every refusal path, one real analyst
question through the registered tool, and the replay + execution-identity-
perturbation determinism proofs.
"""

from __future__ import annotations

import json
import secrets

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.artifact import Interval
from analysis_mcp.engine.growth import run_growth
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import (
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.rank_tool import rank
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp import Client


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


async def _seed_custom_frame(store: InMemoryMaterialisationStore, frame: Frame) -> dict[str, str]:
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


# Refusals ----------------------------------------------------------------------


async def test_op_spec_invalid_refuses(seeded) -> None:
    _store, _handles = seeded
    result = await growth(frame_ref="not-a-valid-pair", window_start="2020", window_end="2023")
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_window_end_before_window_start_refuses_op_spec_invalid(seeded) -> None:
    _store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap, window_start="2023", window_end="2020"
    )
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_unregistered_pair_refuses_frame_not_held(seeded) -> None:
    _store, _handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await growth(frame_ref=never_registered, window_start="2020", window_end="2023")
    assert result.structured_content["code"] == "frame_not_held"


async def test_gap_year_inside_window_refuses(seeded) -> None:
    _store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2017",
        window_end="2022",
        entity_by=("reporter",),
    )
    assert result.structured_content["code"] == "growth_gap_year"
    assert result.structured_content["fix"]["entity_key"] == {"reporter": "dcid:USA"}
    assert result.structured_content["fix"]["gap_years"] == ["2021"]
    assert result.structured_content["fix"]["longest_gap_free_window"] == ["2017", "2020"]
    assert "reporter=dcid:USA" in result.content[0].text


async def test_span_input_refuses_derived_input_unusable_before_the_engine(seeded) -> None:
    store, _handles = seeded
    span_frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2018/2022",
                "value": 1.0,
            }
        ],
    )
    result = await growth(
        frame_ref=await _seed_custom_frame(store, span_frame),
        window_start="2018",
        window_end="2022",
        entity_by=("reporter",),
    )
    assert result.structured_content["code"] == "derived_input_unusable"
    assert result.structured_content["fix"] == {"span": "2018/2022"}


async def test_output_rows_and_trace_carry_the_window_as_a_span(seeded) -> None:
    store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2017",
        window_end="2020",
        entity_by=("reporter",),
    )
    stored_frame, stored_trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    periods = {
        dict(zip(stored_frame.columns, row, strict=True))["time_period"]
        for row in stored_frame.rows
    }
    assert periods == {"2017/2020"}
    assert stored_trace.time == Interval(start="2017", end="2020", contiguous=True)
    assert (
        "time: labels stand for the span 2017 to 2020; every period in the span is present"
        in result.content[0].text
    )


async def test_gap_year_with_no_entity_by_names_the_entity_less_frame(seeded) -> None:
    _store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap, window_start="2017", window_end="2022"
    )
    assert result.structured_content["code"] == "growth_gap_year"
    assert result.structured_content["fix"]["entity_key"] == {}
    # `content` is `message`, a blank line, then the refusal envelope as compact JSON --
    # the envelope legitimately renders `"entity_key":{}` for an empty dict, so the "no
    # bare {}" assertion below is scoped to the message itself, not the whole content.
    message = result.content[0].text.split("\n\n", 1)[0]
    assert "(all rows)" in message
    assert "{}" not in message


async def test_relative_growth_over_a_per_period_rank_refuses_derived_input_ordinal(
    seeded,
) -> None:
    """The tool-level mirror of the plan-level rule: a per-period rank's
    own output carries the rank position, not a measured value, and a
    relative growth rate over it is meaningless. `change="absolute"` (see
    `test_absolute_change_over_a_rank_series_computes_in_places` in
    `test_derived_metadata.py`) is the retry the refusal names."""
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = {
        "spec_id": ranked.structured_content["spec_id"],
        "content_version": ranked.structured_content["content_version"],
    }
    result = await growth(
        frame_ref=ranked_ref, window_start="2021", window_end="2023", entity_by=("reporter",)
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
    assert result.structured_content["retry_mode"] == "correct_and_retry"


async def test_unknown_entity_by_column_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2017",
        window_end="2020",
        entity_by=("donor",),
    )
    assert result.structured_content["code"] == "role_ambiguous"


async def test_duplicate_entity_by_column_refuses_op_spec_invalid_rather_than_crashing(
    seeded,
) -> None:
    """A duplicate `entity_by` entry passes op-spec parsing but
    `op_spec_hash_for` raises `OpSpecIdentityError` -- this must reach the
    tool boundary as a typed refusal, not an unhandled exception."""
    _store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2017",
        window_end="2020",
        entity_by=("reporter", "reporter"),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_zero_base_refuses_non_finite_result(seeded) -> None:
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 0.0},
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2021", "value": 5.0},
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await growth(
        frame_ref=custom_ref, window_start="2020", window_end="2021", entity_by=("reporter",)
    )
    assert result.structured_content["code"] == "non_finite_result"


# Absolute change --------------------------------------------------------------


async def test_zero_base_succeeds_under_absolute_change(seeded) -> None:
    """The refusal `test_zero_base_refuses_non_finite_result` hits above is
    relative-only -- the same zero base succeeds under `change='absolute'`."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 0.0},
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2021", "value": 5.0},
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await growth(
        frame_ref=custom_ref,
        window_start="2020",
        window_end="2021",
        entity_by=("reporter",),
        change="absolute",
    )
    assert result.is_error is False
    assert "+5" in result.content[0].text


async def test_absolute_change_keeps_the_input_unit_in_the_digest(seeded) -> None:
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2021",
                "value": 15.0,
                "unit": "dcid:USDollar",
            },
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await growth(
        frame_ref=custom_ref,
        window_start="2020",
        window_end="2021",
        entity_by=("reporter",),
        change="absolute",
    )
    assert result.is_error is False
    assert "absolute change" in result.content[0].text
    assert "dcid:USDollar" in result.content[0].text


async def test_heterogeneous_unit_frame_refuses_with_disambiguation_candidates(seeded) -> None:
    _store, handles = seeded
    result = await growth(
        frame_ref=handles.multi_unit,
        window_start="2022",
        window_end="2023",
        entity_by=("reporter",),
    )
    assert result.structured_content["code"] == "unit_heterogeneous"


async def test_unit_disambiguation_retry_succeeds(seeded) -> None:
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2021",
                "value": 20.0,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 1.0,
                "unit": "dcid:Percent",
            },
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2021",
                "value": 2.0,
                "unit": "dcid:Percent",
            },
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    refusal = await growth(
        frame_ref=custom_ref, window_start="2020", window_end="2021", entity_by=("reporter",)
    )
    assert refusal.structured_content["code"] == "unit_heterogeneous"

    retried = await growth(
        frame_ref=custom_ref,
        window_start="2020",
        window_end="2021",
        entity_by=("reporter",),
        unit="dcid:USDollar",
    )
    assert retried.is_error is False
    assert "unit: dcid:USDollar" in retried.content[0].text


# Success: one real analyst question end to end, through the registered tool ----


async def test_real_analyst_question_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "USA's ODA disbursed growth from 2017 to 2020" -- the gap-free
    sub-window of the seeded gap fixture -- via the live FastMCP `Client`."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "growth",
            {
                "frame_ref": handles.time_series_with_gap,
                "window_start": "2017",
                "window_end": "2020",
                "entity_by": ["reporter"],
            },
        )

        assert result.is_error is False
        digest = result.content[0].text
        assert result.structured_content["spec_id"].startswith("ops_")
        assert result.structured_content["content_version"].startswith("ocv_")
        assert "reporter=dcid:USA" in digest
        assert "+1.7%" in digest


# Replay + execution-identity perturbation ---------------------------------------


async def test_replay_from_execution_record_content_is_bit_identical(seeded) -> None:
    store, handles = seeded

    result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2017",
        window_end="2020",
        entity_by=("reporter",),
    )
    assert result.is_error is False
    output_ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    stored_frame, stored_trace = await store.get(output_ref)

    canonical_bytes = stored_trace.execution_record.content.encode()
    assert recompute_op_spec_hash(canonical_bytes) == stored_trace.op_spec_hash

    parameters = json.loads(stored_trace.execution_record.content)["parameters"]
    input_frame, _input_trace = await store.get(FrameRef(**handles.time_series_with_gap))
    replayed = run_growth(
        input_frame,
        window_start=parameters["window_start"],
        window_end=parameters["window_end"],
        entity_by=tuple(parameters["entity_by"]),
        change=parameters["change"],
        pinned_unit=parameters["unit"],
    )
    assert replayed.frame == stored_frame

    # Distinct from the recompute-from-parameters replay above. Proves the
    # *stored* frame_content_version was derived over the stored output
    # frame, not the input frame's bytes.
    assert (
        derive_frame_content_version(frame=stored_frame, upstream_trace=stored_trace.upstream_trace)
        == stored_trace.frame_content_version
    )

    recomputed_content_version = recompute_toolkit_content_version(
        op_spec_hash=stored_trace.op_spec_hash,
        input_content_versions=list(stored_trace.input_content_versions),
        execution_identity=stored_trace.execution_identity,
    )
    assert recomputed_content_version == stored_trace.content_version


async def test_execution_identity_perturbation_moves_the_content_version(seeded) -> None:
    store, handles = seeded

    result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2017",
        window_end="2020",
        entity_by=("reporter",),
    )
    output_ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    _stored_frame, stored_trace = await store.get(output_ref)

    bumped_identity = ExecutionIdentity(
        engine_version=stored_trace.execution_identity.engine_version,
        frame_grammar_version=stored_trace.execution_identity.frame_grammar_version,
        reduction_policy_version="9.9.9",
    )
    moved_content_version = mint_toolkit_content_version(
        op_spec_hash=stored_trace.op_spec_hash,
        input_content_versions=list(stored_trace.input_content_versions),
        execution_identity=bumped_identity,
    )
    assert moved_content_version != stored_trace.content_version
