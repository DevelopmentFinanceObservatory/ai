"""The `compare` tool end to end: every refusal path, one real analyst
question through the registered tool, and the replay + execution-identity-
perturbation determinism proofs.
"""

from __future__ import annotations

import json

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.compare_tool import compare
from analysis_mcp.engine.compare import run_compare
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from analysis_mcp.identity import (
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import AggregateOpSpec, CompareOpSpec
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp import Client
from runner_replay import frame_digest


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
    digest = frame_digest(frame)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


# Refusals ----------------------------------------------------------------------


async def test_op_spec_invalid_refuses(seeded) -> None:
    _store, _handles = seeded
    result = await compare(frame_ref="not-a-valid-pair")
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_unregistered_left_pair_refuses_frame_not_held(seeded) -> None:
    _store, _handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await compare(frame_ref=never_registered)
    assert result.structured_content["code"] == "frame_not_held"


async def test_unregistered_right_pair_refuses_frame_not_held(seeded) -> None:
    _store, handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await compare(frame_ref=handles.join_pair[0], right_frame_ref=never_registered)
    assert result.structured_content["code"] == "frame_not_held"


async def test_disjoint_periods_refuse(seeded) -> None:
    store, _handles = seeded
    left = Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0}],
    )
    right = Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:Y", "reporter": "dcid:USA", "time_period": "2023", "value": 2.0}],
    )
    left_ref = await _seed_custom_frame(store, left)
    right_ref = await _seed_custom_frame(store, right)

    result = await compare(frame_ref=left_ref, right_frame_ref=right_ref)
    assert result.structured_content["code"] == "comparison_periods_disjoint"
    assert result.structured_content["fix"]["left_periods"] == ["2020"]
    assert result.structured_content["fix"]["right_periods"] == ["2023"]


async def test_ambiguous_entity_binding_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    # join_compatible_frame_pair's own frames carry three donors -- an empty
    # entity binding leaves more than one row per period.
    result = await compare(frame_ref=handles.join_pair[0], right_frame_ref=handles.join_pair[1])
    assert result.structured_content["code"] == "role_ambiguous"


async def test_heterogeneous_unit_frame_refuses(seeded) -> None:
    _store, handles = seeded
    result = await compare(
        frame_ref=handles.multi_unit,
        left_entity={"reporter": "dcid:USA"},
    )
    assert result.structured_content["code"] == "unit_heterogeneous"


async def test_unknown_entity_column_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await compare(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"donor": "dcid:USA"},
        right_entity={"reporter": "dcid:USA"},
    )
    assert result.structured_content["code"] == "role_ambiguous"
    assert result.structured_content["fix"]["columns"] == ["donor"]


async def test_left_entity_key_order_produces_a_byte_identical_refusal(seeded) -> None:
    """Two calls whose `left_entity` binding differs only in key insertion
    order mint the identical `op_spec_hash`: `identity._normalise_
    parameters`'s own docstring notes entity bindings need no entry there
    because they're JSON objects, and JCS already sorts object members on
    its own before hashing. `validate_entity_columns`
    (`analysis_mcp.engine.common`) builds `UnknownEntityColumnError.columns`
    from the binding's own iteration order it receives, so without
    canonicalising that order first, which of two unknown columns is named
    first in the refusal's `fix["columns"]` -- and therefore the rendered
    digest text -- would depend on the caller's own raw order, even though
    both calls claim "same inputs, same result." Both calls canonicalise
    the binding's own keys (sorted) before the engine ever runs, so the two
    refusals are byte-identical regardless of which order the caller
    happened to supply them in."""
    _store, handles = seeded
    op_spec_a = CompareOpSpec(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"zzz_col": "x", "aaa_col": "y"},
        right_entity={"reporter": "dcid:USA"},
    )
    op_spec_b = CompareOpSpec(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"aaa_col": "y", "zzz_col": "x"},
        right_entity={"reporter": "dcid:USA"},
    )
    assert op_spec_hash_for(op_spec_a) == op_spec_hash_for(op_spec_b)

    first = await compare(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"zzz_col": "x", "aaa_col": "y"},
        right_entity={"reporter": "dcid:USA"},
    )
    second = await compare(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"aaa_col": "y", "zzz_col": "x"},
        right_entity={"reporter": "dcid:USA"},
    )
    assert first.is_error is True
    assert second.is_error is True
    assert first.structured_content["code"] == second.structured_content["code"] == "role_ambiguous"
    assert first.structured_content["fix"] == second.structured_content["fix"]
    assert first.content == second.content


async def test_stat_var_as_entity_selector_refuses_role_ambiguous(seeded) -> None:
    """`compare` disallows `stat_var` as an entity selector because it would
    defeat the point of identifying one entity within one already-resolved
    StatVar. Unlike `growth`/`rank`/`join`, `compare`'s output frame builds
    with `role_keys=[]`, so `left_entity` never reaches `Frame.build`'s own
    `role_keys`. This is a deliberate refusal, not the ordinary role-key
    collision that raises `role_ambiguous` elsewhere."""
    _store, handles = seeded
    result = await compare(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"stat_var": "dcid:some_stat_var"},
        right_entity={"reporter": "dcid:USA"},
    )
    assert result.structured_content["code"] == "role_ambiguous"
    assert result.structured_content["fix"]["columns"] == ["stat_var"]


async def test_cross_frame_unit_mismatch_refuses(seeded) -> None:
    """Each side resolves cleanly to its own single unit, but the two units
    differ -- distinct from `unit_heterogeneous`, which fires within one
    frame."""
    store, _handles = seeded
    left = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 1.0,
                "unit": "dcid:USDollar",
            }
        ],
    )
    right = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 2.0,
                "unit": "dcid:Percent",
            }
        ],
    )
    left_ref = await _seed_custom_frame(store, left)
    right_ref = await _seed_custom_frame(store, right)

    result = await compare(frame_ref=left_ref, right_frame_ref=right_ref)
    assert result.structured_content["code"] == "cross_frame_unit_mismatch"
    assert result.structured_content["fix"]["left"] == "dcid:USDollar"
    assert result.structured_content["fix"]["right"] == "dcid:Percent"


async def test_cross_frame_unit_unbound_refuses_and_is_retryable(seeded) -> None:
    """The left side's own rows leave `unit` unbound entirely; the right
    side resolves to a real unit. Distinct from `cross_frame_unit_mismatch`
    -- and, unlike an unanticipated `internal_error`, this is a
    `correct_and_retry` refusal: retry with a left frame_ref whose own rows
    actually declare a unit."""
    store, _handles = seeded
    left = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 1.0,
            }
        ],
    )
    right = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 2.0,
                "unit": "dcid:USDollar",
            }
        ],
    )
    left_ref = await _seed_custom_frame(store, left)
    right_ref = await _seed_custom_frame(store, right)

    result = await compare(frame_ref=left_ref, right_frame_ref=right_ref)
    assert result.structured_content["code"] == "cross_frame_unit_unbound"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"]["unbound_side"] == "left"
    assert result.structured_content["fix"]["bound_side"] == "right"
    assert result.structured_content["fix"]["bound_unit"] == "dcid:USDollar"


async def test_non_finite_delta_refuses(seeded) -> None:
    store, _handles = seeded
    left = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": -1.7e308,
                "unit": "dcid:USDollar",
            }
        ],
    )
    right = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 1.7e308,
                "unit": "dcid:USDollar",
            }
        ],
    )
    left_ref = await _seed_custom_frame(store, left)
    right_ref = await _seed_custom_frame(store, right)

    result = await compare(frame_ref=left_ref, right_frame_ref=right_ref)
    assert result.structured_content["code"] == "non_finite_result"


# Success: one real analyst question end to end, through the registered tool ----


async def test_real_analyst_question_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "How does USA's ODA disbursed compare to USA's GNI, 2022-2023" via
    the live FastMCP `Client`, exercising the real registered tool."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "compare",
            {
                "frame_ref": handles.join_pair[0],
                "right_frame_ref": handles.join_pair[1],
                "left_entity": {"reporter": "dcid:USA"},
                "right_entity": {"reporter": "dcid:USA"},
            },
        )

        assert result.is_error is False
        digest = result.content[0].text
        assert result.structured_content["spec_id"].startswith("ops_")
        assert result.structured_content["content_version"].startswith("ocv_")
        assert "2 intersected period(s)" in digest
        assert "2022:" in digest and "2023:" in digest


# Two slices of one frame_ref, through the registered tool ----------------------


async def test_two_slices_of_one_frame_ref(seeded) -> None:
    _store, handles = seeded
    result = await compare(
        frame_ref=handles.plain_single_unit,
        left_entity={"reporter": "dcid:USA"},
        right_entity={"reporter": "dcid:GBR"},
    )
    assert result.is_error is False


# Replay + execution-identity perturbation ---------------------------------------


async def test_replay_from_execution_record_content_is_bit_identical(seeded) -> None:
    store, handles = seeded

    result = await compare(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"reporter": "dcid:USA"},
        right_entity={"reporter": "dcid:USA"},
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
    left_frame, _left_trace = await store.get(FrameRef(**handles.join_pair[0]))
    right_frame, _right_trace = await store.get(FrameRef(**handles.join_pair[1]))
    replayed = run_compare(
        left_frame,
        right_frame,
        stat_var=f"dcid:{stored_trace.op_spec_hash}",
        left_entity=parameters["left_entity"],
        right_entity=parameters["right_entity"],
        left_pinned_unit=parameters["left_unit"],
        right_pinned_unit=parameters["right_unit"],
    )
    assert replayed.frame == stored_frame
    assert set(stored_frame.table.column("stat_var").to_pylist()) == {
        f"dcid:{stored_trace.op_spec_hash}"
    }

    # Distinct from the recompute-from-parameters replay above. Proves the
    # *stored* frame_content_version was derived over the stored output
    # frame, not the input frame's bytes.
    assert derive_frame_content_version(frame=stored_frame) == stored_trace.frame_content_version

    recomputed_content_version = recompute_toolkit_content_version(
        op_spec_hash=stored_trace.op_spec_hash,
        input_content_versions=list(stored_trace.input_content_versions),
        execution_identity=stored_trace.execution_identity,
    )
    assert recomputed_content_version == stored_trace.content_version


async def test_execution_identity_perturbation_moves_the_content_version(seeded) -> None:
    store, handles = seeded

    result = await compare(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        left_entity={"reporter": "dcid:USA"},
        right_entity={"reporter": "dcid:USA"},
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


def test_description_names_join_as_the_verb_for_two_variables() -> None:
    assert "join" in compare.__doc__
