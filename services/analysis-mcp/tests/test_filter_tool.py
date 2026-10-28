"""The `filter` tool end to end: every refusal path, the empty-result-is-
not-an-error policy, one real analyst question through the registered tool,
and the replay + execution-identity-perturbation determinism proofs.
"""

from __future__ import annotations

import json

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.engine.filter import run_filter
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from analysis_mcp.identity import (
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import AggregateOpSpec, FilterOpSpec
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp import Client
from rolling_support import anchored, second_version
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
    result = await filter_verb(
        frame_ref="not-a-valid-pair",
        predicates=({"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 1},),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_unregistered_pair_refuses_frame_not_held(seeded) -> None:
    _store, _handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await filter_verb(frame_ref=never_registered)
    assert result.structured_content["code"] == "frame_not_held"


async def test_unknown_predicate_column_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=({"kind": "ordered_threshold", "column": "donor", "op": "gte", "value": 1},),
    )
    assert result.structured_content["code"] == "role_ambiguous"


async def test_predicate_type_mismatch_refuses_rather_than_dropping_every_row(seeded) -> None:
    """An `ordered_threshold` predicate against `reporter` (a plain-string
    role column) must refuse rather than return `False` for every row, which
    would be indistinguishable from a predicate that legitimately matched
    nothing."""
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=({"kind": "ordered_threshold", "column": "reporter", "op": "gte", "value": 1},),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "filter_predicate_type_mismatch"
    assert result.structured_content["fix"]["column"] == "reporter"


async def test_strict_place_level_with_an_untyped_entity_refuses_with_its_column(seeded) -> None:
    store, _handles = seeded
    frame_ref = await _seed_custom_frame(
        store,
        Frame.build(
            role_keys=["observationAbout", "observationAbout_type"],
            rows=[
                {
                    "stat_var": "dcid:Test",
                    "observationAbout": "country/KEN",
                    "observationAbout_type": "untyped",
                    "time_period": "2023",
                    "value": 1.0,
                }
            ],
        ),
    )

    result = await filter_verb(
        frame_ref=frame_ref,
        predicates=(
            {
                "kind": "place_level",
                "column": "observationAbout_type",
                "level": "Country",
                "strict": True,
            },
        ),
    )

    assert result.is_error is True
    assert result.structured_content == {
        "code": "filter_predicate_type_mismatch",
        "message": "A 'place_level' predicate on column 'observationAbout_type' doesn't "
        "match that column's own declared cell type (untyped place entity). Retry with the "
        "correct predicate kind for this column, or a different column.",
        "retry_mode": "correct_and_retry",
        "fix": {
            "column": "observationAbout_type",
            "predicate_kind": "place_level",
            "actual_type": "untyped place entity",
        },
    }


async def test_predicate_operand_type_mismatch_refuses_as_a_typed_refusal(seeded) -> None:
    """An `ordered_threshold` predicate targeting `time_period` with a
    numeric operand -- the column is legal for this predicate kind, only
    the operand's own type is wrong -- reaches the caller as a typed
    refusal, not the `internal_error` backstop."""
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(
            {"kind": "ordered_threshold", "column": "time_period", "op": "gt", "value": 2015},
        ),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "filter_predicate_type_mismatch"
    assert result.structured_content["fix"]["column"] == "time_period"
    assert result.structured_content["fix"]["actual_type"] == "float"
    assert result.structured_content["fix"]["expected_type"] == "str"


async def test_oversized_time_period_operand_refuses_op_spec_invalid_without_echoing_it(
    seeded,
) -> None:
    """A string operand on `time_period` is capped well below the size of a
    legal period string, so an oversized one refuses at the op-spec layer
    before it ever reaches the engine, and the refusal's own message stays
    short rather than echoing the operand back in full."""
    _store, handles = seeded
    oversized = "9" * 100_000
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(
            {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": oversized},
        ),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"
    message = result.content[0].text
    assert len(message) < 500
    assert oversized not in message


@pytest.mark.parametrize("operand", ["hello", "2015"])
async def test_string_operand_against_value_reaches_the_engine_and_refuses_type_mismatch(
    seeded, operand: str
) -> None:
    """Neither `"hello"` nor `"2015"` is a well-formed period, but `value` --
    not `time_period` -- is the column here, so op-spec parsing has no
    reason to check either against the period grammar. Both pass op-spec
    validation and reach the engine's own type check, which refuses both
    the same way: `value`'s own declared cell type is numeric, and a string
    operand of any content is the wrong type for it."""
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(
            {"kind": "ordered_threshold", "column": "value", "op": "gte", "value": operand},
        ),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "filter_predicate_type_mismatch"
    assert result.structured_content["fix"]["column"] == "value"


async def test_malformed_period_against_time_period_still_refuses_at_the_op_spec_layer(
    seeded,
) -> None:
    """Unlike a string against `value`, a string against `time_period` is
    still checked against the period grammar at the op-spec layer -- a
    malformed one never reaches the engine at all."""
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(
            {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": "hello"},
        ),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_predicate_order_produces_a_byte_identical_refusal(seeded) -> None:
    """Two calls whose `predicates` differ only in order mint the identical
    `op_spec_hash` (`identity._normalise_parameters` already sorts
    `predicates` by canonical JCS bytes before hashing). `run_filter`
    validates predicates in the order it receives them, so without
    canonicalising first, which of two type-mismatched predicates is
    reported first -- and therefore the refusal's own `fix["column"]` --
    would depend on the caller's own raw order, even though both calls
    claim "same inputs, same result." Both calls canonicalise `predicates`
    the same way identity does before the engine ever runs, so the two
    refusals are identical regardless of which order the caller happened
    to supply them in."""
    _store, handles = seeded
    reporter_predicate = {
        "kind": "ordered_threshold",
        "column": "reporter",
        "op": "gte",
        "value": 1,
    }
    status_predicate = {
        "kind": "ordered_threshold",
        "column": "status",
        "op": "gte",
        "value": 1,
    }
    first = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(reporter_predicate, status_predicate),
    )
    second = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(status_predicate, reporter_predicate),
    )
    assert first.is_error is True
    assert second.is_error is True
    assert first.structured_content["code"] == second.structured_content["code"]
    assert first.structured_content["fix"] == second.structured_content["fix"]
    assert first.content == second.content


@pytest.mark.parametrize(
    ("unit", "actual_type", "select_one"),
    [
        ({"currency": "USD", "prices": "current"}, "UnitRequest", "get_data"),
        (
            {"numerator": "USD", "denominator": {"stat_var": "dcid:Count_Person"}},
            "ComposedUnit",
            "unit parameter",
        ),
    ],
    ids=["converted", "composed"],
)
async def test_category_match_on_unit_over_a_unit_object_frame_refuses_correct_and_retry(
    seeded, unit: dict[str, object], actual_type: str, select_one: str
) -> None:
    """A `unit` cell holding a `grammar.UnitRequest` or `grammar.ComposedUnit`
    object is not a plain string, so a `category_match` on `unit` matches no
    row. `filter_unit_conversion_unsupported` says so, and names the retries
    that succeed: dropping the unit predicate, or selecting one unit by a
    fetch (converted) or a pin (composed). A predicate on any other column
    filters the same frame."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:oda_disbursed",
                "reporter": "dcid:USA",
                "time_period": "2022",
                "value": 1.0,
                "unit": unit,
            },
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await filter_verb(
        frame_ref=custom_ref,
        predicates=({"kind": "category_match", "column": "unit", "values": ["USD"]},),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "filter_unit_conversion_unsupported"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"]["actual_type"] == actual_type
    message = result.content[0].text
    assert "Retry without the unit predicate" in message
    assert select_one in message

    kept = await filter_verb(
        frame_ref=custom_ref,
        predicates=({"kind": "category_match", "column": "reporter", "values": ["dcid:USA"]},),
    )
    assert kept.is_error is False


async def test_duplicate_predicate_refuses_op_spec_invalid_rather_than_crashing(seeded) -> None:
    """Two identical predicates pass op-spec parsing but `op_spec_hash_for`
    raises `OpSpecIdentityError` -- this must reach the tool boundary as a
    typed refusal, not an unhandled exception."""
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(
            {"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 1},
            {"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 1},
        ),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_scope_narrowing_predicate_shape_refuses_op_spec_invalid(seeded) -> None:
    """The filter exception: `predicates` has no field for a role, window,
    or facet binding -- a caller trying to smuggle one in through the
    predicate shape refuses at schema validation, not at the engine."""
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=({"kind": "role_binding", "column": "reporter", "value": "dcid:USA"},),
    )
    assert result.structured_content["code"] == "op_spec_invalid"


# A frame labelled with one span (what `aggregate` and `rank` write over several
# years) takes no period threshold; every other predicate filters as usual -----


async def _seed_span_frame(store: InMemoryMaterialisationStore) -> dict[str, str]:
    return await _seed_custom_frame(
        store,
        Frame.build(
            role_keys=["reporter"],
            rows=[
                {
                    "stat_var": "dcid:X",
                    "reporter": reporter,
                    "time_period": "2018/2022",
                    "value": value,
                    "unit": "dcid:USDollar",
                }
                for reporter, value in (("dcid:USA", 10.0), ("dcid:DEU", 5.0))
            ],
        ),
    )


@pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte", "eq", "ne"])
async def test_ordered_threshold_on_time_period_over_a_span_refuses_as_interval(
    seeded, op: str
) -> None:
    store, _handles = seeded
    result = await filter_verb(
        frame_ref=await _seed_span_frame(store),
        predicates=(
            {"kind": "ordered_threshold", "column": "time_period", "op": op, "value": "2020"},
        ),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "filter_predicate_type_mismatch"
    assert result.structured_content["fix"] == {
        "column": "time_period",
        "predicate_kind": "ordered_threshold",
        "actual_type": "interval",
    }


@pytest.mark.parametrize(
    ("predicates", "kept"),
    [
        pytest.param(
            ({"kind": "category_match", "column": "time_period", "values": ["2018/2022"]},),
            2,
            id="exact-span",
        ),
        pytest.param(
            ({"kind": "category_match", "column": "time_period", "values": ["2020"]},),
            0,
            id="other-period",
        ),
        pytest.param(
            ({"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 7.0},),
            1,
            id="value-threshold",
        ),
    ],
)
async def test_other_predicates_filter_a_span_frame_as_usual(
    seeded, predicates: tuple[dict[str, object], ...], kept: int
) -> None:
    store, _handles = seeded
    result = await filter_verb(frame_ref=await _seed_span_frame(store), predicates=predicates)
    assert result.is_error is False
    assert f"{kept}/2 row(s) kept" in result.content[0].text


# Empty result is not an error ---------------------------------------------------


async def test_empty_result_after_filtering_is_not_an_error(seeded) -> None:
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=({"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 1e20},),
    )
    assert result.is_error is False
    assert "0/15 row(s) kept" in result.content[0].text


# Success: one real analyst question end to end, through the registered tool ----


async def test_real_analyst_question_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Which of the seeded ODA observations exceed $50 billion?" via the
    live FastMCP `Client`, exercising the real registered tool under its
    wire name `filter`."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "filter",
            {
                "frame_ref": handles.plain_single_unit,
                "predicates": [
                    {
                        "kind": "ordered_threshold",
                        "column": "value",
                        "op": "gte",
                        "value": 50_000_000_000,
                    }
                ],
            },
        )

        assert result.is_error is False
        digest = result.content[0].text
        assert result.structured_content["spec_id"].startswith("ops_")
        assert result.structured_content["content_version"].startswith("ocv_")
        assert "2/15 row(s) kept" in digest
        assert "reporter=dcid:USA" in digest


# Category match on the unit column, through the registered tool ----------------


async def test_category_match_on_unit_narrows_a_heterogeneous_frame(seeded) -> None:
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.multi_unit,
        predicates=({"kind": "category_match", "column": "unit", "values": ["dcid:USDollar"]},),
    )
    assert result.is_error is False


# Replay + execution-identity perturbation ---------------------------------------


async def test_replay_from_execution_record_content_is_bit_identical(seeded) -> None:
    store, handles = seeded

    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(
            {"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 30_000_000_000},
        ),
    )
    assert result.is_error is False
    output_ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    stored_frame, stored_trace = await store.get(output_ref)

    canonical_bytes = stored_trace.execution_record.content.encode()
    assert recompute_op_spec_hash(canonical_bytes) == stored_trace.op_spec_hash

    # `predicates` round-trips as plain dicts through the canonical JSON, not
    # the pydantic model -- `run_filter` reads `.column`/`.kind`/etc via
    # attribute access, so replay parses them back through `FilterOpSpec`
    # rather than passing the raw dicts to `run_filter` directly.
    parameters = json.loads(stored_trace.execution_record.content)["parameters"]
    input_frame, _input_trace = await store.get(FrameRef(**handles.plain_single_unit))
    op_spec = FilterOpSpec(
        frame_ref=FrameRef(**handles.plain_single_unit), predicates=parameters["predicates"]
    )
    replayed = run_filter(input_frame, predicates=op_spec.predicates)
    assert replayed.frame == stored_frame

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

    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(
            {"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 30_000_000_000},
        ),
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


# Rolling overlay ----------------------------------------------------------------

_LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}
_ROOT = "spec_" + "c" * 32
_ANCHOR_KEY = "rak_" + "e" * 32


async def test_a_rolling_filter_keeps_its_fixed_twins_rows_under_another_pair(seeded) -> None:
    store, handles = seeded
    donors = {"kind": "category_match", "column": "reporter", "values": ["dcid:USA", "dcid:FRA"]}
    since = {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": "2022"}
    rolling_since = {**since, "rolling": {**_LATEST, "offset": -1}}
    root = handles.plain_single_unit
    binding = {"reporter": ["dcid:USA", "dcid:FRA"]}
    anchor = await anchored(root, bindings=binding)
    wider = await anchored(root, await second_version(store, root), bindings=binding)
    assert (anchor.period, wider.period) == ("2023", "2023")
    population = {"population": anchor.populations}

    fixed = await filter_verb(frame_ref=handles.plain_single_unit, predicates=(donors, since))
    rolled = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(donors, rolling_since),
        rolling=population,
        anchor_key=anchor.anchor_key,
    )
    rekeyed = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(donors, rolling_since),
        rolling=population,
        anchor_key=wider.anchor_key,
    )

    fixed_frame, fixed_trace = await store.get(FrameRef(**_pair(fixed)))
    rolled_frame, rolled_trace = await store.get(FrameRef(**_pair(rolled)))
    assert fixed_frame.table.num_rows > 0
    assert rolled_frame.table.equals(fixed_frame.table)
    assert rolled_trace.frame_content_version == fixed_trace.frame_content_version
    assert rolled_trace.execution_identity == fixed_trace.execution_identity
    assert rolled_trace.applied_policy == fixed_trace.applied_policy
    assert _pair(rolled)["spec_id"] != _pair(fixed)["spec_id"]
    assert _pair(rolled)["content_version"] != _pair(fixed)["content_version"]
    recorded = json.loads(rolled_trace.execution_record.content)["parameters"]
    assert rolling_since in recorded["predicates"]
    assert recorded["rolling"] == {
        "population": [
            {"spec_id": root["spec_id"], "bindings": {"reporter": ["dcid:FRA", "dcid:USA"]}}
        ]
    }
    assert wider.anchor_key != anchor.anchor_key
    assert _pair(rekeyed) == _pair(rolled)


@pytest.mark.parametrize(
    ("predicate", "rolling"),
    [
        pytest.param(
            {
                "kind": "ordered_threshold",
                "column": "value",
                "op": "gte",
                "value": 1,
                "rolling": _LATEST,
            },
            {"population": [{"spec_id": _ROOT}]},
            id="rolling-off-time-period",
        ),
        pytest.param(
            {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": "2022"},
            {"population": [{"spec_id": _ROOT}]},
            id="population-without-a-rolling-predicate",
        ),
    ],
)
async def test_a_misplaced_rolling_overlay_refuses_op_spec_invalid(
    seeded, predicate: dict[str, object], rolling: dict[str, object] | None
) -> None:
    _store, handles = seeded
    result = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(predicate,),
        rolling=rolling,
        anchor_key=_ANCHOR_KEY,
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


def _pair(result) -> dict[str, str]:
    content = result.structured_content
    return {"spec_id": content["spec_id"], "content_version": content["content_version"]}
