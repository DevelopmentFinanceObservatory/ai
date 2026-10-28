"""The `aggregate` tool end to end: every refusal path, the coverage floor,
one real analyst question through the registered tool, and the replay +
execution-identity-perturbation determinism proofs.

Most tests call the plain `aggregate(...)` async function directly -- it is
the exact function object `server.py` registers via `mcp.tool(aggregate)`,
so this is "direct tool-fn call" per the repo's own test conventions, not a
stand-in for it. One test (`test_real_analyst_question_...`) goes through
the full FastMCP `Client` dispatch instead, to prove the registered tool
itself (schema, dispatch, response envelope) works end to end.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.artifact import Interval, Observation
from analysis_mcp.compare_tool import compare
from analysis_mcp.engine.aggregate import run_aggregate
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.fixtures import plain_single_unit_frame, seed_fixture_store
from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from analysis_mcp.identity import (
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import (
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    HeldFrame,
    InMemoryMaterialisationStore,
    set_store,
)
from common.materialisation.memory_store import FRAME_TTL
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
    result = await aggregate(frame_ref="not-a-valid-pair", aggregation="sum")
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_malformed_frame_ref_refuses_op_spec_invalid(seeded) -> None:
    _store, _handles = seeded
    ref = {"spec_id": "not-a-valid-spec-id", "content_version": "not-a-valid-cv"}
    result = await aggregate(frame_ref=ref, aggregation="sum")
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_unregistered_pair_refuses_frame_not_held(seeded) -> None:
    _store, _handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await aggregate(frame_ref=never_registered, aggregation="sum")
    assert result.structured_content["code"] == "frame_not_held"


async def test_duplicate_group_by_column_refuses_op_spec_invalid_rather_than_crashing(
    seeded,
) -> None:
    """A duplicate `group_by` entry passes op-spec parsing (no uniqueness
    constraint there) but `op_spec_hash_for` raises `OpSpecIdentityError` --
    this must reach the tool boundary as a typed refusal, not an unhandled
    exception escaping the async tool function."""
    _store, handles = seeded
    result = await aggregate(
        frame_ref=handles.plain_single_unit,
        aggregation="sum",
        # `stat_var` is constant across this fixture, so duplicating it in
        # group_by still clears the coverage floor (expected == present == 1)
        # -- the only refusal this call can hit is the identity guard.
        group_by=("stat_var", "stat_var"),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_group_by_order_produces_byte_identical_digests(seeded) -> None:
    """Two calls whose `group_by` differ only in order mint the identical
    `op_spec_hash`/`content_version` (`identity._normalise_parameters` sorts
    `group_by` before hashing). Without canonicalising execution to match,
    the header text and each group's displayed key order would still differ
    between the two calls despite the identical identity. Both calls
    canonicalise `group_by` before the engine runs, so the digests are
    byte-identical."""
    _store, handles = seeded
    first = await aggregate(
        frame_ref=handles.plain_single_unit,
        aggregation="sum",
        group_by=("reporter", "time_period"),
    )
    second = await aggregate(
        frame_ref=handles.plain_single_unit,
        aggregation="sum",
        group_by=("time_period", "reporter"),
    )
    assert first.is_error is False
    assert second.is_error is False
    assert first.content[0].text == second.content[0].text


async def test_expired_frame_refuses_frame_expired() -> None:
    clock = iter(
        [
            datetime(2026, 1, 1, tzinfo=UTC),  # at put()
            datetime(2026, 1, 1, tzinfo=UTC) + FRAME_TTL + timedelta(seconds=1),  # at get()
        ]
    )
    store = InMemoryMaterialisationStore(now=lambda: next(clock))
    ref = FrameRef(spec_id="spec_" + "d" * 32, content_version="cv_" + "e" * 32)
    await store.put(ref, frame=plain_single_unit_frame(), trace=_minimal_trace(), writer="test")
    set_store(store)
    try:
        result = await aggregate(
            frame_ref={"spec_id": ref.spec_id, "content_version": ref.content_version},
            aggregation="sum",
        )
    finally:
        set_store(None)
    assert result.structured_content["code"] == "frame_expired"


async def test_integrity_failed_frame_refuses_frame_integrity_failed() -> None:
    """A store raising `FrameIntegrityError` (the durable, GCS-backed store's
    own read-time bytes check, exercised directly against a fake bucket in
    `packages/common/tests/test_gcs_store.py`) must reach the tool boundary
    as `frame_integrity_failed`, not as an unhandled exception."""

    class _CorruptedStore:
        async def open(self, ref: FrameRef) -> HeldFrame | None:
            raise FrameIntegrityError(
                ref, reason="stored frame bytes do not match the recorded sha256"
            )

        async def put(self, ref: FrameRef, *, frame: Frame, trace: Trace, writer: str) -> None:
            raise NotImplementedError

        async def aclose(self) -> None:
            pass

    set_store(_CorruptedStore())
    try:
        result = await aggregate(
            frame_ref={"spec_id": "spec_" + "f" * 32, "content_version": "cv_" + "0" * 32},
            aggregation="sum",
        )
    finally:
        set_store(None)
    assert result.structured_content["code"] == "frame_integrity_failed"
    assert result.structured_content["fix"]["reason"] == (
        "stored frame bytes do not match the recorded sha256"
    )


async def test_store_ceiling_refuses_frame_too_large_before_decoding() -> None:
    """A store raising `FrameTooLarge` (the durable, GCS-backed store's own
    pre-decode size/row/column ceilings, exercised directly against a fake
    bucket in `packages/common/tests/test_gcs_store.py`) must reach the
    tool boundary as `frame_too_large` -- distinct from
    `frame_integrity_failed`, since a narrower `frame_ref` can succeed
    where retrying the identical one cannot."""

    class _OverCeilingStore:
        async def open(self, ref: FrameRef) -> HeldFrame | None:
            raise FrameTooLarge(ref, reason="stored object is 999 bytes, over the 10-byte ceiling")

        async def put(self, ref: FrameRef, *, frame: Frame, trace: Trace, writer: str) -> None:
            raise NotImplementedError

        async def aclose(self) -> None:
            pass

    set_store(_OverCeilingStore())
    try:
        result = await aggregate(
            frame_ref={"spec_id": "spec_" + "f" * 32, "content_version": "cv_" + "0" * 32},
            aggregation="sum",
        )
    finally:
        set_store(None)
    assert result.structured_content["code"] == "frame_too_large"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"]["reason"] == (
        "stored object is 999 bytes, over the 10-byte ceiling"
    )


async def test_frame_too_large_refuses(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, handles = seeded
    monkeypatch.setenv("MAX_FRAME_ROWS", "3")  # plain_single_unit_frame has 15 rows
    result = await aggregate(frame_ref=handles.plain_single_unit, aggregation="sum")
    assert result.structured_content["code"] == "frame_too_large"
    assert result.structured_content["fix"]["ceiling"] == 3


@pytest.mark.parametrize(
    ("periods", "grains", "named"),
    [
        pytest.param(("2020", "2021-01"), ["year", "month"], ["2020", "2021-01"], id="grain-mix"),
        pytest.param(("2018/2022", "2023"), ["year"], ["2018/2022", "2023"], id="span-and-point"),
        pytest.param(("2018/2022", "2019/2023"), ["year"], ["2018/2022", "2019/2023"], id="spans"),
    ],
)
async def test_mixed_time_grain_refuses_instead_of_crashing(
    seeded, periods: tuple[str, str], grains: list[str], named: list[str]
) -> None:
    """A frame mixing an annual and a monthly `time_period` reaches
    `_expected_periods`'s own grain-arithmetic with two different lengths
    to infer a grain from -- without the admission check, this raised an
    untyped `ValueError` that `_with_internal_error_backstop` only caught
    as `internal_error`. A span beside a point, or two different spans,
    has no reading as one set of points or one window either. Each must
    surface as the typed refusal naming the periods.
    """
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": period, "value": 10.0}
            for period in periods
        ],
    )
    ref = await _seed_custom_frame(store, frame)
    result = await aggregate(frame_ref=ref, aggregation="sum", group_by=("reporter",))
    assert result.is_error is True
    assert result.structured_content["code"] == "mixed_time_grain"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    assert result.structured_content["fix"] == {"grains": grains, "periods": named}


async def test_calendar_invalid_day_grain_frame_aggregates_instead_of_crashing(seeded) -> None:
    """A day-grain frame whose `time_period` carries a calendar-invalid date
    (`dfo_frame_grammar.grammar` deliberately admits a source's own
    out-of-range day, e.g. 30 February -- the source is the authority on its
    own dates) must still aggregate successfully. `build_interval`'s
    calendar arithmetic used to raise an untyped `ValueError` that
    `_with_internal_error_backstop` only caught as `internal_error` for
    legitimate source data; contiguity now degrades to unknown instead."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2024-02-28",
                "value": 10.0,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2024-02-30",
                "value": 20.0,
                "unit": "dcid:USDollar",
            },
        ],
    )
    ref = await _seed_custom_frame(store, frame)
    result = await aggregate(frame_ref=ref, aggregation="sum", group_by=("reporter",))
    assert result.is_error is False
    assert result.structured_content.get("code") != "internal_error"
    assert "contiguity is unknown" in result.content[0].text


@pytest.mark.parametrize(
    ("periods", "label", "time"),
    [
        pytest.param(
            ("2018", "2022"),
            "2018/2022",
            Interval(
                start="2018",
                end="2022",
                contiguous=False,
                absent=("2019", "2020", "2021"),
                absent_count=3,
            ),
            id="points-one-per-group",
        ),
        pytest.param(("2018/2022", "2018/2022"), "2018/2022", Observation(), id="one-span"),
        pytest.param(("2020", "2020"), "2020", Observation(), id="one-point"),
    ],
)
async def test_sum_stamps_the_call_wide_label_and_builds_an_interval_only_over_points(
    seeded, periods: tuple[str, str], label: str, time: Interval | Observation
) -> None:
    """A sum whose groups each hold one period still collapses the call's
    points into one span, so the trace carries its own interval. A span
    input passes through and the trace inherits the input's time."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": reporter, "time_period": period, "value": 1.0}
            for reporter, period in zip(("dcid:USA", "dcid:GBR"), periods, strict=True)
        ],
    )
    ref = await _seed_custom_frame(store, frame)
    result = await aggregate(frame_ref=ref, aggregation="sum", group_by=("reporter",))
    assert result.is_error is False
    stored_frame, stored_trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    time_idx = stored_frame.columns.index("time_period")
    assert [row[time_idx] for row in stored_frame.rows] == [label, label]
    assert stored_trace.time == time
    header = result.content[0].text.splitlines()[0]
    assert header.endswith(f"; every row's time_period is {label}")


async def test_unsupported_aggregation_refuses_operation_uncovered(seeded) -> None:
    _store, handles = seeded
    result = await aggregate(frame_ref=handles.plain_single_unit, aggregation="median")
    assert result.structured_content["code"] == "operation_uncovered"


async def test_unknown_group_by_column_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="sum", group_by=("donor",)
    )
    assert result.structured_content["code"] == "role_ambiguous"


async def test_heterogeneous_unit_frame_refuses_with_disambiguation_candidates(seeded) -> None:
    _store, handles = seeded
    result = await aggregate(
        frame_ref=handles.multi_unit, aggregation="sum", group_by=("reporter",)
    )
    assert result.structured_content["code"] == "unit_heterogeneous"
    candidates = result.structured_content["fix"]["disambiguation"]["candidates"]
    assert {c["id"] for c in candidates} == {"dcid:Percent", "dcid:USDollar"}


async def test_unit_disambiguation_retry_succeeds(seeded) -> None:
    _store, handles = seeded
    refusal = await aggregate(
        frame_ref=handles.multi_unit, aggregation="sum", group_by=("reporter",)
    )
    candidate = refusal.structured_content["fix"]["disambiguation"]["candidates"][0]["id"]

    retried = await aggregate(
        frame_ref=handles.multi_unit, aggregation="sum", group_by=("reporter",), unit=candidate
    )
    assert retried.is_error is False
    assert f"unit: {candidate}" in retried.content[0].text


async def test_unit_pin_against_a_frame_with_no_declared_units_refuses_validly(seeded) -> None:
    """Every row leaves `unit` unbound (`None`) -- `resolve_unit_and_facets`
    resolves `declared_units` to `[]`, so a `unit` pin retry raises
    `UnitNotPresentError` with an empty `units` list. Constructing the
    refusal's `disambiguation` payload from zero candidates would raise
    inside refusal construction itself if not guarded; this proves the
    refusal comes back valid instead, with `disambiguation` simply absent.
    """
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:X", "reporter": "dcid:GBR", "time_period": "2020", "value": 2.0},
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)

    result = await aggregate(frame_ref=custom_ref, aggregation="sum", unit="dcid:USDollar")

    assert result.is_error is True
    assert result.structured_content["code"] == "unit_heterogeneous"
    assert result.structured_content["fix"]["units"] == []
    assert "disambiguation" not in result.structured_content["fix"]


async def test_facet_heterogeneous_frame_refuses(seeded) -> None:
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 1.0,
                "provenance": "dcid:OECD_DAC2A",
            },
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:GBR",
                "time_period": "2020",
                "value": 2.0,
                "provenance": "dcid:OECD_DAC1",
            },
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await aggregate(frame_ref=custom_ref, aggregation="sum", group_by=("reporter",))
    assert result.structured_content["code"] == "facet_heterogeneous"


async def test_multiple_stat_vars_refuse_stat_var_unresolved(seeded) -> None:
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:A", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:B", "reporter": "dcid:USA", "time_period": "2020", "value": 2.0},
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await aggregate(frame_ref=custom_ref, aggregation="sum", group_by=("reporter",))
    assert result.structured_content["code"] == "stat_var_unresolved"


async def test_non_finite_result_refuses(seeded) -> None:
    store, _handles = seeded
    near_max = 1.0e308
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": near_max,
            },
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2021",
                "value": near_max,
            },
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await aggregate(frame_ref=custom_ref, aggregation="sum")
    assert result.structured_content["code"] == "non_finite_result"


async def test_coverage_below_the_configured_floor_refuses(
    seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store, handles = seeded
    monkeypatch.setenv("AGGREGATE_COVERAGE_FLOOR", "0.9")  # sparse_coverage_frame realises 0.75
    result = await aggregate(
        frame_ref=handles.sparse_coverage,
        aggregation="sum",
        group_by=("recipient", "time_period"),
    )
    assert result.structured_content["code"] == "coverage_insufficient"
    assert result.structured_content["fix"]["realised_coverage_fraction"] == pytest.approx(0.75)


# max / min -----------------------------------------------------------------------


async def test_max_aggregation_names_the_winning_period_in_the_digest(seeded) -> None:
    _store, handles = seeded
    result = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="max", group_by=("reporter",)
    )
    assert result.is_error is False
    digest = result.content[0].text
    assert "aggregate: max grouped by reporter" in digest
    assert any(f"({year})" in digest for year in ("2021", "2022", "2023"))


# Success: one real analyst question end to end, through the registered tool ----


async def test_real_analyst_question_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Total ODA disbursed by donor across 2021-2023" -- the years the seeded
    fixture actually covers -- via the live FastMCP `Client`, exercising the
    real registered tool's dispatch and response envelope, not just the bare
    function."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "aggregate",
            {
                "frame_ref": handles.plain_single_unit,
                "aggregation": "sum",
                "group_by": ["reporter"],
            },
        )

        assert result.is_error is False
        digest = result.content[0].text
        assert result.structured_content["spec_id"].startswith("ops_")
        assert result.structured_content["content_version"].startswith("ocv_")
        # Every number is engine-computed, from `plain_single_unit_frame`'s own
        # fixture data (see `analysis_mcp.fixtures`'s `_ODA_DISBURSED_BY_DONOR_YEAR`).
        assert "reporter=dcid:USA" in digest
        assert "166683000000" in digest  # USA: 47_805 + 55_303 + 63_575 (billions)
        assert "unit: dcid:USDollar" in digest
        assert "coverage: 100.0%" in digest

        # The store's own idempotent-put, session-independent `get` (see
        # `common.materialisation.MaterialisationStore`) means the freshly
        # registered pair resolves for any caller. A second call in this
        # same `Client` session, naming the previous call's own pair as its
        # own input, proves a chained toolkit result actually resolves as a
        # downstream verb's `frame_ref`.
        follow_on = await client.call_tool(
            "aggregate",
            {
                "frame_ref": {
                    "spec_id": result.structured_content["spec_id"],
                    "content_version": result.structured_content["content_version"],
                },
                "aggregation": "sum",
            },
        )
        assert follow_on.is_error is False


# Replay + execution-identity perturbation ---------------------------------------


async def test_replay_from_execution_record_content_is_bit_identical(seeded) -> None:
    """The recompute half of this test works from `execution_record.content`
    alone, parsed as plain JSON, with `op_spec_hash`/`content_version`
    recomputed via the `recompute_*` functions the trace schema documents
    for exactly this purpose -- no `AggregateOpSpec` model in the loop."""
    store, handles = seeded

    result = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="sum", group_by=("reporter",)
    )
    assert result.is_error is False
    output_ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    stored_frame, stored_trace = await store.get(output_ref)

    # Recompute op_spec_hash from the trace's own canonical bytes, no OpSpec model.
    canonical_bytes = stored_trace.execution_record.content.encode()
    assert recompute_op_spec_hash(canonical_bytes) == stored_trace.op_spec_hash

    # Recompute the aggregate itself from the raw parsed parameters + the
    # original input frame (fetched fresh, not through the stored output).
    parameters = json.loads(stored_trace.execution_record.content)["parameters"]
    input_frame, _input_trace = await store.get(FrameRef(**handles.plain_single_unit))
    replayed = run_aggregate(
        input_frame,
        group_by=tuple(parameters["group_by"]),
        aggregation=parameters["aggregation"],
        pinned_unit=parameters["unit"],
    )
    assert replayed.frame == stored_frame

    # Distinct from the recompute-from-parameters replay above. Proves the
    # *stored* frame_content_version was derived over the stored output
    # frame, not the input frame's bytes.
    assert derive_frame_content_version(frame=stored_frame) == stored_trace.frame_content_version

    # Recompute content_version (ocv_) from the trace's own identity fields
    # alone -- `input_content_versions` is the trace's own record of what it
    # was minted from, not re-derived from the input frame's bytes.
    recomputed_content_version = recompute_toolkit_content_version(
        op_spec_hash=stored_trace.op_spec_hash,
        input_content_versions=list(stored_trace.input_content_versions),
        execution_identity=stored_trace.execution_identity,
    )
    assert recomputed_content_version == stored_trace.content_version


async def test_execution_identity_perturbation_moves_the_content_version(seeded) -> None:
    """Proves the identity binds behaviour: replaying a real call's own
    `op_spec_hash` and input content_version under a *different*
    `reduction_policy_version` mints a different `content_version`."""
    store, handles = seeded

    result = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="sum", group_by=("reporter",)
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


async def test_max_over_multiple_periods_writes_no_marker_and_reaches_compare(seeded) -> None:
    """A `max` reduction over more than one period per group writes no
    marker: the output row's own `time_period` names the period whose value
    actually won the reduction, a genuine observation rather than a
    borrowed span label. A later `compare` call against a
    frame that shares that winning period succeeds instead of refusing."""
    store, handles = seeded
    maxed = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="max", group_by=("reporter",)
    )
    assert maxed.is_error is False
    maxed_ref = {
        "spec_id": maxed.structured_content["spec_id"],
        "content_version": maxed.structured_content["content_version"],
    }
    _frame, trace = await store.get(FrameRef(**maxed_ref))
    assert "derived" not in trace.applied_policy.detail

    # Every donor's own max in `plain_single_unit_frame` falls in 2023 (see
    # `fixtures.py`'s own values), the same year `handles.plain_single_unit`
    # itself carries for `dcid:USA` -- so the two sides share a period to
    # compare over.
    result = await compare(
        frame_ref=maxed_ref,
        left_entity={"reporter": "dcid:USA"},
        right_frame_ref=handles.plain_single_unit,
        right_entity={"reporter": "dcid:USA"},
    )

    assert result.is_error is False
    digest = result.content[0].text
    assert "1 intersected period(s)" in digest
    assert "2023:" in digest
