"""The `join` tool end to end: every refusal path, one real analyst
question through the registered tool, and the replay + execution-identity-
perturbation determinism proofs.
"""

from __future__ import annotations

import json
import secrets

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.engine.join import run_join
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.fixtures import multi_unit_frame, seed_fixture_store
from analysis_mcp.frame_ref_resolution import derive_frame_content_version, input_content_version
from analysis_mcp.identity import (
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.join_tool import join
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import (
    FrameRef,
    FrameTooLarge,
    GcsMaterialisationStore,
    InMemoryMaterialisationStore,
    set_store,
)
from dfo_frame_grammar import Frame
from fastmcp import Client
from google.api_core.exceptions import NotFound, PreconditionFailed


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
    base_trace = _minimal_trace()
    # `frame_content_version` is derived from `frame`'s own bytes, so two
    # calls seeding byte-identical frames produce the identical
    # `frame_content_version` -- the case
    # `test_two_pairs_resolving_to_the_same_input_cv_still_succeeds` below
    # exercises.
    trace = base_trace.model_copy(
        update={
            "frame_content_version": derive_frame_content_version(
                frame=frame, upstream_trace=base_trace.upstream_trace
            )
        }
    )
    await store.put(ref, frame=frame, trace=trace, writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


# Refusals ----------------------------------------------------------------------


async def test_op_spec_invalid_refuses(seeded) -> None:
    _store, _handles = seeded
    result = await join(frame_ref="not-a-valid-pair", right_frame_ref="also-not-valid")
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_unregistered_left_pair_refuses_frame_not_held(seeded) -> None:
    _store, handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await join(frame_ref=never_registered, right_frame_ref=handles.join_pair[1])
    assert result.structured_content["code"] == "frame_not_held"


async def test_unregistered_right_pair_refuses_frame_not_held(seeded) -> None:
    _store, handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await join(frame_ref=handles.join_pair[0], right_frame_ref=never_registered)
    assert result.structured_content["code"] == "frame_not_held"


async def test_unknown_explicit_join_key_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await join(
        frame_ref=handles.join_pair[0], right_frame_ref=handles.join_pair[1], join_keys=("donor",)
    )
    assert result.structured_content["code"] == "role_ambiguous"


async def test_ambiguous_key_refuses_role_ambiguous(seeded) -> None:
    store, _handles = seeded
    left = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 2.0,
                "measurementMethod": "dcid:M2",
            },
        ],
    )
    right = Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:Y", "reporter": "dcid:USA", "time_period": "2020", "value": 10.0}],
    )
    left_ref = await _seed_custom_frame(store, left)
    right_ref = await _seed_custom_frame(store, right)

    result = await join(frame_ref=left_ref, right_frame_ref=right_ref)
    assert result.structured_content["code"] == "role_ambiguous"


async def test_heterogeneous_unit_frame_refuses(seeded) -> None:
    _store, handles = seeded
    result = await join(frame_ref=handles.multi_unit, right_frame_ref=handles.multi_unit)
    assert result.structured_content["code"] == "unit_heterogeneous"


async def test_identical_sides_refuse_join_output_collision(seeded) -> None:
    """Joining a single-unit frame against itself: both sides resolve to
    the identical stat_var, unit, and (empty) facets, so every matched key's
    two emitted rows would be indistinguishable."""
    _store, handles = seeded
    result = await join(
        frame_ref=handles.plain_single_unit, right_frame_ref=handles.plain_single_unit
    )
    assert result.structured_content["code"] == "join_output_collision"
    assert (
        result.structured_content["fix"]["left_stat_var"]
        == result.structured_content["fix"]["right_stat_var"]
    )


async def test_duplicate_join_key_refuses_op_spec_invalid_rather_than_crashing(seeded) -> None:
    """A duplicate `join_keys` entry passes op-spec parsing but
    `op_spec_hash_for` raises `OpSpecIdentityError` -- this must reach the
    tool boundary as a typed refusal, not an unhandled exception."""
    _store, handles = seeded
    result = await join(
        frame_ref=handles.join_pair[0],
        right_frame_ref=handles.join_pair[1],
        join_keys=("reporter", "reporter"),
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


# Success: one real analyst question end to end, through the registered tool ----


async def test_real_analyst_question_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Show the USA's, GBR's and FRA's ODA disbursed alongside their GNI"
    via the live FastMCP `Client`, exercising the real registered tool."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "join",
            {"frame_ref": handles.join_pair[0], "right_frame_ref": handles.join_pair[1]},
        )

        assert result.is_error is False
        digest = result.content[0].text
        assert result.structured_content["spec_id"].startswith("ops_")
        assert result.structured_content["content_version"].startswith("ocv_")
        assert "6 matched key(s)" in digest
        assert "reporter=dcid:USA" in digest


# Replay + execution-identity perturbation ---------------------------------------


async def test_replay_from_execution_record_content_is_bit_identical(seeded) -> None:
    store, handles = seeded

    result = await join(frame_ref=handles.join_pair[0], right_frame_ref=handles.join_pair[1])
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
    replayed = run_join(
        left_frame,
        right_frame,
        join_keys=tuple(parameters["join_keys"]),
        left_pinned_unit=parameters["left_unit"],
        right_pinned_unit=parameters["right_unit"],
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

    result = await join(frame_ref=handles.join_pair[0], right_frame_ref=handles.join_pair[1])
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


async def test_two_pairs_resolving_to_the_same_input_cv_still_succeeds(seeded) -> None:
    """`left_cv`/`right_cv` are content-addressed (`frame_sha256` plus
    `upstream_trace`'s own lineage) -- two genuinely distinct pairs can
    still resolve to the identical `cv_`, unlike `join_output_collision`
    (raised by `run_join` itself only when a matched key's own stat_var,
    unit, and facets all agree), which this test's own two sides avoid by
    pinning to different units of the identical underlying frame.
    `identity._sorted_unique_strings` *rejects* a duplicate rather than
    deduping it, so without `join_tool.join`'s own `[left_cv] if right_cv ==
    left_cv else [left_cv, right_cv]` collapse, `mint_toolkit_content_
    version`'s `OpSpecIdentityError` would escape `join`'s own typed
    exception handling and only be caught by `server.py`'s registration-site
    backstop -- an `internal_error`, not a genuine refusal this call
    deserves. Seeding the *same* `multi_unit_frame()` content under two
    freshly registered pairs (rather than reusing one pair for both sides,
    like `test_identical_sides_refuse_join_output_collision` does) is what
    makes `left_cv == right_cv` while `frame_ref != right_frame_ref`."""
    store, _handles = seeded
    frame = multi_unit_frame()
    left_ref = await _seed_custom_frame(store, frame)
    right_ref = await _seed_custom_frame(store, frame)
    assert left_ref != right_ref

    _left_frame, left_trace = await store.get(FrameRef(**left_ref))
    _right_frame, right_trace = await store.get(FrameRef(**right_ref))
    left_cv = input_content_version(left_trace)
    right_cv = input_content_version(right_trace)
    assert left_cv == right_cv

    result = await join(
        frame_ref=left_ref,
        right_frame_ref=right_ref,
        left_unit="dcid:USDollar",
        right_unit="dcid:Percent",
    )

    assert result.is_error is False
    assert "spec_id" in result.structured_content
    assert "content_version" in result.structured_content


def test_description_names_compare_as_the_verb_for_two_entities() -> None:
    assert "compare" in join.__doc__


# Result over the durable store's own row ceiling ---------------------------------

# Duplicates `test_frame_store_restart.py`'s own fake `_StorageClient`/`_Bucket`/
# `_Blob` -- see that module's docstring for why a service's tests don't import
# another test module's fakes.
_JOIN_BUCKET = "dev-fin-observatory-frames"


class _FakeBlob:
    def __init__(
        self, objects: dict[str, bytes], metadata_store: dict[str, dict[str, str]], name: str
    ) -> None:
        self._objects = objects
        self._metadata_store = metadata_store
        self._name = name
        self.metadata: dict[str, str] | None = None
        self.size: int | None = None

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert if_generation_match == 0
        if self._name in self._objects:
            raise PreconditionFailed("already exists")
        self._objects[self._name] = data
        self._metadata_store[self._name] = dict(self.metadata or {})

    def download_as_bytes(self) -> bytes:
        if self._name not in self._objects:
            raise NotFound(f"no such object: {self._name}")
        return self._objects[self._name]

    def reload(self) -> None:
        if self._name not in self._objects:
            raise NotFound(f"no such object: {self._name}")
        self.metadata = dict(self._metadata_store.get(self._name, {}))
        self.size = len(self._objects[self._name])


class _FakeBucket:
    def __init__(
        self, objects: dict[str, bytes], metadata_store: dict[str, dict[str, str]]
    ) -> None:
        self._objects = objects
        self._metadata_store = metadata_store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._objects, self._metadata_store, name)


class _FakeStorageClient:
    def __init__(self, objects: dict[str, bytes], metadata: dict[str, dict[str, str]]) -> None:
        self._objects = objects
        self._metadata = metadata

    def bucket(self, name: str) -> _FakeBucket:
        assert name == _JOIN_BUCKET
        return _FakeBucket(self._objects, self._metadata)

    def close(self) -> None:
        pass


def _two_matched_key_frames() -> tuple[Frame, Frame]:
    """Two two-row frames matching on both keys: a four-row join output."""
    return tuple(
        Frame.build(
            role_keys=["reporter"],
            rows=[
                {
                    "stat_var": stat_var,
                    "reporter": "dcid:USA",
                    "time_period": str(2020 + offset),
                    "value": float(offset) * scale,
                    "unit": "dcid:USDollar",
                }
                for offset in range(2)
            ],
        )
        for stat_var, scale in (("dcid:OdaDisbursedAmount", 1), ("dcid:GrossNationalIncome", 10))
    )


async def test_result_over_the_store_row_ceiling_refuses_rather_than_minting_an_unreadable_ref() -> (
    None
):
    """`run_join` emits two output rows per matched key (one per side --
    see `analysis_mcp.engine.join`'s own module docstring), so two inputs
    that each pass the store's own row ceiling on their own can still
    produce a result that doesn't. Reproduces the bug: before the store-side
    write check, this would `put` succeed and hand back a `spec_id`/
    `content_version` pair that every later `get` could only ever refuse --
    unreadable forever, since writes are insert-only. `max_frame_rows=2`
    keeps this cheap: two two-row inputs joined on two matched keys already
    produce four output rows, no large fixture needed.
    """
    objects: dict[str, bytes] = {}
    metadata: dict[str, dict[str, str]] = {}
    store = GcsMaterialisationStore(
        bucket_name=_JOIN_BUCKET,
        client=_FakeStorageClient(objects, metadata),
        max_frame_rows=2,
    )
    left, right = _two_matched_key_frames()
    left_ref = FrameRef(spec_id="spec_" + "e" * 32, content_version="cv_" + "e" * 32)
    right_ref = FrameRef(spec_id="spec_" + "f" * 32, content_version="cv_" + "f" * 32)
    base_trace = _minimal_trace()
    await store.put(
        left_ref,
        frame=left,
        trace=base_trace.model_copy(
            update={
                "frame_content_version": derive_frame_content_version(
                    frame=left, upstream_trace=base_trace.upstream_trace
                )
            }
        ),
        writer="test",
    )
    await store.put(
        right_ref,
        frame=right,
        trace=base_trace.model_copy(
            update={
                "frame_content_version": derive_frame_content_version(
                    frame=right, upstream_trace=base_trace.upstream_trace
                )
            }
        ),
        writer="test",
    )
    object_count_before_join = len(objects)

    set_store(store)
    try:
        result = await join(
            frame_ref={"spec_id": left_ref.spec_id, "content_version": left_ref.content_version},
            right_frame_ref={
                "spec_id": right_ref.spec_id,
                "content_version": right_ref.content_version,
            },
        )
    finally:
        set_store(None)

    assert result.is_error is True
    assert result.structured_content["code"] == "frame_too_large"
    # No object landed for the refused result -- only the two seeded inputs
    # are still there.
    assert len(objects) == object_count_before_join


async def test_result_over_max_frame_rows_refuses_before_the_store_with_its_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`MAX_FRAME_ROWS` bounds the join's output before it is built, with
    the refusal the store's own row ceiling returns on write. The in-memory
    store has no ceiling, so only the early check can refuse here."""
    left, right = _two_matched_key_frames()
    ceiling_store = GcsMaterialisationStore(
        bucket_name=_JOIN_BUCKET, client=_FakeStorageClient({}, {}), max_frame_rows=3
    )
    with pytest.raises(FrameTooLarge) as store_refusal:
        await ceiling_store.put(
            FrameRef(spec_id="ops_" + "e" * 32, content_version="ocv_" + "e" * 32),
            frame=run_join(left, right).frame,
            trace=_minimal_trace(),
            writer="test",
        )

    store = InMemoryMaterialisationStore()
    left_ref = await _seed_custom_frame(store, left)
    right_ref = await _seed_custom_frame(store, right)
    monkeypatch.setenv("MAX_FRAME_ROWS", "3")
    set_store(store)
    try:
        result = await join(frame_ref=left_ref, right_frame_ref=right_ref)
    finally:
        set_store(None)

    assert result.is_error is True
    assert (
        result.structured_content
        == to_tool_result(
            ToolRefusal.frame_too_large_on_write(reason=store_refusal.value.reason)
        ).structured_content
    )
