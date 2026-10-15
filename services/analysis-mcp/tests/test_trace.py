"""The trace schema: opaque upstream mapping plus the toolkit's own fields."""

from __future__ import annotations

import pytest
from analysis_mcp.artifact import Observation
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.steps import LeftOut, Step, VersionRef
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, decode_trace, encode_trace
from pydantic import ValidationError

SNAPSHOT_REF = {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "b" * 32}


def _minimal_trace() -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(**SNAPSHOT_REF), group_by=("reporter",), aggregation="sum"
    )
    op_spec_hash = op_spec_hash_for(op_spec)
    ei = current_execution_identity()
    content_version = "ocv_" + "c" * 32
    return Trace(
        upstream_trace={"graph_revision": "rev1", "resolved_stat_vars": []},
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(kind="op_spec_canonical", content="{}"),
        applied_policy=AppliedPolicy(),
        content_version=content_version,
        input_content_versions=[SNAPSHOT_REF["content_version"]],
        execution_identity=ei,
    )


def test_trace_round_trips() -> None:
    trace = _minimal_trace()
    dumped = trace.model_dump(mode="json")
    assert Trace.model_validate(dumped) == trace


def test_upstream_trace_is_carried_as_an_opaque_mapping() -> None:
    """No field of `ManifestTrace` is redefined -- an arbitrary shaped mapping passes through."""
    trace = Trace(
        upstream_trace={
            "applied_window": {"start": "2015", "end": "2020"},
            "graph_revision": "rev1",
            "source_revisions": [{"source": "x", "revision": "1"}],
            "resolved_stat_vars": [
                {"ref": "dcid:X", "roles": ["reporter"], "observation_count": 1}
            ],
            "facets": [],
            "normalisations": [],
            "conversion": None,
            "frame_grammar_version": "1.0.0",
            "materialiser_build": "build1",
            "capability_version": "cap1",
        },
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash_for(
            AggregateOpSpec(frame_ref=FrameRef(**SNAPSHOT_REF), group_by=(), aggregation="sum")
        ),
        execution_record=ExecutionRecord(kind="op_spec_canonical", content="{}"),
        applied_policy=AppliedPolicy(),
        content_version="ocv_" + "c" * 32,
        input_content_versions=[SNAPSHOT_REF["content_version"]],
        execution_identity=current_execution_identity(),
    )
    assert trace.upstream_trace["graph_revision"] == "rev1"


def test_execution_record_kind_is_op_spec_canonical_not_rendered_sql() -> None:
    record = ExecutionRecord(content="{}")
    assert record.kind == "op_spec_canonical"


def test_applied_policy_defaults_are_all_none() -> None:
    """Defaults are unset; a verb populates them when it records its policy."""
    policy = AppliedPolicy()
    assert policy.policy_name is None
    assert policy.realised_coverage_fraction is None
    assert policy.skipped_row_count is None


def test_malformed_op_spec_hash_is_refused() -> None:
    with pytest.raises(ValidationError):
        Trace(
            upstream_trace={},
            frame_content_version=SNAPSHOT_REF["content_version"],
            op_spec_hash="not-a-hash",
            execution_record=ExecutionRecord(content="{}"),
            applied_policy=AppliedPolicy(),
            content_version="ocv_" + "c" * 32,
            input_content_versions=[SNAPSHOT_REF["content_version"]],
            execution_identity=current_execution_identity(),
        )


def test_malformed_content_version_is_refused() -> None:
    with pytest.raises(ValidationError):
        Trace(
            upstream_trace={},
            frame_content_version=SNAPSHOT_REF["content_version"],
            op_spec_hash=op_spec_hash_for(
                AggregateOpSpec(frame_ref=FrameRef(**SNAPSHOT_REF), group_by=(), aggregation="sum")
            ),
            execution_record=ExecutionRecord(content="{}"),
            applied_policy=AppliedPolicy(),
            content_version="cv_" + "c" * 32,  # frame-grammar's family, not the toolkit's ocv_
            input_content_versions=[SNAPSHOT_REF["content_version"]],
            execution_identity=current_execution_identity(),
        )


def test_malformed_input_content_version_is_refused() -> None:
    with pytest.raises(ValidationError):
        Trace(
            upstream_trace={},
            frame_content_version=SNAPSHOT_REF["content_version"],
            op_spec_hash=op_spec_hash_for(
                AggregateOpSpec(frame_ref=FrameRef(**SNAPSHOT_REF), group_by=(), aggregation="sum")
            ),
            execution_record=ExecutionRecord(content="{}"),
            applied_policy=AppliedPolicy(),
            content_version="ocv_" + "c" * 32,
            input_content_versions=["not-a-content-version"],
            execution_identity=current_execution_identity(),
        )


def test_malformed_frame_content_version_is_refused() -> None:
    with pytest.raises(ValidationError):
        Trace(
            upstream_trace={},
            frame_content_version="not-a-content-version",
            op_spec_hash=op_spec_hash_for(
                AggregateOpSpec(frame_ref=FrameRef(**SNAPSHOT_REF), group_by=(), aggregation="sum")
            ),
            execution_record=ExecutionRecord(content="{}"),
            applied_policy=AppliedPolicy(),
            content_version="ocv_" + "c" * 32,
            input_content_versions=[SNAPSHOT_REF["content_version"]],
            execution_identity=current_execution_identity(),
        )


def test_unknown_field_refused() -> None:
    with pytest.raises(ValidationError):
        AppliedPolicy(not_a_real_field=1)  # type: ignore[call-arg]


def test_trace_round_trips_through_the_common_trace_codec() -> None:
    """`Trace` registers with `common.materialisation`'s codec registry at
    import time (`analysis_mcp.trace`'s own registration call), so a
    durable store built on that registry can persist and recover a `Trace`
    without either module depending on the other's concrete type."""
    trace = _minimal_trace()

    record = encode_trace(trace)

    assert record["kind"] == "analysis_mcp_trace"
    decoded = decode_trace(record)
    assert isinstance(decoded, Trace)
    assert decoded == trace


def test_decode_trace_defaults_a_record_written_before_exclusions_and_time_existed() -> None:
    """A record an earlier deploy wrote, before `Trace` grew `exclusions` and
    `time`, still decodes -- `decode_trace` fills both from their own defaults."""
    record = encode_trace(_minimal_trace())
    del record["trace"]["exclusions"]
    del record["trace"]["time"]

    decoded = decode_trace(record)

    assert isinstance(decoded, Trace)
    assert decoded.exclusions == ()
    assert decoded.time == Observation()


def test_decode_trace_ignores_a_member_no_field_names() -> None:
    """A record carrying a member no field on `Trace` names decodes too --
    the lenient decode shape drops it rather than refusing the whole record."""
    trace = _minimal_trace()
    record = encode_trace(trace)
    record["trace"]["a_future_deploy_added_this"] = "unrecognised"

    decoded = decode_trace(record)

    assert isinstance(decoded, Trace)
    assert decoded == trace


def test_decode_trace_reads_a_record_stored_with_composed_stat_vars() -> None:
    """A record stored while `Trace` carried `composed_stat_vars` decodes, without
    the member."""
    trace = _minimal_trace()
    record = encode_trace(trace)
    record["trace"]["composed_stat_vars"] = [
        {
            "stat_var": "dcid:ops_" + "e" * 32,
            "relation": "quotient",
            "first": ["dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"],
            "second": ["dcid:Count_Person"],
        }
    ]

    decoded = decode_trace(record)

    assert isinstance(decoded, Trace)
    assert decoded == trace


_QUOTIENT_STEP = Step(
    id="ops_" + "e" * 32,
    op="quotient",
    inputs={
        "numerator": VersionRef(id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
        "denominator": VersionRef(id="ops_" + "f" * 32, content_version="ocv_" + "f" * 32),
    },
    policy="quotient_of_intersect_periods",
    params={"numerator_by": ["recipient"]},
    kept={"entities": 38, "periods": 15},
    left_out=[LeftOut(reason="unmatched_key", count=9, of="entities")],
    unit=None,
)


def test_steps_round_trip_through_the_codec_with_only_the_members_they_hold() -> None:
    trace = _minimal_trace().model_copy(update={"steps": (_QUOTIENT_STEP,)})

    record = encode_trace(trace)
    decoded = decode_trace(record)

    assert set(record["trace"]["steps"][0]) == {
        "id",
        "op",
        "inputs",
        "policy",
        "params",
        "kept",
        "left_out",
        "unit",
    }
    assert record["trace"]["steps"][0]["unit"] is None
    assert decoded == trace


def test_decode_trace_reads_a_record_stored_without_steps_as_none() -> None:
    record = encode_trace(_minimal_trace())
    del record["trace"]["steps"]

    decoded = decode_trace(record)

    assert isinstance(decoded, Trace)
    assert decoded.steps == ()


@pytest.mark.parametrize(
    ("role", "accepted"),
    [("financingProvider", True), ("financing_provider", False), ("FinancingProvider", False)],
)
def test_a_pooled_count_names_a_lower_camel_case_role(role: str, accepted: bool) -> None:
    """The role pattern is the kit's, so a role apis' door would refuse is refused
    here first."""
    entry = {"reason": "party_total", "count": 3, "of": "pooled_entities", "role": role}
    if accepted:
        assert LeftOut.model_validate(entry).role == role
    else:
        with pytest.raises(ValidationError):
            LeftOut.model_validate(entry)
