"""`frame_ref_resolution`: the derived-ness marker reader
(`read_derived_marker`, `DerivedMarker`).

Also covers this module's own import-time side effect: registering
`derive_frame_content_version` into `common.materialisation.content_version`'s
shared registry, mirroring `analysis_mcp.trace`'s own `register_trace_codec`
call and its own `test_trace_round_trips_through_the_common_trace_codec`.
"""

from __future__ import annotations

from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.frame_ref_resolution import (
    DerivedMarker,
    derive_frame_content_version,
    read_derived_marker,
)
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, get_content_version_deriver
from dfo_frame_grammar import Frame

_SNAPSHOT_REF = {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "b" * 32}


def _trace_with_detail(detail: dict[str, object]) -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(**_SNAPSHOT_REF), group_by=("reporter",), aggregation="sum"
    )
    return Trace(
        upstream_trace={"graph_revision": "rev1", "resolved_stat_vars": []},
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash_for(op_spec),
        execution_record=ExecutionRecord(kind="op_spec_canonical", content="{}"),
        applied_policy=AppliedPolicy(detail=detail),
        content_version="ocv_" + "c" * 32,
        input_content_versions=[_SNAPSHOT_REF["content_version"]],
        execution_identity=current_execution_identity(),
    )


def test_read_derived_marker_is_none_when_the_input_carries_no_marker() -> None:
    assert read_derived_marker(_trace_with_detail({})) is None


def test_read_derived_marker_parses_every_field() -> None:
    trace = _trace_with_detail(
        {
            "derived": {
                "dimensionless": True,
                "inherited_stat_var": True,
                "ordinal": True,
                "reason": "rank positions",
            }
        }
    )
    assert read_derived_marker(trace) == DerivedMarker(
        dimensionless=True, inherited_stat_var=True, ordinal=True, reason="rank positions"
    )


def test_read_derived_marker_defaults_absent_fields_to_false() -> None:
    marker = read_derived_marker(_trace_with_detail({"derived": {"dimensionless": True}}))
    assert marker == DerivedMarker(dimensionless=True)


def test_read_derived_marker_reads_a_stored_trace_carrying_retired_keys() -> None:
    """A trace stored before `spans_periods`/`rank_relabelled` retired still
    reads, keeping the flags this toolkit still knows."""
    trace = _trace_with_detail(
        {
            "derived": {
                "spans_periods": True,
                "rank_relabelled": True,
                "inherited_stat_var": True,
                "reason": "spans 2021..2023",
            }
        }
    )
    assert read_derived_marker(trace) == DerivedMarker(
        inherited_stat_var=True, reason="spans 2021..2023"
    )


def _frame_with_periods(*periods: str) -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": period, "value": 1.0}
            for period in periods
        ],
    )


def test_derive_frame_content_version_is_registered_as_the_stores_deriver() -> None:
    """Importing this module (transitively, as every verb tool already does) registers
    `derive_frame_content_version` into `common.materialisation.content_version`'s
    registry -- `common.materialisation.build_store` resolves this same registered
    deriver when a service asks for `verify_content_version_on_read`, regardless of
    which service's own lifespan builds the store."""
    registered = get_content_version_deriver()
    assert registered is not None

    frame = Frame.build(
        role_keys=[],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "time_period": "2020",
                "value": 100.0,
                "unit": None,
            }
        ],
    )

    assert registered(frame) == derive_frame_content_version(frame=frame)
