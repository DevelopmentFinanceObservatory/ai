"""`build_op_spec_or_refusal`: a `pydantic.ValidationError` from a verb's own
op-spec constructor reaches the caller as a plain, per-error detail --
never pydantic's own class-name header or its `errors.pydantic.dev` link.

`store_result_and_render`: the returned `structured_content` is a real
`result.VerbResult`, with `role_keys`/`stat_var_count`/`rows` read off
the frame it stores, not the trace.
"""

from __future__ import annotations

from typing import Any

import pytest
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.frame_ref_resolution import holds_input_rows
from analysis_mcp.identity import mint_toolkit_content_version, op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.tool_pipeline import build_op_spec_or_refusal, store_result_and_render
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult

_OVERLONG_CONTENT_VERSION = "cv_" + "0" * 33


def _build_with_overlong_content_version() -> AggregateOpSpec:
    return AggregateOpSpec(
        frame_ref={"spec_id": "spec_" + "a" * 32, "content_version": _OVERLONG_CONTENT_VERSION},
        group_by=(),
        aggregation="sum",
    )


def test_op_spec_invalid_detail_names_the_field_without_the_pydantic_class_or_link() -> None:
    result = build_op_spec_or_refusal(_build_with_overlong_content_version)

    assert result.is_error is True
    detail = result.structured_content["fix"]["detail"]
    assert "frame_ref.content_version" in detail
    assert "pattern" in detail
    assert "AggregateOpSpec" not in detail
    assert "errors.pydantic.dev" not in detail


def test_op_spec_invalid_detail_carries_a_short_offending_input() -> None:
    result = build_op_spec_or_refusal(_build_with_overlong_content_version)

    detail = result.structured_content["fix"]["detail"]
    assert _OVERLONG_CONTENT_VERSION in detail


# `store_result_and_render` --------------------------------------------------


@holds_input_rows
async def _store_and_render(**kwargs: Any) -> ToolResult:
    """`store_result_and_render` inside a verb call's row claim, as every verb runs it."""
    return await store_result_and_render(**kwargs)


@pytest.fixture
async def _store():
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


def _trace(*, group_by: tuple[str, ...] = ()) -> Trace:
    """A minimal, valid `Trace` -- `group_by` varies the op-spec so distinct
    calls in one test mint distinct `op_spec_hash`/`content_version` pairs,
    each writable under its own store key.
    """
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
        group_by=group_by,
        aggregation="sum",
    )
    op_spec_hash = op_spec_hash_for(op_spec)
    execution_identity = current_execution_identity()
    input_cv = "cv_" + "b" * 32
    return Trace(
        upstream_trace={"graph_revision": "rev1"},
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content="{}"),
        applied_policy=AppliedPolicy(),
        content_version=mint_toolkit_content_version(
            op_spec_hash=op_spec_hash,
            input_content_versions=[input_cv],
            execution_identity=execution_identity,
        ),
        input_content_versions=[input_cv],
        execution_identity=execution_identity,
    )


async def test_row_count_equals_the_stored_frames_own_row_count(_store) -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:x",
                "reporter": "dcid:USA",
                "time_period": "2022",
                "value": 1.0,
            },
            {
                "stat_var": "dcid:x",
                "reporter": "dcid:GBR",
                "time_period": "2022",
                "value": 2.0,
            },
        ],
    )

    result = await _store_and_render(
        frame=frame, trace=_trace(group_by=("reporter",)), digest=lambda: "digest"
    )

    assert result.structured_content["rows"] == 2


async def test_row_count_is_zero_for_an_empty_frame(_store) -> None:
    frame = Frame.build(role_keys=["reporter"], rows=[])

    result = await _store_and_render(
        frame=frame, trace=_trace(group_by=("reporter", "time_period")), digest=lambda: "digest"
    )

    assert result.structured_content["rows"] == 0
    assert result.structured_content["stat_var_count"] == 0


async def test_role_keys_is_empty_for_a_keyless_frame(_store) -> None:
    frame = Frame.build(
        role_keys=[],
        rows=[{"stat_var": "dcid:x", "time_period": "2022", "value": 1.0}],
    )

    result = await _store_and_render(frame=frame, trace=_trace(), digest=lambda: "digest")

    assert result.structured_content["role_keys"] == []


async def test_stat_var_count_counts_two_distinct_measures(_store) -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:oda",
                "reporter": "dcid:USA",
                "time_period": "2022",
                "value": 1.0,
            },
            {
                "stat_var": "dcid:gni",
                "reporter": "dcid:USA",
                "time_period": "2022",
                "value": 2.0,
            },
        ],
    )

    result = await _store_and_render(
        frame=frame, trace=_trace(group_by=("reporter", "value")), digest=lambda: "digest"
    )

    assert result.structured_content["stat_var_count"] == 2


async def test_the_structured_content_still_carries_the_identity_and_trace_fields(_store) -> None:
    frame = Frame.build(role_keys=[], rows=[])
    trace = _trace()

    result = await _store_and_render(frame=frame, trace=trace, digest=lambda: "digest")

    assert result.structured_content["spec_id"] == trace.op_spec_hash
    assert result.structured_content["content_version"] == trace.content_version
    assert result.structured_content["frame_content_version"] == trace.frame_content_version
    assert result.structured_content["exclusions"] == []
    assert result.structured_content["time"] == {"kind": "observation"}
