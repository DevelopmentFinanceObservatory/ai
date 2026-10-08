"""`build_op_spec_or_refusal`: a `pydantic.ValidationError` from a verb's own
op-spec constructor reaches the caller as a plain, per-error detail --
never pydantic's own class-name header or its `errors.pydantic.dev` link.

`store_result_and_render`: the returned `structured_content` is a real
`result.VerbResult`, with `role_keys`/`stat_var_count`/`rows` read off
the frame it stores, not the trace. The trace it stores describes every
operation-minted StatVar that frame holds, from the input traces and the
verb's own entry.
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
from common.materialisation import (
    ComposedStatVar,
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    set_store,
)
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
    """`store_result_and_render` inside a verb call's row claim, as every verb
    runs it, reading no input trace unless the test passes some."""
    return await store_result_and_render(**{"input_traces": (), **kwargs})


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


# The composed StatVars the stored trace describes ------------------------------


def _holding(*stat_vars: str) -> Frame:
    return Frame.build(
        role_keys=[],
        rows=[{"stat_var": ref, "time_period": "2022", "value": 1.0} for ref in stat_vars],
    )


def _quotient(stat_var: str, first: str, second: str) -> ComposedStatVar:
    return ComposedStatVar(stat_var=stat_var, relation="quotient", first=(first,), second=(second,))


async def _stored_entries(
    store: InMemoryMaterialisationStore, trace: Trace
) -> tuple[ComposedStatVar, ...]:
    _frame, stored = await store.get(
        FrameRef(spec_id=trace.op_spec_hash, content_version=trace.content_version)
    )
    return stored.composed_stat_vars


_INNER = _quotient("dcid:ops_inner", "dcid:gdp", "dcid:population")
_MIDDLE = _quotient("dcid:ops_middle", "dcid:ops_inner", "dcid:area")
_OUTER = ComposedStatVar(
    stat_var="dcid:ops_outer",
    relation="sum",
    first=("dcid:ops_middle", "dcid:oda"),
    second=("dcid:ops_inner",),
)
_UNRELATED = _quotient("dcid:ops_unrelated", "dcid:oda", "dcid:gni")


async def test_an_input_entry_is_stored_with_every_entry_it_names_before_it(_store) -> None:
    """The input lists arrive outermost first and split across two traces;
    the entry whose StatVar left the frame, and that no kept entry names, is
    dropped."""
    trace = _trace()
    inputs = [
        _trace().model_copy(update={"composed_stat_vars": (_OUTER, _UNRELATED)}),
        MaterialisationTrace(
            upstream_trace={}, spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32
        ),
        _trace().model_copy(update={"composed_stat_vars": (_MIDDLE, _INNER)}),
    ]

    await _store_and_render(
        frame=_holding("dcid:ops_outer", "dcid:oda"),
        trace=trace,
        input_traces=inputs,
        digest=lambda: "digest",
    )

    assert await _stored_entries(_store, trace) == (_INNER, _MIDDLE, _OUTER)


async def test_the_own_entry_is_stored_after_the_input_entries_it_names(_store) -> None:
    trace = _trace()
    own_ref = f"dcid:{trace.op_spec_hash}"
    own = _quotient(own_ref, "dcid:ops_middle", "dcid:population")

    await _store_and_render(
        frame=_holding(own_ref),
        trace=trace,
        input_traces=[_trace().model_copy(update={"composed_stat_vars": (_INNER, _MIDDLE)})],
        own_composed_stat_var=own,
        digest=lambda: "digest",
    )

    assert await _stored_entries(_store, trace) == (_INNER, _MIDDLE, own)


async def test_no_entry_is_stored_for_a_frame_holding_no_described_stat_var(_store) -> None:
    trace = _trace()
    own = _quotient(f"dcid:{trace.op_spec_hash}", "dcid:gdp", "dcid:population")

    await _store_and_render(
        frame=_holding("dcid:gdp"),
        trace=trace,
        input_traces=[_trace().model_copy(update={"composed_stat_vars": (_INNER,)})],
        own_composed_stat_var=own,
        digest=lambda: "digest",
    )

    assert await _stored_entries(_store, trace) == ()


async def test_a_frame_holding_its_own_stat_var_with_no_own_entry_raises(_store) -> None:
    trace = _trace()

    with pytest.raises(RuntimeError):
        await _store_and_render(
            frame=_holding(f"dcid:{trace.op_spec_hash}"), trace=trace, digest=lambda: "digest"
        )

    assert (
        await _store.get(
            FrameRef(spec_id=trace.op_spec_hash, content_version=trace.content_version)
        )
        is None
    )


async def test_an_own_entry_describing_another_stat_var_raises(_store) -> None:
    with pytest.raises(ValueError, match="dcid:ops_inner"):
        await _store_and_render(
            frame=_holding("dcid:gdp"),
            trace=_trace(),
            own_composed_stat_var=_INNER,
            digest=lambda: "digest",
        )
