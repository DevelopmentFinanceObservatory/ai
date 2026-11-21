"""`build_op_spec_or_refusal`: a `pydantic.ValidationError` from a verb's own
op-spec constructor reaches the caller as a plain, per-error detail --
never pydantic's own class-name header or its `errors.pydantic.dev` link.
"""

from __future__ import annotations

from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.tool_pipeline import build_op_spec_or_refusal

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
