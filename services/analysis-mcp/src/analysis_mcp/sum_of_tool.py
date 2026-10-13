"""The `sum_of` tool: op-spec in, two stored-frame slices added period by
period, a `{spec_id, content_version}` pair and a trace out.

Flow mirrors `compare_tool.compare` and shares its two-slice pieces:
`resolve_slice_pair` resolves both sides, `SLICE_PAIR_ERRORS` and
`compare_tool._refusal_for_engine_error` turn the engine's typed exceptions into the
same refusals `compare` gives. As in `compare`, the op-spec hash is minted
before the engine runs, since the result's `stat_var` is
`dcid:<op_spec_hash>`, the operation's own ref, the same `dcid:<spec_id>`
shape a summed `get_data` retrieval carries.

The stored step (`_sum_step`) counts the periods the sum keeps and the
periods one side alone reports, which it leaves out.
"""

from __future__ import annotations

from typing import Annotated

from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.compare_tool import (
    SLICE_PAIR_ERRORS,
    _refusal_for_engine_error,
    resolve_slice_pair,
)
from analysis_mcp.digest import render_sum_of_digest
from analysis_mcp.engine.aggregate import ComposedUnitSumError
from analysis_mcp.engine.sum_of import SumOfResult, run_sum_of
from analysis_mcp.frame_ref_resolution import (
    canonical_entity_binding,
    derive_frame_content_version,
    derived_marker_detail,
    holds_input_rows,
    holds_ratio,
    merge_derived_markers,
    merged_time,
    op_spec_hash_or_refusal,
    read_derived_marker,
)
from analysis_mcp.identity import canonicalize_op_spec
from analysis_mcp.op_spec import PinnedUnit, SumOfOpSpec
from analysis_mcp.steps import LeftOut, Step, build_step, left_out_from_exclusions, step_unit
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import (
    build_op_spec_or_refusal,
    exclusions_for_trace,
    mint_identity_or_refusal,
    store_result_and_render,
)
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from analysis_mcp.units import unit_json


def _sum_step(op_spec: SumOfOpSpec, result: SumOfResult, *, trace: Trace) -> Step:
    """The step record of `result`, stored on `trace`: the periods it summed,
    the periods one side alone reports, and the unit both sides share, the
    ratio unit when the sum is of ratios."""
    one_sided = len({entry.time_period for entry in result.left_out})
    return build_step(
        op="sum",
        trace=trace,
        inputs={"left": op_spec.frame_ref, "right": op_spec.right_frame_ref or op_spec.frame_ref},
        kept={"periods": len(result.intersected_periods)},
        left_out=[
            *(
                [LeftOut(reason="period_unmatched", count=one_sided, of="periods")]
                if one_sided
                else []
            ),
            *left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        ],
        unit=step_unit(result.unit, ratio=holds_ratio(trace)),
    )


def _sum_or_refusal(
    op_spec: SumOfOpSpec, *, left: Frame, right: Frame, stat_var: str
) -> SumOfResult | ToolResult:
    """Run the engine over the two resolved sides, labelling the result
    `dcid:<stat_var>`, or return the refusal of the rule the call breaks:
    one of the pairing's, as `compare` refuses it, or a side in a composed
    unit (`composed_unit_sum_unsupported`)."""
    try:
        return run_sum_of(
            left,
            right,
            stat_var=f"dcid:{stat_var}",
            left_entity=canonical_entity_binding(op_spec.left_entity),
            right_entity=canonical_entity_binding(op_spec.right_entity),
            left_pinned_unit=op_spec.left_unit,
            right_pinned_unit=op_spec.right_unit,
        )
    except SLICE_PAIR_ERRORS as exc:
        return to_tool_result(_refusal_for_engine_error(exc))
    except ComposedUnitSumError as exc:
        return to_tool_result(
            ToolRefusal.composed_unit_sum_unsupported(verb="sum_of", unit=exc.unit, side=exc.side)
        )


@holds_input_rows
async def sum_of(
    frame_ref: Annotated[
        dict[str, str],
        Field(
            description=(
                "The left side's {spec_id, content_version} pair, naming a materialised "
                "frame or a prior verb result. Obtained from get_data's or a prior verb's "
                "own digest -- never invented."
            )
        ),
    ],
    right_frame_ref: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "The right side's frame_ref pair. Omit to add two slices of the same "
                "frame_ref, narrowed by left_entity/right_entity instead."
            ),
        ),
    ] = None,
    left_entity: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{role_column: value} narrowing the left side to one entity's own time "
                "series. Empty when frame_ref already names exactly one entity. Role "
                "columns only -- stat_var and time_period are not accepted selectors."
            ),
        ),
    ] = None,
    right_entity: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Same as left_entity, for the right side."),
    ] = None,
    left_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the left side's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
    right_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the right side's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
) -> ToolResult:
    """Add two single-series slices, period by period, over the periods both report.

    Each side is `frame_ref` (or `right_frame_ref`) narrowed by an `entity`
    binding down to one row per `time_period`. A period only one side
    reports is left out, never counted as zero; the digest names those
    periods and each side's coverage. Periods with no overlap refuse
    (`comparison_periods_disjoint`), and the two sides must resolve to the
    same unit (`cross_frame_unit_mismatch`, `cross_frame_unit_unbound`).
    Differing provenance or measurement method is a warning, and the result
    leaves that facet unbound. The result is one series under its own
    stat_var. Use `compare` for the difference between two slices.

    Args:
        frame_ref: The left side's frame_ref.
        right_frame_ref: The right side's frame_ref; omit for two slices of frame_ref.
        left_entity: Narrows the left side to one entity's series.
        right_entity: Narrows the right side to one entity's series.
        left_unit: Pins the left side's unit.
        right_unit: Pins the right side's unit.

    Returns:
        A digest: each summed period's two values and sum, the summed span,
        each side's coverage and the periods left out, and the result's
        {spec_id, content_version} pair and identity fields. On refusal, a
        typed error with a machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: SumOfOpSpec(
            frame_ref=frame_ref,
            right_frame_ref=right_frame_ref,
            # See `compare_tool.compare` for why `None` maps to `{}` here.
            left_entity={} if left_entity is None else left_entity,
            right_entity={} if right_entity is None else right_entity,
            left_unit=left_unit,
            right_unit=right_unit,
        )
    )
    if isinstance(op_spec, ToolResult):
        return op_spec
    op_spec_hash = op_spec_hash_or_refusal(op_spec)
    if isinstance(op_spec_hash, ToolResult):
        return op_spec_hash

    sides = await resolve_slice_pair(op_spec)
    if isinstance(sides, ToolResult):
        return sides
    left, right, input_content_versions = sides

    result = _sum_or_refusal(op_spec, left=left.frame, right=right.frame, stat_var=op_spec_hash)
    if isinstance(result, ToolResult):
        return result

    identity = mint_identity_or_refusal(op_spec, input_content_versions=input_content_versions)
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    detail: dict[str, object] = {
        "left_periods": list(result.left_periods),
        "right_periods": list(result.right_periods),
        "intersected_periods": list(result.intersected_periods),
        "left_out": [
            {"time_period": entry.time_period, "side": entry.side, "value": entry.value}
            for entry in result.left_out
        ],
        "left_coverage_fraction": result.left_coverage_fraction,
        "right_coverage_fraction": result.right_coverage_fraction,
        "left_stat_vars": list(result.left_stat_vars),
        "right_stat_vars": list(result.right_stat_vars),
        "left_resolved_facets": {
            selection.axis: selection.ref for selection in result.left_facet_selections
        },
        "right_resolved_facets": {
            selection.axis: selection.ref for selection in result.right_facet_selections
        },
    }
    # The result's `stat_var` is its own, so `sum_of` sets no marker of its
    # own. A marker inherited from either side carries forward.
    derived = derived_marker_detail(
        inherited=merge_derived_markers(
            read_derived_marker(left.trace), read_derived_marker(right.trace)
        )
    )
    if derived is not None:
        detail["derived"] = derived

    # The left side's upstream_trace is carried forward, as `compare` does.
    # Both sides' frames are bound into `content_version` through
    # `input_content_versions`.
    output_trace = Trace(
        upstream_trace=left.trace.upstream_trace,
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="sum_of_intersect_periods",
            realised_coverage_fraction=min(
                result.left_coverage_fraction, result.right_coverage_fraction
            ),
            skipped_row_count=0,
            resolved_unit=unit_json(result.unit),
            resolved_facets={
                selection.axis: selection.ref for selection in result.facet_selections
            },
            detail=detail,
        ),
        exclusions=exclusions_for_trace(result.exclusions),
        time=merged_time(own_interval=None, inherited=[left, right]),
        content_version=content_version,
        input_content_versions=input_content_versions,
        execution_identity=execution_identity,
    )

    return await store_result_and_render(
        frame=result.frame,
        trace=output_trace,
        step=_sum_step(op_spec, result, trace=output_trace),
        digest=lambda: render_sum_of_digest(
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
    )
