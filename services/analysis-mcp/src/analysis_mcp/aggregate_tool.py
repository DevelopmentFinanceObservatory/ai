"""The `aggregate` tool: op-spec in, deterministic engine execution over a
stored frame, a `{spec_id, content_version}` pair and a trace out.

Flow: parse/validate the op-spec (own `op_spec_invalid` refusal, never a raw
`pydantic.ValidationError` escaping this function) -> resolve `frame_ref`
via the store -> check the row-count ceiling -> run the engine
(`analysis_mcp.engine.aggregate.run_aggregate`), translating each of its
typed exceptions to its own refusal -> check the coverage floor -> mint
identity, build the trace, write the store under the result's own pair ->
render the digest.

Every typed failure this function itself anticipates is returned via
`to_tool_result`, matching `data_mcp.get_data_tool`'s own discipline of
catching every typed failure at this one boundary. An exception outside
that set would escape this function. `server.py`'s registration-site
wrapper is the backstop that turns it into an `internal_error` refusal
instead of a raw traceback reaching the wire (see that module's own
docstring), and this backstop complements the typed handling below rather
than replacing it.

The stored step (`_aggregate_step`) counts the groups, the output entities
that hold a total and the periods the reduction kept, the input entities each
period's totals pool, and the flow parties and unit-less values it left out.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.digest import render_aggregate_digest
from analysis_mcp.engine.aggregate import (
    AGGREGATIONS,
    AggregateResult,
    ComposedUnitSumError,
    NonFiniteAggregateResultError,
    StatVarUnresolvedError,
    UnknownGroupByColumnError,
    UnsupportedAggregationError,
    run_aggregate,
)
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    PlaceLevelsMixedError,
    UnitHeterogeneousError,
    UnitNotPresentError,
)
from analysis_mcp.frame_ref_resolution import (
    build_interval,
    canonical_columns_and_resolved_frame,
    derive_frame_content_version,
    derived_marker_detail,
    holds_input_rows,
    holds_ratio,
    input_content_version,
    merged_time,
    read_derived_marker,
    resolve_input_frame,
)
from analysis_mcp.identity import canonicalize_op_spec
from analysis_mcp.op_spec import AggregateOpSpec, PinnedUnit
from analysis_mcp.settings import Settings
from analysis_mcp.steps import Step, build_step, left_out_from_exclusions, step_unit
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import (
    build_op_spec_or_refusal,
    exclusions_for_trace,
    mint_identity_or_refusal,
    store_result_and_render,
)
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from analysis_mcp.units import unit_json

# Every typed exception `run_aggregate` can raise maps to exactly one
# refusal, via `_refusal_for_engine_error` below.
_EngineError = (
    UnsupportedAggregationError
    | ComposedUnitSumError
    | UnknownGroupByColumnError
    | StatVarUnresolvedError
    | UnitNotPresentError
    | UnitHeterogeneousError
    | FacetHeterogeneousError
    | NonFiniteAggregateResultError
    | PlaceLevelsMixedError
)


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_aggregate`'s typed exceptions to its refusal.

    An if/elif chain with one assignment and one return, rather than one
    early return per branch. `isinstance` narrowing inside the chain is what
    lets each branch read `exc`'s own subtype-specific attributes, and it
    keeps this function under the workspace's `PLR0911` return-count limit.
    """
    if isinstance(exc, UnsupportedAggregationError):
        refusal = ToolRefusal.operation_uncovered(
            aggregation=exc.aggregation, supported=list(AGGREGATIONS)
        )
    elif isinstance(exc, UnknownGroupByColumnError):
        refusal = ToolRefusal.role_ambiguous(
            parameter="group_by", columns=list(exc.columns), available=list(exc.available)
        )
    elif isinstance(exc, StatVarUnresolvedError):
        refusal = ToolRefusal.stat_var_unresolved(
            reason=exc.reason, distinct_stat_vars=list(exc.distinct_stat_vars)
        )
    elif isinstance(exc, UnitNotPresentError):
        refusal = ToolRefusal.unit_not_present(
            parameter="unit", pinned_unit=exc.pinned_unit, units=list(exc.units)
        )
    elif isinstance(exc, UnitHeterogeneousError):
        refusal = ToolRefusal.unit_heterogeneous(parameter="unit", units=list(exc.units))
    elif isinstance(exc, FacetHeterogeneousError):
        refusal = ToolRefusal.facet_heterogeneous(axis=exc.axis, values=list(exc.values))
    elif isinstance(exc, PlaceLevelsMixedError):
        refusal = ToolRefusal.place_levels_mixed(column=exc.column, types=list(exc.types))
    elif isinstance(exc, ComposedUnitSumError):
        refusal = ToolRefusal.composed_unit_sum_unsupported(verb="aggregate", unit=exc.unit)
    else:
        refusal = ToolRefusal.non_finite_result(
            group_key=exc.group_key, aggregation=exc.aggregation
        )
    return refusal


def _aggregate_step(op_spec: AggregateOpSpec, result: AggregateResult, *, trace: Trace) -> Step:
    """The step record of `result`, stored on `trace`: the groups it
    reduced, the output entities that hold a total, the periods, how many
    input entities each period's totals pool, what it left out, and the
    input's unit, the ratio unit over a ratio."""
    return build_step(
        op="aggregate",
        trace=trace,
        inputs={"input": op_spec.frame_ref},
        kept={
            "groups": result.present_cell_count,
            "entities": result.entity_count,
            "periods": len(result.collapsed_periods),
        },
        left_out=left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        unit=step_unit(result.resolved_unit, ratio=holds_ratio(trace)),
        period_entities=result.period_entities,
    )


@holds_input_rows
async def aggregate(
    frame_ref: Annotated[
        dict[str, str],
        Field(
            description=(
                "The {spec_id, content_version} pair naming a materialised frame or a "
                "prior verb result. Obtained from get_data's or a prior verb's own digest "
                "-- never invented."
            )
        ),
    ],
    aggregation: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description=(
                f"The reduction function. Closed vocabulary: {list(AGGREGATIONS)!r}. Under "
                "max or min, each group's digest line carries the period that produced the "
                "value in parentheses after it, and if time_period isn't in group_by, the "
                "output row's own time_period is that period. When several periods tie for a "
                "group's max or min, the earliest of them is reported. Under sum or mean with "
                "time_period outside group_by, every output row's time_period is the span "
                "the call covers, such as 2018/2022."
            ),
        ),
    ],
    group_by: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Column names to group by before reducing. Empty groups the whole input "
                "frame into one row."
            ),
        ),
    ] = (),
    unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the unit to aggregate over, when the input frame carries more than "
                "one: a unit ref, a converted unit as its {currency, prices, base_year} "
                "object, or a composed unit as its {numerator, denominator} object. "
                "Retry with one of a unit_heterogeneous refusal's fix.units values."
            ),
        ),
    ] = None,
) -> ToolResult:
    """Group a stored frame by `group_by` and reduce each group's `value` with `aggregation`.

    Use it to total or average a frame over one of its dimensions.

    `aggregate` skips absent grid cells, not null values. `Frame.build` cannot
    represent an in-frame null observation today (`value` must be finite,
    `status` is reserved), so "coverage" here means cells present versus
    cells expected for the grouped grain. Both figures land in the trace's
    `applied_policy` on every call. Unit and facet heterogeneity are checked
    before any number is computed. A heterogeneous unit refuses with
    candidates to retry against via `unit`, and a heterogeneous facet
    refuses naming a differently-scoped `frame_ref` to retry with instead. A
    result below the configured coverage floor refuses rather than returning
    a number computed from too sparse a grid. Under sum or mean with
    `time_period` outside `group_by`, every output row's `time_period` is
    one span from the earliest to the latest period the call reduced
    (`2018/2022`), the same on every row.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        aggregation: The reduction function. See the closed vocabulary above.
        group_by: Column names to group by. Empty reduces the whole frame to one row.
        unit: Pins the unit, to answer a prior unit_heterogeneous refusal.

    Returns:
        A digest: the computed numbers per group, the resolved unit, coverage,
        any warnings, and the result's {spec_id, content_version} pair and
        identity fields. Under max or min, each group's line also carries the
        period that produced the value, in parentheses after it. The full
        result frame is never in the digest or in structured content.
        Resolve it again by that pair. On refusal, a typed error with a
        machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: AggregateOpSpec(
            frame_ref=frame_ref, aggregation=aggregation, group_by=tuple(group_by), unit=unit
        )
    )
    if isinstance(op_spec, ToolResult):
        return op_spec

    # `group_by` is canonicalised here, before the engine ever runs, so
    # execution matches exactly the order `identity._normalise_parameters`
    # already normalises `group_by` to before minting `op_spec_hash`.
    # Otherwise two calls that mint the identical identity from a
    # differently-ordered `group_by` could compute (or, via the digest's
    # own magnitude cap, display) a different result. See
    # `frame_ref_resolution.canonical_columns_or_refusal`.
    settings = Settings()
    prelude = await canonical_columns_and_resolved_frame(
        op_spec.group_by,
        op_spec.frame_ref,
        description="group_by",
        resolve=lambda ref: resolve_input_frame(ref, settings=settings),
    )
    if isinstance(prelude, ToolResult):
        return prelude
    group_by, admitted = prelude
    frame, input_trace = admitted.frame, admitted.trace

    try:
        result = run_aggregate(
            frame,
            group_by=group_by,
            aggregation=op_spec.aggregation,
            pinned_unit=op_spec.unit,
        )
    except (
        UnsupportedAggregationError,
        ComposedUnitSumError,
        UnknownGroupByColumnError,
        StatVarUnresolvedError,
        UnitNotPresentError,
        UnitHeterogeneousError,
        FacetHeterogeneousError,
        NonFiniteAggregateResultError,
        PlaceLevelsMixedError,
    ) as exc:
        return to_tool_result(_refusal_for_engine_error(exc))

    if result.realised_coverage_fraction < settings.aggregate_coverage_floor:
        return to_tool_result(
            ToolRefusal.coverage_insufficient(
                realised_coverage_fraction=result.realised_coverage_fraction,
                floor=settings.aggregate_coverage_floor,
                skipped_cell_count=result.skipped_cell_count,
            )
        )

    input_cv = input_content_version(input_trace)
    identity = mint_identity_or_refusal(op_spec, input_content_versions=[input_cv])
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    # Under sum/mean with `time_period` outside `group_by`, every output row
    # carries `result.period_label`. Over a point frame with more than one
    # period that label is an interval, and this call builds its own
    # `Interval` so the trace records the gaps among the collapsed points. A
    # single-span input stamps its span through and `merged_time` inherits
    # the input's interval. Under max/min the output row's `time_period`
    # names the period whose value won the reduction.
    collapses_window = result.period_label is not None and len(result.collapsed_periods) > 1
    derived = derived_marker_detail(inherited=read_derived_marker(input_trace))
    detail: dict[str, object] = {}
    if derived is not None:
        detail["derived"] = derived

    own_interval = (
        build_interval(
            start=result.collapsed_periods[0],
            end=result.collapsed_periods[-1],
            observed=result.collapsed_periods,
        )
        if collapses_window
        else None
    )
    time_fact = merged_time(own_interval=own_interval, inherited=[admitted])

    output_trace = Trace(
        upstream_trace=input_trace.upstream_trace,
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="aggregate_skip_absent_grain_cells",
            realised_coverage_fraction=result.realised_coverage_fraction,
            skipped_row_count=result.skipped_cell_count,
            resolved_unit=unit_json(result.resolved_unit),
            resolved_facets={
                selection.axis: selection.ref for selection in result.facet_selections
            },
            detail=detail,
        ),
        exclusions=exclusions_for_trace(result.exclusions),
        time=time_fact,
        content_version=content_version,
        input_content_versions=[input_cv],
        execution_identity=execution_identity,
    )

    return await store_result_and_render(
        frame=result.frame,
        trace=output_trace,
        step=_aggregate_step(op_spec, result, trace=output_trace),
        digest=lambda: render_aggregate_digest(
            group_by=group_by,
            aggregation=op_spec.aggregation,
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
    )
