"""The `rank` tool: op-spec in, deterministic engine execution over a
stored frame, a `{spec_id, content_version}` pair and a trace out.

Flow mirrors `aggregate_tool.aggregate`: parse/validate the op-spec ->
resolve `frame_ref` via the shared `analysis_mcp.frame_ref_resolution`
helpers -> run the engine (`analysis_mcp.engine.rank.run_rank`), translating
each of its typed exceptions to its own refusal -> check the coverage floor
(reusing `aggregate`'s own `engine.aggregate.COVERAGE_FLOOR` and
`coverage_insufficient` refusal, see `analysis_mcp.engine.rank`'s own module
docstring for why) -> mint identity, build the trace, write the store under
the result's own pair -> render the digest. Every typed failure this
function itself anticipates is returned via `to_tool_result`. An exception
outside that set would escape this function. `server.py`'s
registration-site wrapper is the backstop that turns it into an
`internal_error` refusal instead of a raw traceback reaching the wire, and
this backstop complements the typed handling below rather than replacing
it.

The stored step (`_rank_step`) counts the entities the ranking keeps and
how many of the window's periods each holds, with the flow parties and
unit-less values it left out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.composite import composite_refusal
from analysis_mcp.digest import render_rank_digest
from analysis_mcp.engine.aggregate import (
    AGGREGATIONS,
    COVERAGE_FLOOR,
    ComposedUnitSumError,
    NonFiniteAggregateResultError,
    UnsupportedAggregationError,
)
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    PlaceLevelsMixedError,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnitNotPresentError,
    UnknownEntityColumnError,
)
from analysis_mcp.engine.rank import (
    RankAmbiguousEntityError,
    RankNoRowsInWindowError,
    RankResult,
    run_rank,
)
from analysis_mcp.engine_facts import facts_of
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
from analysis_mcp.op_spec import (
    ANCHOR_KEY_PATTERN,
    MAX_RANK_LIMIT,
    FrameRefArgument,
    PinnedUnit,
    RankOpSpec,
    WindowRolling,
    with_anchor_key,
)
from analysis_mcp.rolling_tool import anchored_input
from analysis_mcp.settings import Settings
from analysis_mcp.steps import (
    Step,
    build_step,
    caller_params,
    entity_ref,
    left_out_from_exclusions,
    step_unit,
    window_params,
)
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import (
    build_op_spec_or_refusal,
    entity_label,
    exclusions_for_trace,
    mint_identity_or_refusal,
    store_result_and_render,
)
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from analysis_mcp.units import unit_json

if TYPE_CHECKING:
    from common.materialisation import FrameRef

    from analysis_mcp.frame_ref_resolution import AdmittedFrame

# Every typed exception `run_rank` can raise maps to exactly one
# refusal by `_refusal_for_engine_error` below. Every code here exists on
# `ToolRefusal`; `place_levels_mixed` is shared with `aggregate` and
# `share_of`.
_EngineError = (
    UnsupportedAggregationError
    | ComposedUnitSumError
    | UnknownEntityColumnError
    | StatVarUnresolvedError
    | UnitNotPresentError
    | UnitHeterogeneousError
    | FacetHeterogeneousError
    | RankNoRowsInWindowError
    | RankAmbiguousEntityError
    | NonFiniteAggregateResultError
    | PlaceLevelsMixedError
)


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_rank`'s typed exceptions to its refusal.

    `UnknownEntityColumnError` and `RankAmbiguousEntityError` both reuse
    `role_ambiguous`, the same reasoning `growth_tool`'s own mapping uses.
    `RankNoRowsInWindowError` reuses `stat_var_unresolved`. "No rows
    present" is exactly the empty-input condition that refusal already
    names for `aggregate`, just reached here after windowing rather than
    before. `NonFiniteAggregateResultError` (raised directly by
    `run_rank`'s own reduction step, reusing `aggregate`'s type) reuses
    `non_finite_result`.
    """
    # An if/elif chain with one assignment and one return, rather than one
    # early return per branch, matching
    # `aggregate_tool._refusal_for_engine_error`'s own shape (which the
    # workspace's `PLR0911` complexity gate requires).
    if isinstance(exc, UnsupportedAggregationError):
        refusal = ToolRefusal.operation_uncovered(
            aggregation=exc.aggregation, supported=list(AGGREGATIONS)
        )
    elif isinstance(exc, UnknownEntityColumnError):
        refusal = ToolRefusal.role_ambiguous(
            parameter="rank_by", columns=list(exc.columns), available=list(exc.available)
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
    elif isinstance(exc, RankNoRowsInWindowError):
        refusal = ToolRefusal.stat_var_unresolved(reason=exc.reason, distinct_stat_vars=[])
    elif isinstance(exc, RankAmbiguousEntityError):
        refusal = ToolRefusal.role_ambiguous(
            parameter="rank_by", columns=list(exc.rank_by), available=list(exc.available)
        )
    elif isinstance(exc, PlaceLevelsMixedError):
        refusal = ToolRefusal.place_levels_mixed(
            column=exc.column,
            types=list(exc.types),
            whole_total=exc.whole_total,
            whole_class=exc.whole_class,
            member_count=exc.member_count,
            members=list(exc.members),
        )
    elif isinstance(exc, ComposedUnitSumError):
        refusal = ToolRefusal.composed_unit_sum_unsupported(verb="rank", unit=exc.unit)
    else:
        refusal = ToolRefusal.non_finite_result(
            group_key=exc.group_key, aggregation=exc.aggregation
        )
    return refusal


def _rank_step(op_spec: RankOpSpec, result: RankResult, *, trace: Trace) -> Step:
    """The step record of `result`, stored on `trace`: the entities the
    output keeps, in ranked order with how many of the window's periods each
    holds, what it left out, and its unit. A ranking whose value is the rank
    position has no unit, and one over a ratio names the ratio unit. The
    params name the order `ascending` or `descending` and the window's
    bounds `start` and `end`."""
    params = window_params(caller_params(trace.execution_record.content))
    params["order"] = "ascending" if params.pop("ascending") else "descending"
    coverage = {
        entity_rank.entity_key: (
            entity_ref(entity_rank.entity_key),
            entity_rank.period_count,
            result.window_period_count,
        )
        for entity_rank in result.kept_ranks
    }
    return build_step(
        op="rank",
        trace=trace,
        inputs={"input": op_spec.frame_ref},
        params=params,
        kept={"entities": len(coverage)},
        left_out=left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        unit=None if result.ordinal else step_unit(result.resolved_unit, ratio=holds_ratio(trace)),
        entity_coverage=coverage.values(),
    )


async def _resolve_measured_frame(
    frame_ref: FrameRef, *, settings: Settings
) -> AdmittedFrame | ToolResult:
    """`resolve_input_frame`, refusing a frame whose `value` is a rank position.

    A rank position is no measure to order entities by: highest first over
    positions 1 and 2 puts the entity placed second on top, and a `limit`
    then keeps it. The refusal (`derived_input_ordinal`) holds in either
    mode and whatever the `limit`, and is raised here since `run_rank`
    reads only the frame and the marker travels with the trace.
    """
    resolved = await resolve_input_frame(frame_ref, settings=settings)
    if isinstance(resolved, ToolResult):
        return resolved
    marker = read_derived_marker(resolved.trace)
    if marker is not None and marker.ordinal:
        return to_tool_result(
            ToolRefusal.derived_input_ordinal(
                reason=(
                    "this frame_ref names a per-period rank's own output (or a result "
                    "derived from one), whose value column holds a rank position"
                )
            )
        )
    return resolved


@holds_input_rows
async def rank(
    frame_ref: Annotated[
        FrameRefArgument,
        Field(
            description=(
                "The {spec_id, content_version} pair naming a materialised frame or a "
                "prior verb result. Obtained from get_data's or a prior verb's own digest "
                "-- never invented. "
                "A bare spec_id also works: the server fetches the data itself."
            )
        ),
    ],
    rank_by: Annotated[
        tuple[str, ...],
        Field(min_length=1, description="Column names identifying each ranked entity."),
    ],
    aggregation: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "The reduction function to apply to each entity's own rows before ranking. "
                f"Closed vocabulary: {list(AGGREGATIONS)!r}. Left None, each entity must "
                "already carry exactly one row within the window."
            ),
        ),
    ] = None,
    window_start: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^\d{4}$",
            description="The window's first calendar year. Must be set together with window_end.",
        ),
    ] = None,
    window_end: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^\d{4}$",
            description="The window's last calendar year. Must be set together with window_start.",
        ),
    ] = None,
    unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the unit to rank over, when the input frame carries more than "
                "one: a unit ref, a converted unit as its {currency, prices, base_year} "
                "object, or a composed unit as its {numerator, denominator} object. "
                "Retry with one of a unit_heterogeneous refusal's fix.units values."
            ),
        ),
    ] = None,
    ascending: Annotated[
        bool, Field(default=False, description="Rank lowest-first instead of highest-first.")
    ] = False,
    per_period: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "False (the default) collapses the window into one ranking, one row per "
                "entity, whose value is the measured value and whose time_period is the "
                "extent of the periods the window holds, such as 2018/2022. True keeps "
                "the window's periods apart: each period is ranked on its own, and the "
                "output carries one row per entity per period whose value is that entity's "
                "rank position in that period, not the measured value, so a rank series can "
                "be grown; with limit set, the value stays the measured value. Set this when the question asks how a position changed over time "
                "rather than who is first; name the entity the question is about with a "
                "filter between this call and a growth(change='absolute') -- the series "
                "carries the whole population, and a growth over it needs one entity's own "
                "position, not everyone's."
            ),
        ),
    ] = False,
    limit: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=MAX_RANK_LIMIT,
            description=(
                "Keep only the entities ranked at this position or better: 10 keeps the "
                "highest 10, or the lowest 10 with ascending=True. Entities that share the "
                "value at the boundary are all kept, so more than limit can remain. Applies "
                "within each period when per_period=True, and the output's value is then "
                "the measured value, not the rank position. Left None, every ranked entity "
                "is kept."
            ),
        ),
    ] = None,
    rolling: Annotated[
        WindowRolling | None,
        Field(
            default=None,
            description=(
                "Marks window_start, window_end, or both as following the latest period of "
                "the data: {window_end: {anchor: 'latest', offset: 0, rule: 'half1'}}, with "
                "offset counting whole years back from the latest. The window's years stay "
                "literal and are what is computed; two rolling bounds must be the same latest "
                "year at their offsets. population is required: one {spec_id, bindings} for "
                "each root dataset of the question, bindings narrowing its series as {column: "
                "[values]}; the latest period is taken over those series. Left None, the "
                "window is fixed."
            ),
        ),
    ] = None,
    anchor_key: Annotated[
        str | None,
        Field(
            default=None,
            pattern=ANCHOR_KEY_PATTERN,
            description=(
                "The handle of the stored record of the latest period this call's rolling "
                "periods were written from: rak_ and 32 hex digits. Required when the call "
                "is rolling and refused when it is not. It is no part of the result's identity."
            ),
        ),
    ] = None,
) -> ToolResult:
    """Rank entities identified by `rank_by`, optionally reducing each
    entity's own rows with `aggregation` first.

    Use it to order entities by a value, such as the largest donors in a
    year, and to keep the top or bottom few.

    Ranks what's present. An entity absent from the window simply isn't
    ranked. Competition ranking ("1224"). Entities sharing a value share a
    rank number, and the rank number never breaks a tie. The *order* two
    tied entities are listed in is broken deterministically by the frame
    grammar's own total ordering over `rank_by`'s values. Refuses
    (`coverage_insufficient`, reusing `aggregate`'s own floor) when the
    grid, every ranked entity times every distinct period any entity
    contributes as reflected in the input data rather than the window's
    full calendar range, falls below the configured coverage floor. A
    period where no entity has any data at all is invisible to this check.

    Collapsed, every output row's `time_period` is the extent of the
    periods the window holds, such as `2018/2022`, or that one period when
    the window holds only one. A frame whose rows carry one span, such as
    a growth result, keeps that span; a window keeps a span row only when
    it covers every year of the span.

    `per_period=True` ranks each of the window's periods on its own instead
    of collapsing them: with no `limit`, the output frame's own `value`
    column carries each entity's rank *position* per period, not its
    measured value, and no
    verb here may read that position as a quantity to ratio (`share_of`) or
    grow at a percentage rate (`growth` with `change="relative"`) or rank
    again -- all three refuse (`derived_input_ordinal`) over this call's own
    output, or over anything derived from it. A top or bottom N per period
    is this call with `limit` set.

    `limit` keeps the entities ranked at that position or better in the
    output frame, every one tied at the boundary included, with their
    measured values in either mode: a limited per-period rank writes no
    ordinal marker. Coverage and the floor check cover every ranked entity,
    and the rows left out are reported as a `beyond_limit` exclusion.

    `rolling` records that a window year follows the latest period of the
    data. The call ranks over the literal years it is given, as a fixed call
    does, and the result's {spec_id, content_version} pair differs from the
    fixed call's. Such a call runs only when the record `anchor_key` names
    covers it (`rolling_tool.anchored_input`), and refuses
    `rolling_anchor_unresolved` otherwise.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        rank_by: Column names identifying each ranked entity.
        aggregation: The reduction function. None expects one row per entity already.
        window_start: The window's first calendar year, paired with window_end.
        window_end: The window's last calendar year, paired with window_start.
        unit: Pins the unit, to answer a prior unit_heterogeneous refusal.
        ascending: Rank lowest-first instead of highest-first.
        per_period: Rank each period on its own; with no limit, the output's value is the
            rank position.
        limit: Keep only the entities ranked at this position or better.
        rolling: Which window years follow the latest period, and the
            populations that period is taken over.
        anchor_key: The handle of the stored latest period, set on a rolling call.

    Returns:
        A digest: each entity's rank, value, and coverage, the resolved
        unit, overall coverage, and the result's {spec_id, content_version}
        pair and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: with_anchor_key(
            RankOpSpec(
                frame_ref=frame_ref,
                rank_by=tuple(rank_by),
                aggregation=aggregation,
                window_start=window_start,
                window_end=window_end,
                unit=unit,
                ascending=ascending,
                per_period=per_period,
                limit=limit,
                rolling=rolling,
            ),
            anchor_key,
        )
    )
    if isinstance(op_spec, ToolResult):
        return op_spec

    # `rank_by` is canonicalised before the engine runs. See
    # `aggregate_tool.aggregate`'s identical comment on its own `group_by`.
    settings = Settings()
    prelude = await canonical_columns_and_resolved_frame(
        op_spec.rank_by,
        op_spec.frame_ref,
        description="rank_by",
        resolve=lambda ref: anchored_input(
            op_spec, anchor_key, _resolve_measured_frame(ref, settings=settings)
        ),
    )
    if isinstance(prelude, ToolResult):
        return prelude
    rank_by, admitted = prelude
    frame, input_trace = admitted.frame, admitted.trace

    try:
        result = run_rank(
            frame,
            rank_by=rank_by,
            aggregation=op_spec.aggregation,
            window_start=op_spec.window_start,
            window_end=op_spec.window_end,
            ascending=op_spec.ascending,
            pinned_unit=op_spec.unit,
            per_period=op_spec.per_period,
            limit=op_spec.limit,
        )
    except (
        UnsupportedAggregationError,
        ComposedUnitSumError,
        UnknownEntityColumnError,
        StatVarUnresolvedError,
        UnitNotPresentError,
        UnitHeterogeneousError,
        FacetHeterogeneousError,
        RankNoRowsInWindowError,
        RankAmbiguousEntityError,
        NonFiniteAggregateResultError,
        PlaceLevelsMixedError,
    ) as exc:
        return to_tool_result(composite_refusal(_refusal_for_engine_error(exc), [input_trace]))

    if result.overall_coverage_fraction < COVERAGE_FLOOR:
        return to_tool_result(
            ToolRefusal.coverage_insufficient(
                realised_coverage_fraction=result.overall_coverage_fraction,
                floor=COVERAGE_FLOOR,
                skipped_cell_count=result.expected_cell_count - result.present_cell_count,
            )
        )

    input_cv = input_content_version(input_trace)
    identity = mint_identity_or_refusal(op_spec, input_content_versions=[input_cv])
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    # Collapsing mode over more than one period reduces a point frame's
    # window to one row per entity, so the trace records that window as an
    # interval below. An admitted frame with more than one period holds only
    # points; a span frame holds one period and passes its span through.
    # `per_period=True` never collapses. The `ordinal` flag marks the
    # frame's own `value` column as a rank position rather than a measured
    # value (`RankResult.ordinal`, set per period with no `limit`); see
    # `frame_ref_resolution.DerivedMarker`.
    collapses_window = not result.per_period and len(result.periods_in_window) > 1
    derived = derived_marker_detail(
        inherited=read_derived_marker(input_trace), ordinal=result.ordinal
    )
    # Coverage per entity the output frame holds, and three numbers for the
    # whole ranked population, so the stored trace grows with the rows kept
    # however many entities were ranked. Coverage is entity-wide, so an
    # entity repeated across periods carries one fraction.
    ranked = result.entity_ranks
    detail: dict[str, object] = {
        "per_entity_coverage": {
            entity_label(entity_rank.entity_key): entity_rank.coverage_fraction
            for entity_rank in result.kept_ranks
        },
        "ranked_population": {
            "entity_count": len({entity_rank.entity_key for entity_rank in ranked}),
            "min_coverage_fraction": min(entity_rank.coverage_fraction for entity_rank in ranked),
            "max_coverage_fraction": max(entity_rank.coverage_fraction for entity_rank in ranked),
        },
    }
    if derived is not None:
        detail["derived"] = derived

    # The interval this call's window drew from, when it collapses one: the
    # earliest and latest period any ranked entity contributed
    # (`RankResult.periods_in_window`), which also serve as
    # `build_interval`'s `observed` argument for `contiguous`/`absent`.
    # Otherwise `merged_time` inherits the input's interval, a span frame's
    # included.
    own_interval = (
        build_interval(
            start=min(result.periods_in_window),
            end=max(result.periods_in_window),
            observed=result.periods_in_window,
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
            policy_name="rank_what_is_present",
            realised_coverage_fraction=result.overall_coverage_fraction,
            skipped_row_count=result.expected_cell_count - result.present_cell_count,
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
        step=_rank_step(op_spec, result, trace=output_trace),
        digest=lambda: render_rank_digest(
            rank_by=rank_by,
            result=facts_of(result),
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
        ranked_before_limit=len(result.entity_ranks) if result.limit is not None else None,
    )
