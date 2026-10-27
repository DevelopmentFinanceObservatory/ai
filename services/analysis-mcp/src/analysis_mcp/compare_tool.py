"""The `compare` tool: op-spec in, deterministic engine execution over two
stored-frame slices, a `{spec_id, content_version}` pair and a trace out.

Flow mirrors `aggregate_tool.aggregate`: parse/validate the op-spec -> mint
the op-spec hash, which names the result's `stat_var` (`dcid:<op_spec_hash>`)
as `sum_of_tool.sum_of` names a sum's -> resolve the left
`frame_ref` via the shared `analysis_mcp.frame_ref_resolution` helpers, and
the right `right_frame_ref` too when it names a different result (left alone
stands for "two slices of one frame_ref"; see
`analysis_mcp.engine.compare`'s own docstring) -> run the engine
(`analysis_mcp.engine.compare.run_compare`), translating each of its typed
exceptions to its own refusal -> mint identity, build the trace, write the
store under the result's own pair -> render the digest. Every typed failure this
function itself anticipates is returned via `to_tool_result`, never raised.
`resolve_slice_pair`, `SLICE_PAIR_ERRORS` and `_refusal_for_engine_error`
serve `sum_of_tool.sum_of` too, which pairs its sides through the same
engine helpers.
An exception outside that set would escape this function; `server.py`'s
registration-site wrapper is the backstop that turns it into an
`internal_error` refusal instead of a raw traceback reaching the wire;
this backstop complements the typed handling below rather than replacing it.

The stored step (`_compare_step`) counts the periods the comparison keeps
and the periods one side alone reports, which it leaves out.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.composite import composite_refusal
from analysis_mcp.digest import render_compare_digest
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    UnitHeterogeneousError,
    UnitNotPresentError,
    UnknownEntityColumnError,
)
from analysis_mcp.engine.compare import (
    CompareAmbiguousSliceError,
    CompareFacetMismatchError,
    CompareNonFiniteResultError,
    CompareResult,
    CompareUnitMismatchError,
    CompareUnitUnboundError,
    ComparisonPeriodsDisjointError,
    run_compare,
)
from analysis_mcp.frame_ref_resolution import (
    AdmittedFrame,
    canonical_entity_binding,
    derive_frame_content_version,
    derived_marker_detail,
    holds_input_rows,
    holds_ratio,
    input_content_version,
    merge_derived_markers,
    merged_time,
    op_spec_hash_or_refusal,
    read_derived_marker,
    resolve_input_frames,
)
from analysis_mcp.identity import canonicalize_op_spec
from analysis_mcp.op_spec import CompareOpSpec, PinnedUnit, SumOfOpSpec
from analysis_mcp.settings import Settings
from analysis_mcp.steps import LeftOut, Step, build_step, left_out_from_exclusions, shared_step_unit
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import (
    build_op_spec_or_refusal,
    dedupe_input_content_versions,
    exclusions_for_trace,
    mint_identity_or_refusal,
    store_result_and_render,
)
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from analysis_mcp.units import unit_json

# Every typed exception `run_compare` or `run_sum_of` can raise maps to
# exactly one refusal by `_refusal_for_engine_error` below.
# `comparison_periods_disjoint`, `cross_frame_unit_mismatch`,
# `cross_frame_unit_unbound`, and `cross_frame_facet_mismatch` are the
# two-slice verbs' own; the rest reuse `aggregate`'s existing refusal
# factories directly. `run_sum_of` never raises `CompareFacetMismatchError`.
_EngineError = (
    CompareAmbiguousSliceError
    | ComparisonPeriodsDisjointError
    | CompareUnitMismatchError
    | CompareUnitUnboundError
    | CompareFacetMismatchError
    | CompareNonFiniteResultError
    | UnitHeterogeneousError
    | UnitNotPresentError
    | UnknownEntityColumnError
    | FacetHeterogeneousError
)
SLICE_PAIR_ERRORS = (
    CompareAmbiguousSliceError,
    ComparisonPeriodsDisjointError,
    CompareUnitMismatchError,
    CompareUnitUnboundError,
    CompareFacetMismatchError,
    CompareNonFiniteResultError,
    UnitNotPresentError,
    UnitHeterogeneousError,
    UnknownEntityColumnError,
    FacetHeterogeneousError,
)

# `resolve_unit_and_facets`/`validate_entity_columns` (`analysis_mcp.engine.
# common`), the shared helper both sides resolve through, don't thread a
# `side` parameter into `UnitHeterogeneousError`/`UnitNotPresentError`/
# `UnknownEntityColumnError` the way `_slice_by_period` threads its own
# `side` into `CompareAmbiguousSliceError`. Naming both sides' parameters
# together is more precise than aggregate's single-sided `unit` vocabulary.
_LEFT_OR_RIGHT_UNIT = "left_unit or right_unit"
_LEFT_OR_RIGHT_ENTITY = "left_entity or right_entity"


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_compare`'s or `run_sum_of`'s typed exceptions to its refusal.

    `CompareAmbiguousSliceError` reuses `role_ambiguous`: an `entity`
    binding that fails to narrow a side down to one row per period is, in
    the same sense `aggregate`'s own refusal already covers, the wrong
    grouping for what the caller asked -- named columns here are the
    binding's own keys, empty when the binding itself was empty and the
    frame's own multiple entities are what's ambiguous. `exc.side` names
    which of `left_entity` or `right_entity` was at fault.
    `UnknownEntityColumnError` carries no side, since the shared helper that
    raises it serves both. `CompareUnitMismatchError`
    is distinct from `UnitHeterogeneousError` (one frame, many units): here
    each side resolved cleanly to its own single unit, and it's the pairing
    across sides that's unsafe, so it gets its own `cross_frame_unit_mismatch`
    code rather than overloading `unit_heterogeneous`'s message.
    `CompareUnitUnboundError` is a narrower case of the same cross-side
    pairing: one side's own rows leave `unit` unbound entirely rather than
    resolving to a second, differing unit, so it gets its own
    `cross_frame_unit_unbound` code instead of `cross_frame_unit_mismatch`'s
    "two resolved units differ" phrasing. `CompareFacetMismatchError` is the
    identical cross-side shape applied to a facet axis instead of `unit`
    (`cross_frame_facet_mismatch`), distinct from `FacetHeterogeneousError`
    (one side's own slice internally heterogeneous on an axis).
    `CompareNonFiniteResultError` reuses `non_finite_result`, the same
    overflow discipline `aggregate` and `growth` already carry, naming the
    quantity that overflowed (`delta` or `sum`).
    """
    # An if/elif chain with one assignment and one return, rather than one
    # early return per branch, matching `aggregate_tool._refusal_for_engine_error`'s
    # own shape (which the workspace's `PLR0911` complexity gate requires).
    if isinstance(exc, CompareAmbiguousSliceError):
        refusal = ToolRefusal.role_ambiguous(
            parameter=f"{exc.side}_entity", columns=list(exc.entity), available=list(exc.available)
        )
    elif isinstance(exc, UnknownEntityColumnError):
        refusal = ToolRefusal.role_ambiguous(
            parameter=_LEFT_OR_RIGHT_ENTITY,
            columns=list(exc.columns),
            available=list(exc.available),
        )
    elif isinstance(exc, ComparisonPeriodsDisjointError):
        refusal = ToolRefusal.comparison_periods_disjoint(
            left_periods=list(exc.left_periods), right_periods=list(exc.right_periods)
        )
    elif isinstance(exc, CompareUnitMismatchError):
        refusal = ToolRefusal.cross_frame_unit_mismatch(
            left_label="left",
            left_unit=exc.left_unit,
            right_label="right",
            right_unit=exc.right_unit,
        )
    elif isinstance(exc, CompareUnitUnboundError):
        refusal = ToolRefusal.cross_frame_unit_unbound(
            unbound_label=exc.unbound_side,
            bound_label=exc.bound_side,
            bound_unit=exc.bound_unit,
        )
    elif isinstance(exc, CompareFacetMismatchError):
        refusal = ToolRefusal.cross_frame_facet_mismatch(
            left_label="left",
            right_label="right",
            axis=exc.axis,
            left_ref=exc.left_ref,
            right_ref=exc.right_ref,
            stat_vars=exc.stat_vars,
        )
    elif isinstance(exc, CompareNonFiniteResultError):
        refusal = ToolRefusal.non_finite_result(
            group_key={"time_period": exc.time_period}, aggregation=exc.quantity
        )
    elif isinstance(exc, UnitNotPresentError):
        refusal = ToolRefusal.unit_not_present(
            parameter=_LEFT_OR_RIGHT_UNIT, pinned_unit=exc.pinned_unit, units=list(exc.units)
        )
    elif isinstance(exc, UnitHeterogeneousError):
        refusal = ToolRefusal.unit_heterogeneous(
            parameter=_LEFT_OR_RIGHT_UNIT, units=list(exc.units)
        )
    else:
        refusal = ToolRefusal.facet_heterogeneous(axis=exc.axis, values=list(exc.values))
    return refusal


def _compare_step(
    op_spec: CompareOpSpec, result: CompareResult, *, trace: Trace, ratio: tuple[bool, bool]
) -> Step:
    """The step record of `result`, stored on `trace`: the periods it
    compared, the periods one side alone reports, the unit both sides share
    (`shared_step_unit`, with each side's `ratio` flag), and the entities
    each side holds."""
    one_sided = len(set(result.left_periods) ^ set(result.right_periods))
    return build_step(
        op="compare",
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
        unit=shared_step_unit(result.left_resolved_unit, result.right_resolved_unit, ratio=ratio),
        sides={"left": result.left_entities, "right": result.right_entities},
    )


async def resolve_slice_pair(
    op_spec: CompareOpSpec | SumOfOpSpec,
) -> tuple[AdmittedFrame, AdmittedFrame, list[str]] | ToolResult:
    """The left and right inputs `op_spec` names, and the distinct input
    content_versions its identity is minted over, or the resolution refusal.

    Both sides resolve in one call, so the call claims both sides' rows
    before decoding either (`resolve_input_frames`). With no
    `right_frame_ref` the right side is the left frame itself.
    """
    frame_refs = [op_spec.frame_ref]
    if op_spec.right_frame_ref is not None:
        frame_refs.append(op_spec.right_frame_ref)
    resolved = await resolve_input_frames(frame_refs, settings=Settings())
    if isinstance(resolved, ToolResult):
        return resolved
    left, right = resolved[0], resolved[-1]
    left_cv = input_content_version(left.trace)
    input_content_versions = (
        [left_cv]
        if op_spec.right_frame_ref is None
        else dedupe_input_content_versions(left_cv, input_content_version(right.trace))
    )
    return left, right, input_content_versions


@holds_input_rows
async def compare(
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
                "The right side's frame_ref pair. Omit to compare two slices of the same "
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
    """Compare a value between two slices over their intersected periods.

    Each side is `frame_ref` (or `right_frame_ref`) narrowed by an `entity`
    binding down to one row per `time_period`. The compared periods are the
    intersection of what each side actually offers; an empty intersection
    refuses (`comparison_periods_disjoint`), naming each side's own
    available periods. `delta` is `right - left` at each intersected
    period. The result is a variable of its own, right minus left. Use
    `join` instead to combine two different variables for one entity.

    Args:
        frame_ref: The left side's frame_ref.
        right_frame_ref: The right side's frame_ref; omit for two slices of frame_ref.
        left_entity: Narrows the left side to one entity's series.
        right_entity: Narrows the right side to one entity's series.
        left_unit: Pins the left side's unit.
        right_unit: Pins the right side's unit.

    Returns:
        A digest: each intersected period's two values and delta, both
        sides' coverage, and the result's {spec_id, content_version} pair
        and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: CompareOpSpec(
            frame_ref=frame_ref,
            right_frame_ref=right_frame_ref,
            # `{} if ... is None else ...` avoids a mutable `= {}` default on
            # the parameter itself (ruff B006). `CompareOpSpec.left_entity`/
            # `right_entity` use `Field(default_factory=dict)` for the same
            # "fresh dict per call" guarantee, but that default only resolves
            # through a schema-validating framework layer -- a caller
            # invoking this function directly would get the `FieldInfo`
            # object itself, not the dict it describes.
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
    left_frame, left_trace = left.frame, left.trace
    right_frame, right_trace = right.frame, right.trace

    # `left_entity`/`right_entity` are canonicalised (key order sorted)
    # before the engine runs -- see `frame_ref_resolution.canonical_entity_
    # binding`'s own docstring; the same "execution order must match the
    # order identity was minted from" discipline `aggregate_tool.aggregate`
    # already applies to `group_by`.
    try:
        result = run_compare(
            left_frame,
            right_frame,
            stat_var=f"dcid:{op_spec_hash}",
            left_entity=canonical_entity_binding(op_spec.left_entity),
            right_entity=canonical_entity_binding(op_spec.right_entity),
            left_pinned_unit=op_spec.left_unit,
            right_pinned_unit=op_spec.right_unit,
        )
    except SLICE_PAIR_ERRORS as exc:
        return to_tool_result(
            composite_refusal(_refusal_for_engine_error(exc), [left_trace, right_trace])
        )

    identity = mint_identity_or_refusal(op_spec, input_content_versions=input_content_versions)
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    detail: dict[str, object] = {
        "left_periods": list(result.left_periods),
        "right_periods": list(result.right_periods),
        "intersected_periods": list(result.intersected_periods),
        "left_coverage_fraction": result.left_coverage_fraction,
        "right_coverage_fraction": result.right_coverage_fraction,
        "left_stat_vars": list(result.left_stat_vars),
        "right_stat_vars": list(result.right_stat_vars),
        "right_resolved_unit": unit_json(result.right_resolved_unit),
        "right_resolved_facets": {
            selection.axis: selection.ref for selection in result.right_facet_selections
        },
    }
    # `compare` sets no marker of its own, since the result's `stat_var` is its
    # own. A marker either side carries is passed on.
    derived = derived_marker_detail(
        inherited=merge_derived_markers(
            read_derived_marker(left_trace), read_derived_marker(right_trace)
        )
    )
    if derived is not None:
        detail["derived"] = derived

    # `compare` sets no interval of its own: `delta` is a single subtraction
    # of two already-finite floats.
    time_fact = merged_time(own_interval=None, inherited=[left, right])

    # The left side's own upstream_trace is carried forward, the same
    # "left is the reference" choice `run_compare`'s own output frame
    # makes for unit/facets. Both sides' own frames are
    # bound into `content_version` via `input_content_versions` above, and
    # carried onto the trace's own `input_content_versions` field verbatim.
    output_trace = Trace(
        upstream_trace=left_trace.upstream_trace,
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="compare_intersect_periods",
            realised_coverage_fraction=min(
                result.left_coverage_fraction, result.right_coverage_fraction
            ),
            skipped_row_count=0,
            resolved_unit=unit_json(result.left_resolved_unit),
            resolved_facets={
                selection.axis: selection.ref for selection in result.left_facet_selections
            },
            detail=detail,
        ),
        exclusions=exclusions_for_trace(result.exclusions),
        time=time_fact,
        content_version=content_version,
        input_content_versions=input_content_versions,
        execution_identity=execution_identity,
    )

    return await store_result_and_render(
        frame=result.frame,
        trace=output_trace,
        step=_compare_step(
            op_spec,
            result,
            trace=output_trace,
            ratio=(holds_ratio(left_trace), holds_ratio(right_trace)),
        ),
        digest=lambda: render_compare_digest(
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
    )
