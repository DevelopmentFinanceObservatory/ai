"""The `quotient_of` tool: op-spec in, one stored-frame slice divided by
another per key and period, a `{spec_id, content_version}` pair and a trace
out.

Flow mirrors `share_of_tool.share_of`, with `sum_of_tool.sum_of`'s first
step: the op-spec hash is minted before the engine runs, since the result's
`stat_var` is `dcid:<op_spec_hash>`, the operation's own ref.

Three side rules need more than the engine sees. A side whose trace carries
`DerivedMarker.ordinal` refuses `derived_input_ordinal`, and one whose trace
carries `DerivedMarker.dimensionless` refuses
`quotient_of_side_unit_unsupported` with reason `dimensionless`; both are
checked when the sides resolve (`_resolve_sides`). A side that declares no
unit and whose StatVar is an operation's own ref refuses with reason
`operation_stat_var`. Whether a side declares a unit is settled only once
the engine has resolved it over the side's slice, pin included, so that
rule reads the composed unit the engine returns (`_operation_ref_side`),
before identity is minted and anything is stored. The engine raises the
`composed` reason itself.

`applied_policy.detail` carries each side's StatVar, resolved unit, facets
and periods, both coverage fractions, the matched periods, and four counts
of what the result leaves out: `unmatched_key_count`,
`keys_without_shared_period_count`, `keys_with_zero_denominator_count` and
`zero_denominator_cell_count`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from common.materialisation import ComposedStatVar
from dfo_frame_grammar.grammar import StatVarPart
from fastmcp.tools import ToolResult
from pydantic import Field

if TYPE_CHECKING:
    from analysis_mcp.engine.quotient_of import QuotientOfResult, Side
    from analysis_mcp.frame_ref_resolution import AdmittedFrame

from analysis_mcp.digest import render_quotient_of_digest
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    PlaceLevelsMixedError,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnitNotPresentError,
    UnknownEntityColumnError,
)
from analysis_mcp.engine.quotient_of import (
    QuotientOfNoKeyMatchedError,
    QuotientOfPeriodsDisjointError,
    QuotientOfSideUnitUnsupportedError,
    QuotientOfZeroDenominatorError,
    run_quotient_of,
)
from analysis_mcp.engine.two_sided import NonFiniteCellError
from analysis_mcp.frame_ref_resolution import (
    canonical_columns_or_refusal,
    canonical_entity_binding,
    derive_frame_content_version,
    derived_marker_detail,
    holds_input_rows,
    input_content_version,
    merge_derived_markers,
    merged_time,
    op_spec_hash_or_refusal,
    read_derived_marker,
    resolve_input_frames,
)
from analysis_mcp.identity import OP_SPEC_HASH_PATTERN, canonicalize_op_spec
from analysis_mcp.op_spec import PinnedUnit, QuotientOfOpSpec
from analysis_mcp.settings import Settings
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

# Every typed exception `run_quotient_of` can raise maps to exactly one
# refusal by `_refusal_for_engine_error` below. The first four are
# `quotient_of`'s own; the rest are the unit, facet, stat_var, entity and
# place-level rules every verb shares, and reuse `aggregate`'s factories.
_ENGINE_ERRORS = (
    QuotientOfSideUnitUnsupportedError,
    QuotientOfNoKeyMatchedError,
    QuotientOfPeriodsDisjointError,
    QuotientOfZeroDenominatorError,
    NonFiniteCellError,
    StatVarUnresolvedError,
    UnitNotPresentError,
    UnitHeterogeneousError,
    UnknownEntityColumnError,
    FacetHeterogeneousError,
    PlaceLevelsMixedError,
)
_EngineError = (
    QuotientOfSideUnitUnsupportedError
    | QuotientOfNoKeyMatchedError
    | QuotientOfPeriodsDisjointError
    | QuotientOfZeroDenominatorError
    | NonFiniteCellError
    | StatVarUnresolvedError
    | UnitNotPresentError
    | UnitHeterogeneousError
    | UnknownEntityColumnError
    | FacetHeterogeneousError
    | PlaceLevelsMixedError
)

# Both sides resolve through one shared helper that does not record which
# side a unit or entity error came from, so these name both parameters.
_NUMERATOR_OR_DENOMINATOR_UNIT = "numerator_unit or denominator_unit"
_NUMERATOR_OR_DENOMINATOR_ENTITY = "numerator_entity or denominator_entity"


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_quotient_of`'s typed exceptions to its refusal, in
    the one-assignment shape `share_of_tool._refusal_for_engine_error` has."""
    if isinstance(exc, QuotientOfSideUnitUnsupportedError):
        refusal = ToolRefusal.quotient_of_side_unit_unsupported(
            side=exc.side, reason=exc.reason, unit=exc.unit
        )
    elif isinstance(exc, QuotientOfNoKeyMatchedError):
        refusal = ToolRefusal.quotient_of_no_key_matched(
            numerator_keys=[dict(key) for key in exc.numerator_keys],
            denominator_keys=[dict(key) for key in exc.denominator_keys],
            numerator_key_count=exc.numerator_key_count,
            denominator_key_count=exc.denominator_key_count,
        )
    elif isinstance(exc, QuotientOfPeriodsDisjointError):
        refusal = ToolRefusal.comparison_periods_disjoint(
            left_periods=list(exc.numerator_periods),
            right_periods=list(exc.denominator_periods),
            left_label="numerator",
            right_label="denominator",
        )
    elif isinstance(exc, QuotientOfZeroDenominatorError):
        refusal = ToolRefusal.quotient_of_zero_denominator(
            time_periods=list(exc.time_periods), cell_count=exc.cell_count
        )
    elif isinstance(exc, NonFiniteCellError):
        # The engine leaves a zero denominator out, so this is an overflow.
        refusal = ToolRefusal.non_finite_result(
            group_key={"time_period": exc.time_period}, aggregation="quotient_of"
        )
    elif isinstance(exc, StatVarUnresolvedError):
        refusal = ToolRefusal.stat_var_unresolved(
            reason=exc.reason, distinct_stat_vars=list(exc.distinct_stat_vars)
        )
    elif isinstance(exc, UnitNotPresentError):
        refusal = ToolRefusal.unit_not_present(
            parameter=_NUMERATOR_OR_DENOMINATOR_UNIT,
            pinned_unit=exc.pinned_unit,
            units=list(exc.units),
        )
    elif isinstance(exc, UnitHeterogeneousError):
        refusal = ToolRefusal.unit_heterogeneous(
            parameter=_NUMERATOR_OR_DENOMINATOR_UNIT, units=list(exc.units)
        )
    elif isinstance(exc, UnknownEntityColumnError):
        refusal = ToolRefusal.role_ambiguous(
            parameter=_NUMERATOR_OR_DENOMINATOR_ENTITY,
            columns=list(exc.columns),
            available=list(exc.available),
        )
    elif isinstance(exc, PlaceLevelsMixedError):
        refusal = ToolRefusal.place_levels_mixed(column=exc.column, types=list(exc.types))
    else:
        refusal = ToolRefusal.facet_heterogeneous(axis=exc.axis, values=list(exc.values))
    return refusal


def _marker_refusal(side: Side, frame: AdmittedFrame) -> ToolRefusal | None:
    """The refusal `side`'s derived marker earns, or `None`: a rank
    position is no quantity to divide, and a share or a relative growth rate
    is a ratio already, no amount to be a part of the composed unit.
    """
    marker = read_derived_marker(frame.trace)
    if marker is None:
        return None
    if marker.ordinal:
        return ToolRefusal.derived_input_ordinal(
            reason=(
                f"the {side} frame_ref names a per-period rank's own output (or a result "
                "derived from one), whose value column holds a rank position"
            )
        )
    if marker.dimensionless:
        return ToolRefusal.quotient_of_side_unit_unsupported(side=side, reason="dimensionless")
    return None


async def _resolve_sides(
    *, op_spec: QuotientOfOpSpec, settings: Settings
) -> tuple[AdmittedFrame, AdmittedFrame, list[str]] | ToolResult:
    """Resolve the numerator and denominator in one `resolve_input_frames`
    call, so the call claims both sides' rows before decoding either, reusing
    the numerator's frame when `denominator_frame_ref` is `None`, and refuse
    a side its derived marker rules out (`_marker_refusal`). Returns both
    sides and the input content versions.
    """
    frame_refs = [op_spec.frame_ref]
    if op_spec.denominator_frame_ref is not None:
        frame_refs.append(op_spec.denominator_frame_ref)
    resolved = await resolve_input_frames(frame_refs, settings=settings)
    if isinstance(resolved, ToolResult):
        return resolved
    numerator, denominator = resolved[0], resolved[-1]

    sides: tuple[tuple[Side, AdmittedFrame], ...] = (
        ("numerator", numerator),
        ("denominator", denominator),
    )
    for side, frame in sides:
        refusal = _marker_refusal(side, frame)
        if refusal is not None:
            return to_tool_result(refusal)

    numerator_cv = input_content_version(numerator.trace)
    input_content_versions = (
        [numerator_cv]
        if op_spec.denominator_frame_ref is None
        else dedupe_input_content_versions(numerator_cv, input_content_version(denominator.trace))
    )
    return numerator, denominator, input_content_versions


def _divide_or_refusal(
    op_spec: QuotientOfOpSpec, *, numerator: AdmittedFrame, denominator: AdmittedFrame
) -> QuotientOfResult | ToolResult:
    """Run the engine over the two resolved sides, labelling the result
    `dcid:<op_spec_hash>`, or return the refusal of the first rule the call
    breaks: an op-spec that cannot be hashed, one of the engine's typed
    exceptions, or a unit-less side under an operation's own StatVar
    (`_operation_ref_side`).

    `numerator_by` and the bindings are canonicalised before the engine
    runs, so execution order matches the order `op_spec_hash` is minted
    over; see `share_of_tool.share_of`.
    """
    numerator_by = canonical_columns_or_refusal(op_spec.numerator_by, description="numerator_by")
    if isinstance(numerator_by, ToolResult):
        return numerator_by
    op_spec_hash = op_spec_hash_or_refusal(op_spec)
    if isinstance(op_spec_hash, ToolResult):
        return op_spec_hash
    try:
        result = run_quotient_of(
            numerator.frame,
            denominator.frame,
            stat_var=f"dcid:{op_spec_hash}",
            numerator_entity=canonical_entity_binding(op_spec.numerator_entity),
            denominator_entity=canonical_entity_binding(op_spec.denominator_entity),
            numerator_by=numerator_by,
            denominator_by=canonical_entity_binding(op_spec.denominator_by),
            numerator_pinned_unit=op_spec.numerator_unit,
            denominator_pinned_unit=op_spec.denominator_unit,
        )
    except _ENGINE_ERRORS as exc:
        return to_tool_result(_refusal_for_engine_error(exc))
    refusal = _operation_ref_side(result)
    return result if refusal is None else to_tool_result(refusal)


def _operation_ref_side(result: QuotientOfResult) -> ToolRefusal | None:
    """`quotient_of_side_unit_unsupported` with reason `operation_stat_var`
    for the first side whose part of `result.unit` is a StatVar that is an
    operation's own ref, or `None`. The engine writes a `StatVarPart` only
    for a side that resolved to no unit, so the part itself says the side
    declares none. An operation's ref is `dcid:` and an op-spec hash, the
    form `sum_of` and this verb write and no graph node or stored definition
    carries; the StatVar keeps that form through any verb that passes it on,
    where the last hop's trace names only its own operation.
    """
    parts: tuple[tuple[Side, object], ...] = (
        ("numerator", result.unit.numerator),
        ("denominator", result.unit.denominator),
    )
    for side, part in parts:
        if isinstance(part, StatVarPart) and OP_SPEC_HASH_PATTERN.fullmatch(
            part.stat_var.removeprefix("dcid:")
        ):
            return ToolRefusal.quotient_of_side_unit_unsupported(
                side=side, reason="operation_stat_var", stat_var=part.stat_var
            )
    return None


@holds_input_rows
async def quotient_of(
    frame_ref: Annotated[
        dict[str, str],
        Field(
            description=(
                "The numerator's {spec_id, content_version} pair, naming a materialised "
                "frame or a prior verb result. Obtained from get_data's or a prior verb's "
                "own digest -- never invented."
            )
        ),
    ],
    denominator_frame_ref: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "The denominator's frame_ref. Omit to divide two slices of the same "
                "frame_ref, narrowed by numerator_entity/denominator_entity instead."
            ),
        ),
    ] = None,
    numerator_entity: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{role_column: value} narrowing the numerator's rows before summing each "
                "period. Empty sums every row on this side per period. Role columns only "
                "-- stat_var and time_period are not accepted selectors."
            ),
        ),
    ] = None,
    denominator_entity: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Same as numerator_entity, for the denominator side."),
    ] = None,
    numerator_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the numerator's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
    denominator_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the denominator's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
    numerator_by: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Role column(s), on the numerator's frame, to key the output by -- one row "
                "per (key, period) instead of one row per period. These become the output "
                "frame's role columns. Empty (the default) computes a single series."
            ),
        ),
    ] = None,
    denominator_by: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{numerator_by column: denominator column holding the same entities}, "
                "pairing the two sides by name, so each key divides by its own counterpart. "
                "Every numerator_by column must be a key here. Omit (the default) to divide "
                "every numerator key by the denominator's one total per period."
            ),
        ),
    ] = None,
) -> ToolResult:
    """Divide one measure by another, per period both report, such as GDP by population.

    Each side is `frame_ref` (or `denominator_frame_ref`) narrowed by an
    `entity` binding and summed per period. The result's unit is composed
    from the two sides, "<numerator's unit> per <denominator's unit>", and is
    never one side's alone; a side that declares no unit is named by its
    stat_var ("USDollar per Count_Person"). The result is one series under
    its own stat_var. `numerator_by` keys the output, one series per key.
    `denominator_by` pairs each `numerator_by` column with the denominator
    column holding the same entities, by name; without it every key divides
    by the denominator's one total. A period only one side reports, a
    numerator key with no denominator match or no shared period, and a cell
    whose denominator sums to zero are left out, never counted as zero; the
    digest names them. The call refuses when nothing is left to divide
    (`comparison_periods_disjoint`, `quotient_of_no_key_matched`,
    `quotient_of_zero_denominator`) and when a side is a share, a growth
    rate or a quotient already (`quotient_of_side_unit_unsupported`). Use
    `share_of` for a part of its whole in one unit.

    Args:
        frame_ref: The numerator's frame_ref.
        denominator_frame_ref: The denominator's frame_ref; omit for a slice of frame_ref.
        numerator_entity: Narrows the numerator's rows before summing.
        denominator_entity: Narrows the denominator's rows before summing.
        numerator_unit: Pins the numerator's unit.
        denominator_unit: Pins the denominator's unit.
        numerator_by: Role column(s) to key the numerator's rows by.
        denominator_by: {numerator_by column: denominator column}, pairing by name.

    Returns:
        A digest: each matched (key, period)'s two values and quotient in
        the composed unit, each side's stat_var and unit, both coverages,
        what was left out, and the result's {spec_id, content_version} pair
        and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: QuotientOfOpSpec(
            frame_ref=frame_ref,
            denominator_frame_ref=denominator_frame_ref,
            # See `compare_tool.compare` for why `None` maps to `{}` here.
            numerator_entity={} if numerator_entity is None else numerator_entity,
            denominator_entity={} if denominator_entity is None else denominator_entity,
            numerator_unit=numerator_unit,
            denominator_unit=denominator_unit,
            numerator_by=() if numerator_by is None else tuple(numerator_by),
            denominator_by={} if denominator_by is None else denominator_by,
        )
    )
    if isinstance(op_spec, ToolResult):
        return op_spec

    resolved_sides = await _resolve_sides(op_spec=op_spec, settings=Settings())
    if isinstance(resolved_sides, ToolResult):
        return resolved_sides
    numerator, denominator, input_content_versions = resolved_sides

    result = _divide_or_refusal(op_spec, numerator=numerator, denominator=denominator)
    if isinstance(result, ToolResult):
        return result

    identity = mint_identity_or_refusal(op_spec, input_content_versions=input_content_versions)
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    detail: dict[str, object] = {
        "numerator_stat_var": result.numerator_stat_var,
        "denominator_stat_var": result.denominator_stat_var,
        "numerator_resolved_unit": unit_json(result.numerator_resolved_unit),
        "denominator_resolved_unit": unit_json(result.denominator_resolved_unit),
        "numerator_resolved_facets": {
            selection.axis: selection.ref for selection in result.numerator_facet_selections
        },
        "denominator_resolved_facets": {
            selection.axis: selection.ref for selection in result.denominator_facet_selections
        },
        "numerator_periods": list(result.numerator_periods),
        "denominator_periods": list(result.denominator_periods),
        "matched_periods": list(result.matched_periods),
        "numerator_coverage_fraction": result.numerator_coverage_fraction,
        "denominator_coverage_fraction": result.denominator_coverage_fraction,
        "unmatched_key_count": len(result.unmatched_keys),
        "keys_without_shared_period_count": len(result.keys_without_shared_period),
        "keys_with_zero_denominator_count": len(result.keys_with_zero_denominator),
        "zero_denominator_cell_count": len(result.zero_denominator_cells),
    }
    # The result's unit and `stat_var` are its own, and a later verb reads
    # the unit from the frame, so `quotient_of` sets no marker of its own. A
    # marker inherited from either side carries forward.
    derived = derived_marker_detail(
        inherited=merge_derived_markers(
            read_derived_marker(numerator.trace), read_derived_marker(denominator.trace)
        )
    )
    if derived is not None:
        detail["derived"] = derived

    # The numerator's upstream_trace is carried forward, as `share_of` does.
    # Both sides' frames are bound into `content_version` through
    # `input_content_versions`.
    output_trace = Trace(
        upstream_trace=numerator.trace.upstream_trace,
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="quotient_of_intersect_periods",
            realised_coverage_fraction=min(
                result.numerator_coverage_fraction, result.denominator_coverage_fraction
            ),
            skipped_row_count=0,
            resolved_unit=unit_json(result.unit),
            resolved_facets={
                selection.axis: selection.ref for selection in result.facet_selections
            },
            detail=detail,
        ),
        exclusions=exclusions_for_trace(result.exclusions),
        time=merged_time(own_interval=None, inherited=[numerator, denominator]),
        content_version=content_version,
        input_content_versions=input_content_versions,
        execution_identity=execution_identity,
    )

    return await store_result_and_render(
        frame=result.frame,
        trace=output_trace,
        input_traces=[numerator.trace, denominator.trace],
        own_composed_stat_var=ComposedStatVar(
            stat_var=f"dcid:{op_spec_hash}",
            relation="quotient",
            first=(result.numerator_stat_var,),
            second=(result.denominator_stat_var,),
        ),
        digest=lambda: render_quotient_of_digest(
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
        dropped_key_count=result.dropped_key_count,
    )
