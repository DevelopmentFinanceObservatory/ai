"""The `share_of` tool: op-spec in, deterministic engine execution over two
stored-frame slices, a `{spec_id, content_version}` pair and a trace out.

Flow mirrors `compare_tool.compare`: parse/validate the op-spec -> resolve
the numerator `frame_ref` via the shared `analysis_mcp.frame_ref_resolution`
helpers, and the denominator `denominator_frame_ref` too when it names a
different result (left alone stands for "two slices of one frame_ref", the
same unification `CompareOpSpec`'s own docstring explains) -> run the engine
(`analysis_mcp.engine.share_of.run_share_of`), translating each of its typed
exceptions to its own refusal -> mint identity, build the trace, write the
store under the result's own pair -> render the digest. The result's rows sit
under `dcid:<op_spec_hash>`, the operation's own StatVar, so the hash is minted
before the engine runs. Every typed failure this
function itself anticipates is returned via `to_tool_result`, never raised.
An exception outside that set would escape this function; `server.py`'s
registration-site wrapper is the backstop that turns it into an
`internal_error` refusal instead of a raw traceback reaching the wire;
this backstop complements the typed handling below rather than replacing it.

The stored step (`_share_step`) counts the numerator keys and periods the
share keeps, the entities each period's denominator sums, and the values one
side alone reports, which it leaves out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from common.unit_refs import PERCENT_UNIT
from fastmcp.tools import ToolResult
from pydantic import Field

if TYPE_CHECKING:
    from common.materialisation import StoredFrameTrace

    from analysis_mcp.frame_ref_resolution import AdmittedFrame

from analysis_mcp.digest import render_share_of_digest
from analysis_mcp.engine.aggregate import ComposedUnitSumError
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    PlaceLevelsMixedError,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnitNotPresentError,
    UnknownEntityColumnError,
)
from analysis_mcp.engine.share_of import (
    ShareOfFacetMismatchError,
    ShareOfGrainMismatchError,
    ShareOfKeyUnknownError,
    ShareOfKeyUnmatchedError,
    ShareOfNonFiniteResultError,
    ShareOfResult,
    ShareOfUnitMismatchError,
    ShareOfUnitUnboundError,
    run_share_of,
)
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
from analysis_mcp.identity import canonicalize_op_spec
from analysis_mcp.op_spec import PinnedUnit, ShareOfOpSpec
from analysis_mcp.settings import Settings
from analysis_mcp.steps import Step, build_step, left_out_from_exclusions, step_unit
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

# Every typed exception `run_share_of` can raise maps to exactly one
# refusal by `_refusal_for_engine_error` below. `share_of_grain_mismatch`,
# `share_of_zero_denominator`, `share_of_key_unknown`,
# `share_of_key_unmatched`, `cross_frame_unit_mismatch`,
# `cross_frame_unit_unbound`, and `cross_frame_facet_mismatch` are what
# share_of introduces or reuses in a new way; the others reuse
# `aggregate`'s existing refusal factories directly, the same unit/facet/
# stat_var/entity resolution rule every verb here shares, and
# `composed_unit_sum_unsupported` names the side as `sum_of`'s does.
_EngineError = (
    ShareOfGrainMismatchError
    | ShareOfKeyUnknownError
    | ShareOfKeyUnmatchedError
    | ShareOfNonFiniteResultError
    | ShareOfUnitMismatchError
    | ShareOfUnitUnboundError
    | ShareOfFacetMismatchError
    | StatVarUnresolvedError
    | UnitHeterogeneousError
    | UnitNotPresentError
    | UnknownEntityColumnError
    | FacetHeterogeneousError
    | PlaceLevelsMixedError
    | ComposedUnitSumError
)

# `run_share_of` never records which side (numerator/denominator) an
# UnitHeterogeneousError/UnitNotPresentError/UnknownEntityColumnError came
# from -- both sides resolve through the identical shared helper. Naming
# both of share_of's own parameters together is strictly more accurate
# than aggregate's single `unit`/no entity vocabulary, which share_of does
# not have.
_NUMERATOR_OR_DENOMINATOR_UNIT = "numerator_unit or denominator_unit"
_NUMERATOR_OR_DENOMINATOR_ENTITY = "numerator_entity or denominator_entity"


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_share_of`'s typed exceptions to its refusal.

    An if/elif chain with one assignment and one return, rather than one
    early return per branch, matching `aggregate_tool._refusal_for_engine_error`'s
    own shape (which the workspace's `PLR0911` complexity gate requires).
    """
    if isinstance(exc, ShareOfGrainMismatchError):
        refusal = ToolRefusal.share_of_grain_mismatch(
            numerator_periods=list(exc.numerator_periods),
            denominator_periods=list(exc.denominator_periods),
            key=dict(exc.key),
        )
    elif isinstance(exc, ShareOfKeyUnknownError):
        refusal = ToolRefusal.share_of_key_unknown(
            column=exc.column,
            numerator_by=list(exc.numerator_by),
            denominator_by_keys=list(exc.denominator_by_keys),
        )
    elif isinstance(exc, ShareOfKeyUnmatchedError):
        refusal = ToolRefusal.share_of_key_unmatched(
            key=dict(exc.key),
            available=[dict(candidate) for candidate in exc.available],
        )
    elif isinstance(exc, ShareOfNonFiniteResultError):
        # `reason == "zero denominator"` is retryable (a different
        # denominator frame_ref/entity can sum to nonzero); a genuine
        # overflow is not, so only the latter reuses `non_finite_result`.
        if exc.reason == "zero denominator":
            refusal = ToolRefusal.share_of_zero_denominator(time_period=exc.time_period)
        else:
            refusal = ToolRefusal.non_finite_result(
                group_key={"time_period": exc.time_period}, aggregation="share_of"
            )
    elif isinstance(exc, ShareOfUnitMismatchError):
        refusal = ToolRefusal.cross_frame_unit_mismatch(
            left_label="numerator",
            left_unit=exc.numerator_unit,
            right_label="denominator",
            right_unit=exc.denominator_unit,
        )
    elif isinstance(exc, ShareOfUnitUnboundError):
        refusal = ToolRefusal.cross_frame_unit_unbound(
            unbound_label=exc.unbound_side,
            bound_label=exc.bound_side,
            bound_unit=exc.bound_unit,
        )
    elif isinstance(exc, ShareOfFacetMismatchError):
        refusal = ToolRefusal.cross_frame_facet_mismatch(
            left_label="numerator",
            right_label="denominator",
            axis=exc.axis,
            left_ref=exc.numerator_ref,
            right_ref=exc.denominator_ref,
            stat_vars=(exc.stat_var,),
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
        refusal = ToolRefusal.place_levels_mixed(
            column=exc.column,
            types=list(exc.types),
            whole_total=exc.whole_total,
            whole_class=exc.whole_class,
            member_count=exc.member_count,
            members=list(exc.members),
        )
    elif isinstance(exc, ComposedUnitSumError):
        refusal = ToolRefusal.composed_unit_sum_unsupported(
            verb="share_of", unit=exc.unit, side=exc.side, sums_a_cell=True
        )
    else:
        refusal = ToolRefusal.facet_heterogeneous(axis=exc.axis, values=list(exc.values))
    return refusal


def _ordinal_side_refusal(
    numerator_trace: StoredFrameTrace, denominator_trace: StoredFrameTrace
) -> ToolResult | None:
    """`derived_input_ordinal`, checked over both sides independently, or
    `None` when neither carries `DerivedMarker.ordinal` -- a share over a
    per-period rank's own output (or a result derived from one) is
    meaningless on either side.
    """
    for side_trace, side_label in (
        (numerator_trace, "numerator"),
        (denominator_trace, "denominator"),
    ):
        side_marker = read_derived_marker(side_trace)
        if side_marker is not None and side_marker.ordinal:
            return to_tool_result(
                ToolRefusal.derived_input_ordinal(
                    reason=(
                        f"the {side_label} frame_ref names a per-period rank's own output "
                        "(or a result derived from one), whose value column holds a rank "
                        "position"
                    )
                )
            )
    return None


async def _resolve_sides(
    *, op_spec: ShareOfOpSpec, settings: Settings
) -> tuple[AdmittedFrame, AdmittedFrame, list[str]] | ToolResult:
    """Resolve the numerator and denominator sides in one `resolve_input_frames`
    call, so the call claims both sides' rows before decoding either, reusing
    the numerator's own frame when `denominator_frame_ref` is `None` (the
    same unification `ShareOfOpSpec`'s own docstring explains), and check
    both sides for `DerivedMarker.ordinal` in the identical step -- one
    guard, so `share_of` itself pays this combined resolve-and-check step
    with one `return`, not two, under the workspace's `PLR0911` complexity
    gate. Returns both sides and the input content versions.
    """
    frame_refs = [op_spec.frame_ref]
    if op_spec.denominator_frame_ref is not None:
        frame_refs.append(op_spec.denominator_frame_ref)
    resolved = await resolve_input_frames(frame_refs, settings=settings)
    if isinstance(resolved, ToolResult):
        return resolved
    numerator, denominator = resolved[0], resolved[-1]
    numerator_trace, denominator_trace = numerator.trace, denominator.trace
    numerator_cv = input_content_version(numerator_trace)
    input_content_versions = (
        [numerator_cv]
        if op_spec.denominator_frame_ref is None
        else dedupe_input_content_versions(numerator_cv, input_content_version(denominator_trace))
    )

    ordinal_refusal = _ordinal_side_refusal(numerator_trace, denominator_trace)
    if ordinal_refusal is not None:
        return ordinal_refusal

    return numerator, denominator, input_content_versions


def _share_step(op_spec: ShareOfOpSpec, result: ShareOfResult, *, trace: Trace) -> Step:
    """The step record of `result`, stored on `trace`: the numerator keys and
    periods it kept, each period's denominator coverage, the values it left
    out, and per cent."""
    coverage = result.coverage
    return build_step(
        op="share",
        trace=trace,
        inputs={
            "numerator": op_spec.frame_ref,
            "denominator": op_spec.denominator_frame_ref or op_spec.frame_ref,
        },
        kept={
            "entities": coverage.entities,
            "periods": coverage.periods,
            "entities_with_fewer_periods": coverage.entities_with_fewer_periods,
        },
        left_out=left_out_from_exclusions(result.exclusions),
        unit=step_unit(PERCENT_UNIT),
        period_entities=coverage.period_entities,
    )


def _share_or_refusal(
    op_spec: ShareOfOpSpec, *, numerator: AdmittedFrame, denominator: AdmittedFrame
) -> ShareOfResult | ToolResult:
    """Run the engine over the two resolved sides, labelling the result
    `dcid:<op_spec_hash>`, or return the refusal of the first rule the call
    breaks: a `numerator_by` that cannot be canonicalised, an op-spec that
    cannot be hashed, or one of the engine's typed exceptions.

    `numerator_by` is canonicalised before the engine runs, so execution
    matches the order `identity._normalise_parameters` already normalises it
    to before minting `op_spec_hash` (`aggregate_tool.aggregate` has the
    identical comment for `group_by`). `denominator_by` and both entity
    bindings need no canonicalisation for correctness (their lookups are by
    key, not iteration order), but go through `canonical_entity_binding` for
    a stable execution-side key order, the same "canonicalise before
    executing" discipline every array/object parameter here follows.
    """
    numerator_by = canonical_columns_or_refusal(op_spec.numerator_by, description="numerator_by")
    if isinstance(numerator_by, ToolResult):
        return numerator_by
    op_spec_hash = op_spec_hash_or_refusal(op_spec)
    if isinstance(op_spec_hash, ToolResult):
        return op_spec_hash
    try:
        return run_share_of(
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
    except (
        ShareOfGrainMismatchError,
        ShareOfKeyUnknownError,
        ShareOfKeyUnmatchedError,
        ShareOfNonFiniteResultError,
        ShareOfUnitMismatchError,
        ShareOfUnitUnboundError,
        ShareOfFacetMismatchError,
        StatVarUnresolvedError,
        UnitNotPresentError,
        UnitHeterogeneousError,
        UnknownEntityColumnError,
        FacetHeterogeneousError,
        PlaceLevelsMixedError,
        ComposedUnitSumError,
    ) as exc:
        return to_tool_result(_refusal_for_engine_error(exc))


@holds_input_rows
async def share_of(
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
                "The denominator's frame_ref. Omit to compute a share within two slices of "
                "the same frame_ref, narrowed by numerator_entity/denominator_entity instead."
            ),
        ),
    ] = None,
    numerator_entity: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{role_column: value} narrowing the numerator's own rows before summing "
                "each period. Empty sums every row on this side per period -- the usual "
                "shape for a deliberate total, e.g. 'all DAC donors'. Role columns only -- "
                "stat_var and time_period are not accepted selectors."
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
                "Role column(s), on the numerator's own frame, to key the output by -- "
                "one row per (key, period) instead of one row per period. These become "
                "the output frame's own role columns. Empty (the default) computes "
                "a single figure, with no role columns."
            ),
        ),
    ] = None,
    denominator_by: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{numerator_by column: denominator column carrying that same role}, "
                "pairing the two sides by name, never by position. Every numerator_by "
                "column must be a key here (a partial mapping refuses). Omit (the "
                "default) for a pooled denominator: one total per period, shared by "
                "every numerator key."
            ),
        ),
    ] = None,
) -> ToolResult:
    """Compute the numerator's share of the denominator, per matched (key, period).

    The result is in per cent, unit `Percent` (5.3 means 5.3%), the unit a
    published share such as a "% of GDP" series carries, under its own
    stat_var. Each side is narrowed by its own `entity` binding, then every remaining
    row sharing both `numerator_by`'s own key values and a `time_period` is
    summed (`math.fsum`); more than one row per (key, period) is a
    deliberate total here, though `compare` treats it as an ambiguity.
    `numerator_by` left empty (the default) computes a single-figure share,
    one row per period, no role columns. `denominator_by` pairs each
    `numerator_by` column with the denominator's own column carrying that
    role, by name; left empty, the
    denominator is pooled (one total per period, shared by every numerator
    key). A `denominator_by` key outside `numerator_by` refuses
    (`op_spec_invalid`); a non-empty `denominator_by` that leaves a
    `numerator_by` column out refuses (`share_of_key_unknown`); a numerator
    key with no matching denominator group under that mapping refuses
    (`share_of_key_unmatched`). Each key divides over the periods both
    sides report for it, and every period one side holds alone is left out
    and recorded as an exclusion on that side, as `quotient_of` records it;
    a key sharing no period refuses (`share_of_grain_mismatch`), naming
    both sides' periods and the key. A period whose denominator sums to zero
    refuses (`share_of_zero_denominator`) before any division is attempted,
    and is retryable, since a different denominator source can sum to
    nonzero; a genuine overflow (`non_finite_result`) is not. One measure
    per unit of another in two different units, such as GDP per person, is
    `quotient_of`.

    Args:
        frame_ref: The numerator's frame_ref.
        denominator_frame_ref: The denominator's frame_ref; omit for a slice of frame_ref.
        numerator_entity: Narrows the numerator's rows before summing.
        denominator_entity: Narrows the denominator's rows before summing.
        numerator_unit: Pins the numerator's unit.
        denominator_unit: Pins the denominator's unit.
        numerator_by: Role column(s) to key the numerator's own rows by.
        denominator_by: {numerator_by column: denominator column}, pairing by name.

    Returns:
        A digest: the first and latest share, each matched (key,
        period)'s numerator/denominator values and share in per cent, each
        side's stat_var and unit, and the result's {spec_id,
        content_version} pair and identity fields. On refusal, a typed
        error with a machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: ShareOfOpSpec(
            frame_ref=frame_ref,
            denominator_frame_ref=denominator_frame_ref,
            # `{} if ... is None else ...`, not a mutable `= {}` default on
            # the parameter itself (ruff B006) -- see `compare_tool.compare`'s
            # own identical comment for why a bare `Field(default_factory=
            # dict)` can't stand in as this function's own Python-level
            # default.
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

    # Resolves both sides (the denominator reuses the numerator's own frame
    # when `denominator_frame_ref` is `None`) and checks both for
    # `DerivedMarker.ordinal` in the same step -- a share over a per-period
    # rank's own output is meaningless on either side, and unlike growth,
    # share_of has no retry that fixes this (there is no "absolute" reading
    # of a share), so the refusal names only the reason.
    resolved_sides = await _resolve_sides(op_spec=op_spec, settings=Settings())
    if isinstance(resolved_sides, ToolResult):
        return resolved_sides
    numerator, denominator, input_content_versions = resolved_sides
    numerator_trace, denominator_trace = numerator.trace, denominator.trace

    result = _share_or_refusal(op_spec, numerator=numerator, denominator=denominator)
    if isinstance(result, ToolResult):
        return result

    identity = mint_identity_or_refusal(op_spec, input_content_versions=input_content_versions)
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    # The derived-ness marker (see `frame_ref_resolution.DerivedMarker`): a
    # share is dimensionless, a per cent of its own denominator, so
    # `quotient_of` refuses it as a side. Its unit and `stat_var` are its
    # own. Routed through `derived_marker_detail`, merging both sides' own
    # markers (`merge_derived_markers`), so a flag either side already
    # carried survives onto this call's own output too.
    derived = derived_marker_detail(
        inherited=merge_derived_markers(
            read_derived_marker(numerator_trace), read_derived_marker(denominator_trace)
        ),
        dimensionless=True,
    )

    # `share_of` sets no interval of its own: each period's share is a
    # single division of two already-summed floats.
    time_fact = merged_time(own_interval=None, inherited=[numerator, denominator])

    # The numerator side's own upstream_trace is carried forward -- the same
    # "the first-named side is the reference" choice `compare_tool.compare`
    # already makes for its own left side. Both sides' own frames are bound
    # into `content_version` via `input_content_versions` above, and carried
    # onto the trace's own `input_content_versions` field verbatim.
    output_trace = Trace(
        upstream_trace=numerator_trace.upstream_trace,
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="share_of_intersect_periods",
            realised_coverage_fraction=min(
                result.numerator_coverage_fraction, result.denominator_coverage_fraction
            ),
            skipped_row_count=0,
            resolved_unit=PERCENT_UNIT,
            resolved_facets={
                selection.axis: selection.ref for selection in result.numerator_facet_selections
            },
            detail={
                "numerator_stat_var": result.numerator_stat_var,
                "denominator_stat_var": result.denominator_stat_var,
                "numerator_resolved_unit": unit_json(result.numerator_resolved_unit),
                "numerator_periods": list(result.numerator_periods),
                "denominator_periods": list(result.denominator_periods),
                "matched_periods": list(result.matched_periods),
                "numerator_coverage_fraction": result.numerator_coverage_fraction,
                "denominator_coverage_fraction": result.denominator_coverage_fraction,
                "denominator_resolved_unit": unit_json(result.denominator_resolved_unit),
                "denominator_resolved_facets": {
                    selection.axis: selection.ref
                    for selection in result.denominator_facet_selections
                },
                "derived": derived,
            },
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
        step=_share_step(op_spec, result, trace=output_trace),
        digest=lambda: render_share_of_digest(
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
    )
