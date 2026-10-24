"""The `join` tool: op-spec in, deterministic engine execution over two
stored frames, a `{spec_id, content_version}` pair and a trace out.

Flow mirrors `compare_tool.compare`, but with a required `right_frame_ref`
(see `JoinOpSpec`'s own docstring for why join has no "two slices of one
frame" reading): parse/validate the op-spec -> resolve both `frame_ref`s via
the shared `analysis_mcp.frame_ref_resolution` helpers -> run the engine
(`analysis_mcp.engine.join.run_join`), translating each of its typed
exceptions to its own refusal -> mint identity, build the trace, write the
store under the result's own pair -> render the digest. Every typed failure this
function itself anticipates is returned via `to_tool_result`, never raised.
An exception outside that set would escape this function; `server.py`'s
registration-site wrapper is the backstop that turns it into an
`internal_error` refusal instead of a raw traceback reaching the wire;
this backstop complements the typed handling below rather than replacing it.

The stored step (`_join_step`) counts the key-period pairs the join keeps
and the values it leaves out, unmatched or unit-less.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.digest import render_join_digest
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnitNotPresentError,
)
from analysis_mcp.engine.join import (
    JoinAmbiguousKeyError,
    JoinNoCommonKeyColumnsError,
    JoinOutputCollisionError,
    JoinResult,
    JoinResultTooLargeError,
    JoinUnknownKeyError,
    run_join,
)
from analysis_mcp.frame_ref_resolution import (
    canonical_columns_or_refusal,
    derive_frame_content_version,
    derived_marker_detail,
    holds_input_rows,
    holds_ratio,
    input_content_version,
    merge_derived_markers,
    merged_time,
    read_derived_marker,
    resolve_input_frames,
)
from analysis_mcp.identity import canonicalize_op_spec
from analysis_mcp.op_spec import JoinOpSpec, PinnedUnit
from analysis_mcp.settings import Settings
from analysis_mcp.steps import Step, build_step, left_out_from_exclusions, shared_step_unit
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

# Every typed exception `run_join` can raise maps to exactly one
# refusal by `_refusal_for_engine_error` below. `join_output_collision` and
# `join_no_common_key_columns` are new; every other code reused here
# already exists on `ToolRefusal`.
_EngineError = (
    JoinUnknownKeyError
    | JoinAmbiguousKeyError
    | JoinOutputCollisionError
    | JoinNoCommonKeyColumnsError
    | JoinResultTooLargeError
    | StatVarUnresolvedError
    | UnitNotPresentError
    | UnitHeterogeneousError
    | FacetHeterogeneousError
)

# `resolve_unit_and_facets` (`analysis_mcp.engine.common`), the shared
# helper both sides resolve through, doesn't thread a `side` parameter into
# `UnitHeterogeneousError`/`UnitNotPresentError` the way `_keyed_rows`
# threads its own `side` into `JoinAmbiguousKeyError`. Naming both sides'
# parameters together is more precise than aggregate's single-sided `unit`
# vocabulary.
_LEFT_OR_RIGHT_UNIT = "left_unit or right_unit"


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_join`'s typed exceptions to its refusal.

    `JoinUnknownKeyError` and `JoinAmbiguousKeyError` both reuse
    `role_ambiguous` -- the first because `join_keys` (or `right_keys`)
    names a column a frame it keys has no grain for, the same shape
    `aggregate`'s `group_by` refusal already covers; the second because a key that fails
    to uniquely identify one row per side is, in the same sense, the wrong
    key list for what the caller is asking. `JoinOutputCollisionError` is
    one condition genuinely specific to `join`: no other verb pairs two
    independently-resolved sides into one output row per matched key.
    `JoinNoCommonKeyColumnsError` is the other: auto-detection (empty
    `join_keys`) found no role column both frames declare in common while
    both sides declare some -- proceeding on `time_period` alone would
    silently pair unrelated rows. `JoinResultTooLargeError` is the store's
    own row-ceiling refusal on write (`tool_pipeline.store_result_and_render`),
    raised before the output is built rather than after, with the same
    payload: `Settings.max_frame_rows` defaults to the store's ceiling.
    """
    # An if/elif chain with one assignment and one return, rather than one
    # early return per branch, matching `aggregate_tool._refusal_for_engine_error`'s
    # own shape (which the workspace's `PLR0911` complexity gate requires).
    if isinstance(exc, JoinUnknownKeyError):
        refusal = ToolRefusal.role_ambiguous(
            parameter=exc.parameter, columns=list(exc.columns), available=list(exc.available)
        )
    elif isinstance(exc, JoinAmbiguousKeyError):
        refusal = ToolRefusal.role_ambiguous(
            parameter="join_keys", columns=list(exc.key_columns), available=list(exc.available)
        )
    elif isinstance(exc, JoinOutputCollisionError):
        refusal = ToolRefusal.join_output_collision(
            left_stat_var=exc.left_stat_var, right_stat_var=exc.right_stat_var
        )
    elif isinstance(exc, JoinNoCommonKeyColumnsError):
        refusal = ToolRefusal.join_no_common_key_columns(
            left_columns=list(exc.left_columns), right_columns=list(exc.right_columns)
        )
    elif isinstance(exc, JoinResultTooLargeError):
        refusal = ToolRefusal.frame_too_large_on_write(
            reason=f"frame has {exc.row_count} rows, over this store's {exc.ceiling}-row ceiling"
        )
    elif isinstance(exc, StatVarUnresolvedError):
        refusal = ToolRefusal.stat_var_unresolved(
            reason=exc.reason, distinct_stat_vars=list(exc.distinct_stat_vars)
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


def _join_step(
    op_spec: JoinOpSpec, result: JoinResult, *, trace: Trace, ratio: tuple[bool, bool]
) -> Step:
    """The step record of `result`, stored on `trace`: the key-period pairs
    it matched, the values each side holds that the other does not match or
    that declare no unit, and the unit both sides share (`shared_step_unit`,
    with each side's `ratio` flag). Sides in different units each keep their
    own, so the step names none."""
    return build_step(
        op="join",
        trace=trace,
        inputs={"left": op_spec.frame_ref, "right": op_spec.right_frame_ref},
        kept={"entity_periods": len(result.matched_keys)},
        left_out=left_out_from_exclusions(
            result.exclusions, row_reasons=("unit_unbound", "unmatched_key")
        ),
        unit=shared_step_unit(result.left_resolved_unit, result.right_resolved_unit, ratio=ratio),
    )


@holds_input_rows
async def join(
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
        dict[str, str],
        Field(
            description="The right side's frame_ref pair -- required; join always needs two frames."
        ),
    ],
    join_keys: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Role columns to join on, beyond time_period (always matched implicitly). "
                "Empty auto-detects every role column both frames declare in common."
            ),
        ),
    ] = (),
    right_keys: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{join_keys column: right frame's column holding the same entities}, "
                "pairing the two sides by name, so Kenya under observationAbout on the left "
                "matches Kenya under destination on the right. Every join_keys column must "
                "be a key here. The output keeps the left names. Omit (the default) to match "
                "each join_keys column under its own name on both sides."
            ),
        ),
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
    """Inner-join `frame_ref` and `right_frame_ref` on `join_keys` (or every
    shared role column, auto-detected) plus `time_period`.

    `right_keys` pairs a left column with a right column of another name
    holding the same entities, as `quotient_of`'s `denominator_by` does:
    `join_keys=["observationAbout"]`, `right_keys={"observationAbout":
    "destination"}` joins a WHO series to a CRS series where both name
    Kenya. Values match by equality, so Kenya against Uganda matches
    nothing and both sides' keys are dropped. Auto-detection pairs
    same-name columns only.

    A joined pair of rows never collapses into one row: the result frame is
    long-format, one row per matched key per side, `stat_var` distinguishing
    which side a row's own value came from -- `Frame` allows exactly one
    `value` column, so this is the only representable shape. A key present
    on one side and absent on the other is dropped (inner join); both
    sides' own dropped counts are recorded in the trace and surfaced as a
    digest warning, never a fabricated null for the missing side.

    Auto-detection picks up any entity-identifying role column (a reporting
    country, say) both frames declare, so joining two different entities'
    own series of the same variable auto-detects that column as a join key,
    every row's key value differs between the two sides, and the join
    matches zero rows on both sides. Use `compare` instead to combine one
    variable across two entities; `join` is for combining two different
    variables for one entity.

    Args:
        frame_ref: The left side's frame_ref.
        right_frame_ref: The right side's frame_ref.
        join_keys: Role columns to match on, beyond time_period; empty auto-detects.
        right_keys: {join_keys column: right column}, pairing by name.
        left_unit: Pins the left side's unit.
        right_unit: Pins the right side's unit.

    Returns:
        A digest: each matched key's left/right values, both sides' units,
        the dropped-key counts, and the result's {spec_id, content_version}
        pair and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: JoinOpSpec(
            frame_ref=frame_ref,
            right_frame_ref=right_frame_ref,
            join_keys=tuple(join_keys),
            # See `compare_tool.compare` for why `None` maps to `{}` here.
            right_keys={} if right_keys is None else right_keys,
            left_unit=left_unit,
            right_unit=right_unit,
        )
    )
    if isinstance(op_spec, ToolResult):
        return op_spec

    # `join_keys` is canonicalised before the engine runs -- see
    # `aggregate_tool.aggregate`'s identical comment on its own `group_by`.
    # A no-op when `join_keys` is left empty (auto-detect): `run_join`'s own
    # `_resolve_join_keys` already returns a sorted intersection in that
    # case.
    join_keys_canonical = canonical_columns_or_refusal(op_spec.join_keys, description="join_keys")
    if isinstance(join_keys_canonical, ToolResult):
        return join_keys_canonical

    # Both sides in one call, so the call claims both sides' rows before
    # decoding either (`resolve_input_frames`).
    settings = Settings()
    resolved = await resolve_input_frames(
        (op_spec.frame_ref, op_spec.right_frame_ref), settings=settings
    )
    if isinstance(resolved, ToolResult):
        return resolved
    left, right = resolved
    left_frame, left_trace = left.frame, left.trace
    right_frame, right_trace = right.frame, right.trace
    left_cv = input_content_version(left_trace)
    right_cv = input_content_version(right_trace)
    input_content_versions = dedupe_input_content_versions(left_cv, right_cv)

    try:
        result = run_join(
            left_frame,
            right_frame,
            join_keys=join_keys_canonical,
            right_keys=op_spec.right_keys,
            left_pinned_unit=op_spec.left_unit,
            right_pinned_unit=op_spec.right_unit,
            max_output_rows=settings.max_frame_rows,
        )
    except (
        JoinUnknownKeyError,
        JoinAmbiguousKeyError,
        JoinOutputCollisionError,
        JoinNoCommonKeyColumnsError,
        JoinResultTooLargeError,
        StatVarUnresolvedError,
        UnitNotPresentError,
        UnitHeterogeneousError,
        FacetHeterogeneousError,
    ) as exc:
        return to_tool_result(_refusal_for_engine_error(exc))

    identity = mint_identity_or_refusal(op_spec, input_content_versions=input_content_versions)
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    # `join` sets no derived flag of its own, for the same reason as
    # `compare_tool.compare`.
    detail: dict[str, object] = {
        "join_keys": list(result.join_keys),
        "left_dropped_key_count": result.left_dropped_key_count,
        "right_dropped_key_count": result.right_dropped_key_count,
        "right_resolved_unit": unit_json(result.right_resolved_unit),
        "right_resolved_facets": {
            selection.axis: selection.ref for selection in result.right_facet_selections
        },
    }
    derived = derived_marker_detail(
        inherited=merge_derived_markers(
            read_derived_marker(left_trace), read_derived_marker(right_trace)
        )
    )
    if derived is not None:
        detail["derived"] = derived

    # `join` sets no interval of its own: a matched key's row copies an
    # already-finite input cell verbatim.
    time_fact = merged_time(own_interval=None, inherited=[left, right])

    # The left side's own upstream_trace is carried forward -- the same
    # "the first-named side is the reference" choice `compare_tool.compare`
    # already makes. Both sides' own frames are bound into `content_version`
    # via `input_content_versions` above, and carried onto the trace's own
    # `input_content_versions` field verbatim.
    output_trace = Trace(
        upstream_trace=left_trace.upstream_trace,
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="join_inner",
            realised_coverage_fraction=(
                len(result.matched_keys)
                / (
                    len(result.matched_keys)
                    + result.left_dropped_key_count
                    + result.right_dropped_key_count
                )
                if (
                    len(result.matched_keys)
                    + result.left_dropped_key_count
                    + result.right_dropped_key_count
                )
                else 1.0
            ),
            skipped_row_count=result.left_dropped_key_count + result.right_dropped_key_count,
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
        step=_join_step(
            op_spec,
            result,
            trace=output_trace,
            ratio=(holds_ratio(left_trace), holds_ratio(right_trace)),
        ),
        digest=lambda: render_join_digest(
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
        dropped_key_count=result.left_unmatched_entity_count,
        right_dropped_key_count=result.right_unmatched_entity_count,
    )
