"""The `growth` tool: op-spec in, deterministic engine execution over a
stored frame, a `{spec_id, content_version}` pair and a trace out.

Flow mirrors `aggregate_tool.aggregate` exactly: parse/validate the op-spec
-> resolve `frame_ref` via the shared `analysis_mcp.frame_ref_resolution`
helpers -> run the engine (`analysis_mcp.engine.growth.run_growth`),
translating each of its typed exceptions to its own refusal -> mint
identity, build the trace, write the store under the result's own pair ->
render the digest. Every typed failure this function itself anticipates is returned via
`to_tool_result`, never raised. An exception outside that set would escape
this function; `server.py`'s registration-site wrapper is the backstop that
turns it into an `internal_error` refusal instead of a raw traceback
reaching the wire; this backstop complements the typed handling below
rather than replacing it.

A growth result is a rate (`change="relative"`) or a difference
(`change="absolute"`) over `[window_start, window_end]`, and its output
row's own `time_period` is that window as a span, `2018/2022`
(`engine/growth.py`). Two growth results over different windows carry
different labels, so `compare` and `join` match only results over the same
window. An input that already carries a span refuses
(`derived_input_unusable`) before the engine runs, since growth reads one row
per period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.artifact import Interval
from analysis_mcp.digest import render_growth_digest
from analysis_mcp.engine.common import (
    FacetHeterogeneousError,
    StatVarUnresolvedError,
    UnitHeterogeneousError,
    UnitNotPresentError,
    UnknownEntityColumnError,
)
from analysis_mcp.engine.growth import (
    GrowthAmbiguousEntityError,
    GrowthGapYearError,
    GrowthNonFiniteResultError,
    run_growth,
)
from analysis_mcp.frame_ref_resolution import (
    canonical_columns_and_resolved_frame,
    derive_frame_content_version,
    derived_marker_detail,
    holds_input_rows,
    input_content_version,
    merged_time,
    read_derived_marker,
    resolve_input_frame,
)
from analysis_mcp.identity import canonicalize_op_spec
from analysis_mcp.op_spec import GrowthOpSpec, PinnedUnit
from analysis_mcp.settings import Settings
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
    from analysis_mcp.frame_ref_resolution import AdmittedFrame

# Every typed exception `run_growth` can raise maps to exactly one
# refusal by `_refusal_for_engine_error` below. Every code reused here
# already exists on `ToolRefusal`; only `growth_gap_year` is new.
_EngineError = (
    UnknownEntityColumnError
    | StatVarUnresolvedError
    | UnitNotPresentError
    | UnitHeterogeneousError
    | FacetHeterogeneousError
    | GrowthAmbiguousEntityError
    | GrowthGapYearError
    | GrowthNonFiniteResultError
)


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_growth`'s typed exceptions to its refusal.

    `UnknownEntityColumnError` and `GrowthAmbiguousEntityError` both reuse
    `role_ambiguous` -- the first because `entity_by` names a column the
    frame has no grain for, the same shape `aggregate`'s `group_by` refusal
    already covers; the second because an entity_by that fails to uniquely
    identify one row per period is, in the same sense, the wrong set of
    grouping columns for what the caller is asking, just discovered later
    (after windowing) than a genuinely unknown column name is.
    `StatVarUnresolvedError`, `UnitNotPresentError`, `UnitHeterogeneousError`,
    and `FacetHeterogeneousError` reuse `aggregate`'s existing refusal
    factories directly -- their shapes already fit a second verb with the
    identical resolution rule. `GrowthNonFiniteResultError` reuses
    `non_finite_result`, the same non-finite discipline extended here to a
    zero/sign-crossing growth base rather than an overflowing sum.
    """
    # An if/elif chain with one assignment and one return, rather than one
    # early return per branch, matching
    # `aggregate_tool._refusal_for_engine_error`'s own shape (which the
    # workspace's `PLR0911` complexity gate requires).
    if isinstance(exc, UnknownEntityColumnError):
        refusal = ToolRefusal.role_ambiguous(
            parameter="entity_by", columns=list(exc.columns), available=list(exc.available)
        )
    elif isinstance(exc, GrowthAmbiguousEntityError):
        refusal = ToolRefusal.role_ambiguous(
            parameter="entity_by", columns=list(exc.entity_by), available=list(exc.available)
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
    elif isinstance(exc, GrowthGapYearError):
        refusal = ToolRefusal.growth_gap_year(
            entity_key=exc.entity_key,
            entity_label=entity_label(tuple(exc.entity_key.items())),
            gap_years=list(exc.gap_years),
            longest_gap_free_window=list(exc.longest_gap_free_window),
        )
    else:
        refusal = ToolRefusal.non_finite_result(group_key=exc.entity_key, aggregation="growth_rate")
    return refusal


def _unreadable_input_refusal(
    admitted: AdmittedFrame, *, change: Literal["relative", "absolute"]
) -> ToolRefusal | None:
    """The refusal for an input growth cannot read, checked before the engine.

    Growth reads one row per period, so a span input has nothing to window.
    The engine would report it as one gap year per period it cannot find.
    A relative growth rate over a rank position is meaningless, since a rank
    moves by places. Only `change="relative"` refuses it, since
    `change="absolute"` is the retry the refusal names, and `run_growth`
    computes it correctly over an ordinal input (the arithmetic does not
    care what the numbers mean).
    """
    if admitted.span is not None:
        return ToolRefusal.derived_input_unusable(span=admitted.span)
    marker = read_derived_marker(admitted.trace)
    if change == "relative" and marker is not None and marker.ordinal:
        return ToolRefusal.derived_input_ordinal(
            reason=(
                "this frame_ref names a per-period rank's own output (or a result "
                "derived from one), whose value column holds a rank position"
            )
        )
    return None


@holds_input_rows
async def growth(
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
    window_start: Annotated[
        str, Field(pattern=r"^\d{4}$", description="The window's first calendar year, e.g. '2020'.")
    ],
    window_end: Annotated[
        str,
        Field(pattern=r"^\d{4}$", description="The window's last calendar year, e.g. '2023'."),
    ],
    entity_by: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Column names identifying each entity whose own growth rate is computed. "
                "Empty treats the whole frame as one entity's series."
            ),
        ),
    ] = (),
    unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the unit to compute over, when the input frame carries more than "
                "one: a unit ref, or a converted unit as its {currency, prices, base_year} "
                "object. Retry with one of a unit_heterogeneous refusal's fix.units values."
            ),
        ),
    ] = None,
    change: Annotated[
        Literal["relative", "absolute"],
        Field(
            default="relative",
            description=(
                "'relative' (the default) computes (value_end - value_start) / value_start; "
                "refused over a zero or sign-crossing base, since a percentage change is "
                "undefined there. 'absolute' computes value_end - value_start directly, "
                "keeping the input's own unit; defined over any base, including zero or a "
                "sign crossing."
            ),
        ),
    ] = "relative",
) -> ToolResult:
    """Compute each entity's growth across `[window_start, window_end]`.

    `change="relative"` (the default) computes
    `growth_rate = (value_end - value_start) / value_start`, the total
    percentage change across the window; a zero or sign-crossing base makes
    that undefined and refuses (`non_finite_result`) before any division is
    attempted. `change="absolute"` computes `value_end - value_start`
    directly, in the input's own unit; that arithmetic is defined over any
    base, so the zero/sign-crossing refusal does not apply. Either mode
    treats a gap year inside the window as a break, not skipped: the whole
    call refuses (`growth_gap_year`), naming the missing years and the
    longest gap-free sub-window an entity actually has.

    Each output row's `time_period` is the window as a span, such as
    `2020/2023`. A frame whose rows already carry a span, such as an
    `aggregate` or `rank` result over several years, refuses
    (`derived_input_unusable`), because growth reads one row per period.
    Retry over the frame that result was computed from.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        window_start: The window's first calendar year.
        window_end: The window's last calendar year, strictly after window_start.
        entity_by: Column names identifying each entity; empty is one entity.
        unit: Pins the unit, to answer a prior unit_heterogeneous refusal.
        change: 'relative' for a percentage rate, 'absolute' for a difference
            in the input's own unit.

    Returns:
        A digest: each entity's growth value and its endpoint values, and the
        result's {spec_id, content_version} pair and identity fields. A
        relative growth rate is dimensionless -- `unit` only selected which
        input rows `value_start`/`value_end` were read from, it does not
        denominate the ratio they produced -- so the output frame's own
        `unit` is `None`; an absolute change keeps the input's own unit
        instead. On refusal, a typed error with a machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: GrowthOpSpec(
            frame_ref=frame_ref,
            window_start=window_start,
            window_end=window_end,
            entity_by=tuple(entity_by),
            unit=unit,
            change=change,
        )
    )
    if isinstance(op_spec, ToolResult):
        return op_spec

    # `entity_by` is canonicalised before the engine runs -- see
    # `aggregate_tool.aggregate`'s identical comment on its own `group_by`.
    settings = Settings()
    prelude = await canonical_columns_and_resolved_frame(
        op_spec.entity_by,
        op_spec.frame_ref,
        description="entity_by",
        resolve=lambda ref: resolve_input_frame(ref, settings=settings),
    )
    if isinstance(prelude, ToolResult):
        return prelude
    entity_by, admitted = prelude
    frame, input_trace = admitted.frame, admitted.trace

    if refusal := _unreadable_input_refusal(admitted, change=op_spec.change):
        return to_tool_result(refusal)

    try:
        result = run_growth(
            frame,
            window_start=op_spec.window_start,
            window_end=op_spec.window_end,
            entity_by=entity_by,
            change=op_spec.change,
            pinned_unit=op_spec.unit,
        )
    except (
        UnknownEntityColumnError,
        StatVarUnresolvedError,
        UnitNotPresentError,
        UnitHeterogeneousError,
        FacetHeterogeneousError,
        GrowthAmbiguousEntityError,
        GrowthGapYearError,
        GrowthNonFiniteResultError,
    ) as exc:
        return to_tool_result(_refusal_for_engine_error(exc))

    input_cv = input_content_version(input_trace)
    identity = mint_identity_or_refusal(op_spec, input_content_versions=[input_cv])
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    # The derived-ness marker (see `frame_ref_resolution.DerivedMarker`): a
    # relative growth rate is dimensionless unconditionally -- the output
    # frame's own `unit` is already honestly `None` (`engine/growth.py`), not
    # a borrowed label -- so `change="relative"` sets `dimensionless=True`
    # regardless of the input. An absolute change keeps the input's own unit
    # (`resolved.unit`, possibly itself `None`), so `change="absolute"`
    # passes `dimensionless=False` here and lets `derived_marker_detail`'s
    # own OR-merge carry the flag forward only when the *input* already
    # carried it.
    # `engine/growth.py` does stamp the input's own `stat_var` verbatim onto
    # the output row in either mode, the same situation `share_of` flags with
    # `inherited_stat_var` -- both are informational only (see that
    # dataclass's own docstring), routed through `derived_marker_detail` so
    # any marker growth's own input already carried survives alongside them.
    derived = derived_marker_detail(
        inherited=read_derived_marker(input_trace),
        dimensionless=op_spec.change == "relative",
        inherited_stat_var=True,
    )
    # The input is a frame of points (a span input refused above) collapsed to
    # the window. A gap year inside the window refuses, so the window is
    # contiguous by construction.
    time_fact = merged_time(
        own_interval=Interval(start=op_spec.window_start, end=op_spec.window_end, contiguous=True),
        inherited=[admitted],
    )
    detail: dict[str, object] = {
        "window_start": result.window_start,
        "window_end": result.window_end,
        "per_entity_coverage": {
            entity_label(growth_.entity_key): growth_.realised_coverage_fraction
            for growth_ in result.entity_growths
        },
        "derived": derived,
    }

    output_trace = Trace(
        upstream_trace=input_trace.upstream_trace,
        frame_content_version=derive_frame_content_version(
            frame=result.frame, upstream_trace=input_trace.upstream_trace
        ),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="growth_refuse_gap_year",
            # The worst of the per-entity fractions already in `detail`
            # above, not a bare `1.0` literal -- see `engine/growth.py`'s
            # `EntityGrowth.realised_coverage_fraction` for why this stays
            # truthful even though every entity here today always carries
            # exactly 1.0. `default=1.0` only matters for the vacuous
            # empty-window case, itself a refusal (`GrowthGapYearError`)
            # before any entity exists to average over.
            realised_coverage_fraction=min(
                (growth_.realised_coverage_fraction for growth_ in result.entity_growths),
                default=1.0,
            ),
            skipped_row_count=0,
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
        digest=lambda: render_growth_digest(
            window_start=op_spec.window_start,
            window_end=op_spec.window_end,
            entity_by=entity_by,
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
    )
