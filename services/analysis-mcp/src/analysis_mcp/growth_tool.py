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
(`change="absolute"`) between each entity's values in `window_start` and
`window_end`, computed for every entity with a row in both; its output
row's own `time_period` is that window as a span, `2018/2022`
(`engine/growth.py`). Two growth results over different windows carry
different labels, so `compare` and `join` match only results over the same
window. An input that already carries a span refuses
(`derived_input_unusable`) before the engine runs, since growth reads one row
per period.

The stored step (`_growth_step`) counts the entities the growth keeps and
those it leaves out by reason, and names a relative growth's unit as a
dimensionless ratio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.composite import composite_refusal
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
    GrowthEndpointsMissingError,
    GrowthNonFiniteResultError,
    GrowthResult,
    run_growth,
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
    FrameRefArgument,
    GrowthOpSpec,
    PinnedUnit,
    WindowRolling,
    with_anchor_key,
)
from analysis_mcp.rolling_tool import anchored_input
from analysis_mcp.settings import Settings
from analysis_mcp.steps import (
    LeftOut,
    Step,
    build_step,
    caller_params,
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
    from analysis_mcp.frame_ref_resolution import AdmittedFrame

# Every typed exception `run_growth` can raise maps to exactly one
# refusal by `_refusal_for_engine_error` below. `growth_gap_year` is the
# one code only growth raises.
_EngineError = (
    UnknownEntityColumnError
    | StatVarUnresolvedError
    | UnitNotPresentError
    | UnitHeterogeneousError
    | FacetHeterogeneousError
    | GrowthAmbiguousEntityError
    | GrowthEndpointsMissingError
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
    identical resolution rule. `GrowthNonFiniteResultError` carries
    `non_finite_result`'s code through `growth_non_finite_result`, which
    counts the excluded entities by cause in place of one group key.
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
    elif isinstance(exc, GrowthEndpointsMissingError):
        refusal = ToolRefusal.growth_gap_year(
            window_start=exc.window_start,
            window_end=exc.window_end,
            entity_count=exc.entity_count,
            missing_start=exc.missing_start,
            missing_end=exc.missing_end,
            suggested_window=list(exc.suggested_window),
            suggested_window_entities=exc.suggested_window_entities,
        )
    else:
        refusal = ToolRefusal.growth_non_finite_result(
            entity_count=exc.entity_count,
            zero_or_sign_crossing_base=exc.zero_or_sign_crossing_base,
            non_finite=exc.non_finite,
        )
    return refusal


def _growth_step(op_spec: GrowthOpSpec, result: GrowthResult, *, trace: Trace) -> Step:
    """The step record of `result`, stored on `trace`: the entities it
    computed, the entities it left out by reason, and its unit, a
    dimensionless ratio for a relative change and the input's unit for an
    absolute one. The window's bounds are the params `start` and `end`."""
    return build_step(
        op="growth",
        trace=trace,
        inputs={"input": op_spec.frame_ref},
        params=window_params(caller_params(trace.execution_record.content)),
        kept={"entities": len(result.entity_growths)},
        left_out=[
            *(
                LeftOut(reason=reason, count=count, of="entities")
                for reason, count in result.entities_left_out.items()
            ),
            *left_out_from_exclusions(result.exclusions, row_reasons=("unit_unbound",)),
        ],
        unit=step_unit(
            result.resolved_unit if result.change == "absolute" else None,
            ratio=holds_ratio(trace),
        ),
    )


def _unreadable_input_refusal(
    admitted: AdmittedFrame, *, change: Literal["relative", "absolute"]
) -> ToolRefusal | None:
    """The refusal for an input growth cannot read, checked before the engine.

    Growth reads one row per period, so a span input has nothing to window.
    The engine would find no entity with a row in either endpoint year.
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
                "one: a unit ref, a converted unit as its {currency, prices, base_year} "
                "object, or a composed unit as its {numerator, denominator} object. "
                "Retry with one of a unit_heterogeneous refusal's fix.units values."
            ),
        ),
    ] = None,
    change: Annotated[
        Literal["relative", "absolute"],
        Field(
            default="relative",
            description=(
                "'relative' (the default) computes (value_end - value_start) / value_start; "
                "an entity with a zero or sign-crossing base is excluded, since a percentage "
                "change is undefined there. 'absolute' computes value_end - value_start directly, "
                "keeping the input's own unit; defined over any base, including zero or a "
                "sign crossing."
            ),
        ),
    ] = "relative",
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
    """Compute each entity's growth across `[window_start, window_end]`.

    Use it for the change between two periods, as a percentage or an amount.

    `change="relative"` (the default) computes
    `growth_rate = (value_end - value_start) / value_start`, the total
    percentage change across the window; a zero or sign-crossing base makes
    that undefined, so such an entity is excluded as
    `zero_or_sign_crossing_base` before any division is attempted.
    `change="absolute"` computes `value_end - value_start` directly, in the
    input's own unit; that arithmetic is defined over any base. Either mode
    excludes an entity whose value overflows as `non_finite`, and the call
    refuses (`non_finite_result`) only when every entity is excluded so.

    Both modes read only each entity's rows in `window_start` and
    `window_end`. Every entity with a row in both is computed, whatever
    years between them it lacks; a warning counts the entities with such
    gaps. An entity missing either endpoint year is left out and listed
    under exclusions as `endpoint_missing`. When no entity has both, the
    call refuses (`growth_gap_year`), counting the entities missing each
    endpoint and suggesting a window at least one entity holds both ends of,
    when one exists.

    Each output row's `time_period` is the window as a span, such as
    `2020/2023`. A frame whose rows already carry a span, such as an
    `aggregate` or `rank` result over several years, refuses
    (`derived_input_unusable`), because growth reads one row per period.
    Retry over the frame that result was computed from.

    `rolling` records that a window year follows the latest period of the
    data. The call computes over the literal years it is given, as a fixed
    call does, and the result's {spec_id, content_version} pair differs from
    the fixed call's. Such a call runs only when the record `anchor_key` names
    covers it (`rolling_tool.anchored_input`), and refuses
    `rolling_anchor_unresolved` otherwise.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        window_start: The window's first calendar year.
        window_end: The window's last calendar year, strictly after window_start.
        entity_by: Column names identifying each entity; empty is one entity.
        unit: Pins the unit, to answer a prior unit_heterogeneous refusal.
        change: 'relative' for a percentage rate, 'absolute' for a difference
            in the input's own unit.
        rolling: Which window years follow the latest period, and the
            populations that period is taken over.
        anchor_key: The handle of the stored latest period, set on a rolling call.

    Returns:
        A digest: each computed entity's growth value and its endpoint
        values, the entities excluded for a missing endpoint, and the
        result's {spec_id, content_version} pair and identity fields. A
        relative growth rate is dimensionless -- `unit` only selected which
        input rows `value_start`/`value_end` were read from, it does not
        denominate the ratio they produced -- so the output frame's own
        `unit` is `None`; an absolute change keeps the input's own unit
        instead. On refusal, a typed error with a machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: with_anchor_key(
            GrowthOpSpec(
                frame_ref=frame_ref,
                window_start=window_start,
                window_end=window_end,
                entity_by=tuple(entity_by),
                unit=unit,
                change=change,
                rolling=rolling,
            ),
            anchor_key,
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
        resolve=lambda ref: anchored_input(
            op_spec, anchor_key, resolve_input_frame(ref, settings=settings)
        ),
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
        GrowthEndpointsMissingError,
        GrowthNonFiniteResultError,
    ) as exc:
        return to_tool_result(composite_refusal(_refusal_for_engine_error(exc), [input_trace]))

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
    # the output row in either mode, which `inherited_stat_var` flags.
    # Both flags are informational only (see that dataclass's own docstring),
    # and are routed through `derived_marker_detail` so any marker growth's
    # own input already carried survives alongside them.
    derived = derived_marker_detail(
        inherited=read_derived_marker(input_trace),
        dimensionless=op_spec.change == "relative",
        inherited_stat_var=True,
    )
    # The input is a frame of points (a span input refused above) collapsed to
    # the window. Every computed entity holds both endpoints, so the observed
    # periods run from `window_start` to `window_end`. A year no computed
    # entity holds makes the window non-contiguous.
    time_fact = merged_time(
        own_interval=build_interval(
            start=op_spec.window_start,
            end=op_spec.window_end,
            observed=result.observed_periods,
        ),
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
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="growth_endpoint_years",
            # The worst of the per-entity fractions already in `detail`
            # above: below 1.0 when a computed entity lacks interior years.
            # `run_growth` refuses when no entity is computed, so `default`
            # is never reached.
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
        step=_growth_step(op_spec, result, trace=output_trace),
        digest=lambda: render_growth_digest(
            window_start=op_spec.window_start,
            window_end=op_spec.window_end,
            entity_by=entity_by,
            result=facts_of(result),
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
            derived=derived,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
    )
