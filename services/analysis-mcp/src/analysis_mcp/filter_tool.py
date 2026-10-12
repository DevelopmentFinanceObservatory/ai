"""The `filter` tool: op-spec in, deterministic engine execution over one
stored frame, a `{spec_id, content_version}` pair and a trace out.

Flow mirrors `aggregate_tool.aggregate`: parse/validate the op-spec ->
resolve `frame_ref` via the shared `analysis_mcp.frame_ref_resolution`
helpers -> run the engine (`analysis_mcp.engine.filter.run_filter`),
translating each of its typed exceptions to its own refusal -> mint
identity, build the trace, write the store under the result's own pair ->
render the digest. Every typed failure this function itself anticipates is returned
via `to_tool_result`. An exception outside that set would escape this
function. `server.py`'s registration-site wrapper is the backstop that
turns it into an `internal_error` refusal instead of a raw traceback
reaching the wire, and this backstop complements the typed handling below
rather than replacing it.

Registered under the wire name `filter` (see `server.py`) even though the
Python function here is named `filter_verb`. Shadowing the `filter`
builtin at module scope would work, but naming the function distinctly
keeps this module's own code (and any future `from analysis_mcp.filter_tool
import filter_verb`) unambiguous, while `mcp.tool(filter_verb, name="filter")`
still exposes it under the verb's own closed-vocabulary name.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.digest import render_filter_digest
from analysis_mcp.engine.filter import (
    FilterPredicateTypeMismatchError,
    FilterPredicateValueTypeMismatchError,
    FilterUnitConversionUnsupportedError,
    FilterUnknownColumnError,
    run_filter,
)
from analysis_mcp.frame_ref_resolution import (
    canonical_predicates_or_refusal,
    derive_frame_content_version,
    derived_marker_detail,
    holds_input_rows,
    input_content_version,
    merged_time,
    read_derived_marker,
    resolve_input_frame,
)
from analysis_mcp.identity import canonicalize_op_spec
from analysis_mcp.op_spec import (
    ANCHOR_KEY_PATTERN,
    CategoryMatchPredicate,
    FilterOpSpec,
    FilterRolling,
    OrderedThresholdPredicate,
    PlaceLevelPredicate,
    with_anchor_key,
)
from analysis_mcp.rolling_tool import anchored_input
from analysis_mcp.settings import Settings
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import (
    build_op_spec_or_refusal,
    exclusions_for_trace,
    mint_identity_or_refusal,
    store_result_and_render,
)
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace

# Every typed exception `run_filter` can raise maps to exactly one refusal
# by `_refusal_for_engine_error` below.
_EngineError = (
    FilterUnknownColumnError
    | FilterPredicateTypeMismatchError
    | FilterPredicateValueTypeMismatchError
    | FilterUnitConversionUnsupportedError
)


def _refusal_for_engine_error(exc: _EngineError) -> ToolRefusal:
    """Map one of `run_filter`'s typed exceptions to its refusal.

    `FilterUnknownColumnError` reuses `role_ambiguous`. A predicate naming
    a column this frame doesn't declare is, in the same sense `aggregate`'s
    own refusal already covers, the wrong column list for what the caller
    asked. `FilterPredicateTypeMismatchError` and
    `FilterPredicateValueTypeMismatchError` both fire when a predicate
    doesn't fit its column and share one retryable code. `FilterPredicate
    TypeMismatchError` covers a wrong kind or column, `FilterPredicateValue
    TypeMismatchError` a wrong operand type for the right column.
    `FilterUnitConversionUnsupportedError` has its own code, since its retry
    drops the unit predicate or selects one unit another way.
    See each exception's own docstring for why they're distinct conditions.
    """
    # An if/elif chain with one assignment and one return, rather than one
    # early return per branch, matching `aggregate_tool._refusal_for_engine_error`'s
    # own shape (which the workspace's `PLR0911` complexity gate requires).
    if isinstance(exc, FilterUnknownColumnError):
        refusal = ToolRefusal.role_ambiguous(
            parameter="predicates", columns=list(exc.columns), available=list(exc.available)
        )
    elif isinstance(exc, FilterPredicateTypeMismatchError):
        refusal = ToolRefusal.filter_predicate_type_mismatch(
            column=exc.column, predicate_kind=exc.predicate_kind, actual_type=exc.actual_type
        )
    elif isinstance(exc, FilterPredicateValueTypeMismatchError):
        refusal = ToolRefusal.filter_predicate_value_type_mismatch(
            column=exc.column, actual_type=exc.actual_type, expected_type=exc.expected_type
        )
    else:
        refusal = ToolRefusal.filter_unit_conversion_unsupported(
            column=exc.column, predicate_kind=exc.predicate_kind, actual_type=exc.actual_type
        )
    return refusal


@holds_input_rows
async def filter_verb(
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
    predicates: Annotated[
        tuple[OrderedThresholdPredicate | CategoryMatchPredicate | PlaceLevelPredicate, ...],
        Field(
            default=(),
            description=(
                "Row-value predicates, applied as a conjunction (a row survives only if it "
                "satisfies every predicate). The closed vocabulary: an ordered_threshold "
                "(column/op/value, value a number against value or a period string against "
                "time_period; refused on time_period when that holds a span such as "
                "2018/2022), a category_match (column/values), or a place_level "
                "(column/level), which keeps a row whose place-type column, such as "
                "observationAbout_type, names a place at that level, such as Country, to keep "
                "one level of place. A role, a window, or a facet binding is out of this "
                "grammar -- filter narrows by value only, never by scope; scope narrowing "
                "belongs at the Datasets door. An ordered_threshold on time_period may carry "
                "rolling, {anchor: 'latest', offset: 0, rule: 'half1'}, marking its value as "
                "following the latest period of the data, with offset counting whole periods "
                "back; the value stays the literal period compared against."
            ),
        ),
    ] = (),
    rolling: Annotated[
        FilterRolling | None,
        Field(
            default=None,
            description=(
                "The populations the latest period is taken over, {population: [{spec_id, "
                "bindings}]}: one entry for each root dataset of the question, bindings "
                "narrowing its series as {column: [values]}. Required when a predicate "
                "carries rolling and refused otherwise."
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
    """Keep only the rows of `frame_ref` satisfying every predicate in `predicates`.

    Row-value predicates only. An `ordered_threshold` compares a column's
    own ordered cell against a threshold, numerically against `value`, or
    lexically against `time_period`'s own period string at any grain
    (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`). A `category_match` tests a column's
    own string cell against a closed set of values. This includes the
    `unit` column itself. A `place_level` reads a place's types, such as
    the `observationAbout_type` column `get_data(place_types=true)` adds,
    by `common.place_type.levels_of`: a frame that names no place holds
    every level of place, and `place_level` on it with `"Country"` keeps
    the countries. A `category_match` on `unit` is an ordinary way
    to narrow a heterogeneous-unit frame down to one unit, since filter
    inspects each row in isolation and never combines two rows' own values
    (unlike `aggregate`/`compare`/`growth`/`rank`/`share_of`, filter resolves
    no unit or facet before running). An empty result is valid, and the
    frame grammar accepts a columns-only empty frame. `frame_ref` may
    itself name a prior `aggregate`, `rank` or `growth` result whose
    `time_period` is one span such as `2018/2022`. An `ordered_threshold` on
    `time_period` refuses over it (`filter_predicate_type_mismatch`, `actual_type` `interval`),
    since the span's string says nothing about the periods it covers. A
    `category_match` on the exact span string keeps its rows, and a predicate
    on any other column filters as usual.

    A threshold on `time_period` that carries `rolling` records that its
    period follows the latest period of the data. The call compares against
    the literal `value`, as a fixed threshold does, and the result's
    {spec_id, content_version} pair differs from the fixed call's. Such a call
    runs only when the record `anchor_key` names covers it
    (`rolling_tool.anchored_input`), and refuses `rolling_anchor_unresolved`
    otherwise.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        predicates: Row-value predicates, applied as a conjunction. Empty keeps every row.
        rolling: The populations the latest period is taken over, beside a rolling predicate.
        anchor_key: The handle of the stored latest period, set on a rolling call.

    Returns:
        A digest: how many rows were kept versus dropped, a capped preview
        of the kept rows, and the result's {spec_id, content_version} pair
        and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    op_spec = build_op_spec_or_refusal(
        lambda: with_anchor_key(
            FilterOpSpec(frame_ref=frame_ref, predicates=tuple(predicates), rolling=rolling),
            anchor_key,
        )
    )
    if isinstance(op_spec, ToolResult):
        return op_spec

    # Canonicalised before the engine runs. See
    # `aggregate_tool.aggregate`'s identical comment on its own `group_by`.
    # A conjunction is order-independent for which rows survive, but the
    # duplicate-predicate refusal `identity._normalise_parameters` already
    # applies at hash-minting time is surfaced here too, before wasting an
    # engine run on a predicate set the identity step would refuse anyway.
    predicates_result = canonical_predicates_or_refusal(op_spec.predicates)
    if isinstance(predicates_result, ToolResult):
        return predicates_result
    canonical_predicates = predicates_result

    settings = Settings()
    resolved = await anchored_input(
        op_spec, anchor_key, resolve_input_frame(op_spec.frame_ref, settings=settings)
    )
    if isinstance(resolved, ToolResult):
        return resolved
    frame, input_trace = resolved.frame, resolved.trace

    try:
        result = run_filter(frame, predicates=canonical_predicates, span=resolved.span)
    except (
        FilterUnknownColumnError,
        FilterPredicateTypeMismatchError,
        FilterPredicateValueTypeMismatchError,
        FilterUnitConversionUnsupportedError,
    ) as exc:
        return to_tool_result(_refusal_for_engine_error(exc))

    input_cv = input_content_version(input_trace)
    identity = mint_identity_or_refusal(op_spec, input_content_versions=[input_cv])
    if isinstance(identity, ToolResult):
        return identity
    op_spec_hash, execution_identity, content_version = identity

    # `filter` narrows rows without changing what a kept row's own
    # `time_period`/`unit`/`stat_var` mean, so a derived input's own marker
    # carries straight through onto this call's own output, unchanged, via
    # `derived_marker_detail`.
    #
    # `kept_row_count`/`dropped_row_count` live in `detail`, not in
    # `AppliedPolicy.realised_coverage_fraction`/`skipped_row_count`.
    # `trace.py`'s own `AppliedPolicy` docstring defines those two scalars as
    # *grain* coverage (cells present versus cells expected for a grouped
    # grid), the quantity `aggregate`/`rank` populate them with. `filter`'s
    # own kept/dropped counts are predicate survival, a different quantity
    # entirely that happens to share a "kept over total" shape. Reusing the
    # grain-coverage fields would conflate a `filter` result's "coverage"
    # with an `aggregate` result's, though both have the same numeric form.
    #
    # `filter` is the only verb that leaves those two fields `None`. Every
    # other verb populates both: `aggregate`/`rank` with a real
    # grain-coverage fraction, `compare`/`share_of` with a cross-side
    # coverage fraction, `growth` with a coverage fraction computed from its
    # own present/expected year counts (below 1.0 when a computed entity
    # lacks interior years), and `join` with
    # `matched / (matched + dropped)` (the identical kept-over-total shape).
    detail: dict[str, object] = {
        "predicate_count": len(canonical_predicates),
        "kept_row_count": result.kept_row_count,
        "dropped_row_count": result.dropped_row_count,
        "total_row_count": result.total_row_count,
    }
    derived = derived_marker_detail(inherited=read_derived_marker(input_trace))
    if derived is not None:
        detail["derived"] = derived

    # `filter` never collapses a span of its own: its predicates test each
    # row's value in isolation and don't change what a kept row's
    # `time_period` means, so a prior `aggregate`/`rank` result's interval
    # carries straight through to this call's output, unchanged.
    time_fact = merged_time(own_interval=None, inherited=[resolved])

    output_trace = Trace(
        upstream_trace=input_trace.upstream_trace,
        frame_content_version=derive_frame_content_version(frame=result.frame),
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(
            policy_name="filter_row_value_predicates",
            realised_coverage_fraction=None,
            skipped_row_count=None,
            resolved_unit=None,
            resolved_facets={},
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
        input_traces=[input_trace],
        digest=lambda: render_filter_digest(
            predicate_count=len(canonical_predicates),
            result=result,
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            derived=derived,
            execution_identity=execution_identity,
            exclusions=output_trace.exclusions,
            time=output_trace.time,
        ),
    )
