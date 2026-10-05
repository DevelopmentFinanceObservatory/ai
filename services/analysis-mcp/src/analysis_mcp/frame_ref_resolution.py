"""Shared `frame_ref` resolution, frame `content_version` derivation and
lookup, the `op_spec_hash` identity guard, array-parameter canonicalisation,
and the derived-ness marker reader, used by every verb tool that resolves a
stored frame before computing and mints identity after. Every verb tool
calls this module's functions, and `analysis_mcp.tool_pipeline`'s, directly.

`canonical_columns_or_refusal`/`canonical_predicates_or_refusal` close the
gap between what `analysis_mcp.identity` hashes and what the engine
executes over: `identity._normalise_parameters` sorts and dedupes every
array-shaped op-spec field (`group_by`, `entity_by`, `rank_by`, `join_keys`,
`predicates`) before minting `op_spec_hash`/`content_version`, so two
callers asking the identical question in a different array order mint the
identical identity -- but until a verb tool also canonicalises before
calling its own engine, that identical identity can back two calls that
actually computed (or displayed) different things. These two helpers reuse
`identity`'s own private normalisation functions directly so a verb tool's
execution order always matches the order its identity was minted from.

`canonical_entity_binding` closes the identical gap for `compare`'s
`left_entity`/`right_entity` and `share_of`'s `numerator_entity`/
`denominator_entity`. JCS sorts object members before minting
`op_spec_hash` for these fields, yet the execution-side order is otherwise
unsorted, so `validate_entity_columns` (`analysis_mcp.engine.common`),
reading the binding's raw iteration order, could render two
differently-ordered `role_ambiguous` refusals for two calls sharing one
identity.

`canonical_columns_and_resolved_frame` combines `canonical_columns_or_refusal`
with either one-input resolve function into one call, so
`aggregate_tool.aggregate`, `growth_tool.growth` and `rank_tool.rank` each need
only one early-return check to canonicalise their column-list parameter and
resolve their input frame.

`read_derived_marker` reads `applied_policy.detail["derived"]`, the marker a
producer verb writes onto its output's trace, and `DerivedMarker` names its
flags.

Every admitted frame's `time_period` column holds either points at one grain
or exactly one span (`_period_refusal`). `AdmittedFrame.span` names that span,
and `AdmittedFrame.time` the input's time kind. A
result an engine before 0.3.0 computed is refused `derived_input_unrecoverable`
(`_predates_span_labels`): those engines labelled a collapsed or grown row with
one borrowed point, and its trace cannot say which rows did.

`holds_input_rows` wraps every verb in one claim on the process's frame row
budget (`common.row_budget`). `resolve_input_frames` opens every input the
verb reads (`MaterialisationStore.open`), adds all their rows to that claim in
one step, and only then reads them, so the budget counts a two-input verb's
both sides from their download and decode through its engine call, store write
and digest, and a call never holds one side's rows while it waits for the
other's. The
claim ends when the verb returns or raises and its result's store write
(`tool_pipeline.store_result_and_render`) has finished.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import rfc8785
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    HeldFrame,
    get_store,
    register_content_version_deriver,
)
from common.row_budget import RowBudgetExhausted, RowClaim, row_budget
from common.time_grain import TimeGrain, distinct_time_grains, is_interval, time_grain
from dfo_frame_grammar.content_version import frame_sha256, mint_content_version
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN, parse_time_period
from fastmcp.tools import ToolResult

from analysis_mcp.artifact import Interval, Observation, Time
from analysis_mcp.identity import (
    OpSpecIdentityError,
    _sorted_unique_predicates,
    _sorted_unique_strings,
    op_spec_hash_for,
)
from analysis_mcp.op_spec import Predicate, PredicateAdapter
from analysis_mcp.settings import Settings
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.trace import Trace

if TYPE_CHECKING:
    from common.materialisation import StoredFrameTrace
    from dfo_frame_grammar import Frame

    from analysis_mcp.identity import OpSpecVariant


# The `basis` vocabulary `dfo_frame_grammar.content_version.mint_content_version`'s
# own preimage builder accepts (`_FACET_BASES` in that module) -- anything
# outside this set is normalised to `"declared_default"` by `_sanitised_facets`
# below.
_MINTABLE_FACET_BASES = frozenset({"bound", "declared_default"})


def _sanitised_facets(facets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project each of `facets` down to `mint_content_version`'s own closed
    `{axis, ref, basis}` shape, preserving entry order.

    A live `upstream_trace["facets"]` entry carries bookkeeping members
    (`default_policy`, `default_policy_version`, `dispatch_facet_id`,
    `selection_reason`, observed on real dfo-apis manifests) beyond that
    closed shape, and its own `basis` vocabulary is wider than the two
    literals (`bound`, `declared_default`) the preimage builder accepts --
    `"deployment_fallback"` is one observed value. Dropping the extra
    members and mapping any out-of-vocabulary `basis` to
    `"declared_default"` reproduces upstream's own minted `content_version`
    bit-for-bit on every live manifest checked so far.

    A live entry's `ref` can be `None` -- an axis with no facet selected.
    Upstream's own mint omits such an entry from its preimage and records
    it only in the trace, and the preimage builder refuses a `None` ref,
    so the entry is dropped here before minting.
    """
    return [
        {
            "axis": facet["axis"],
            "ref": facet["ref"],
            "basis": (
                facet["basis"] if facet["basis"] in _MINTABLE_FACET_BASES else "declared_default"
            ),
        }
        for facet in facets
        if facet["ref"] is not None
    ]


def derive_frame_content_version(*, frame: Frame, upstream_trace: dict[str, Any]) -> str:
    """`frame`'s frame-grammar `cv_` content version, minted fresh from
    `upstream_trace`'s lineage fields plus `frame`'s bytes.

    A verb tool calls this once, when its output is written
    (`Trace.frame_content_version`). A verb resolving a stored input reads
    that input's `frame_content_version` field directly
    (`input_content_version`, below).

    A `frame_ref` pair names either an original materialisation (a
    `spec_`/`cv_` pair, `get_data`'s own fixtures included) or a prior toolkit result (a chained verb call, an
    `ops_`/`ocv_` pair); this function returns the frame-grammar `cv_`
    family in both cases -- a chained result's own `Trace.content_version`
    carries the toolkit's `ocv_` family instead, which
    `mint_toolkit_content_version`'s `input_content_versions` parameter
    requires the `cv_` family for.

    `upstream_trace` is carried forward verbatim through every toolkit hop,
    so its `graph_revision`/`source_revisions`/`facets` are always the
    original materialisation's lineage, however many toolkit ops sit
    between it and this call. `facets` is sanitised (`_sanitised_facets`)
    before the derivation, since a live trace's `facets` entries cannot be
    passed through unmodified.
    """
    conversion = upstream_trace.get("conversion")
    kwargs: dict[str, Any] = {}
    if conversion is not None:
        kwargs["conversion"] = conversion
    return mint_content_version(
        frame_sha256=frame_sha256(frame),
        graph_revision=upstream_trace["graph_revision"],
        source_revisions=upstream_trace.get("source_revisions", []),
        facets=_sanitised_facets(upstream_trace.get("facets", [])),
        **kwargs,
    )


def _derive_content_version_for_store(frame: Frame, upstream_trace: dict[str, Any]) -> str:
    """Adapts `derive_frame_content_version`'s keyword-only signature to the
    positional `(frame, upstream_trace)` shape
    `common.materialisation.content_version`'s registry calls a deriver
    with."""
    return derive_frame_content_version(frame=frame, upstream_trace=upstream_trace)


register_content_version_deriver(_derive_content_version_for_store)


def input_content_version(trace: StoredFrameTrace) -> str:
    """A resolved input's own frame-grammar `cv_` content version, read
    straight off the resolved `StoredFrameTrace` as
    `trace.frame_content_version`.

    For a root frame (a `MaterialisationTrace`), this is the manifest token
    the caller's `frame_ref` already named. For a prior toolkit result (a
    `Trace`), it is the value `derive_frame_content_version` computed once
    when that result was written. A stored trace always carries the field.
    """
    return trace.frame_content_version


async def _open_or_refusal(frame_ref: FrameRef) -> HeldFrame | ToolResult:
    """`store.open(frame_ref)`, translating every store-level failure into its own
    refusal (`_store_refusal`), and a plain `None` (never registered, or evicted
    under this store's own cap) into `frame_not_held`.

    A durable store's own `open` can raise `FrameTooLarge` or `FrameIntegrityError`
    before `resolve_input_frames`'s own `settings.max_frame_rows` check runs, and
    its `HeldFrame.read` after it: `FrameTooLarge` exists to bound what the store
    decodes at all (its own size, row, column, and uncompressed-size ceilings, and
    the rows the read claimed), and
    `FrameIntegrityError` exists to catch a stored object that isn't the one it
    claims to be (a digest, row-count or envelope-identity mismatch). The row-count
    check in `resolve_input_frames` is a second, service-level ceiling, not a
    restatement of the store's own.
    """
    try:
        held = await get_store().open(frame_ref)
    except (FrameExpired, FrameIntegrityError, FrameTooLarge) as exc:
        return _store_refusal(frame_ref, exc)
    if held is None:
        return to_tool_result(
            ToolRefusal.frame_not_held(
                spec_id=frame_ref.spec_id, content_version=frame_ref.content_version
            )
        )
    return held


def _store_refusal(
    frame_ref: FrameRef, exc: FrameExpired | FrameIntegrityError | FrameTooLarge
) -> ToolResult:
    """The refusal for a store failure on `open` or `HeldFrame.read`: `FrameExpired`
    -> `frame_expired`, `FrameIntegrityError` -> `frame_integrity_failed`, and
    `FrameTooLarge` -> a pre-decode `frame_too_large`."""
    if isinstance(exc, FrameExpired):
        return to_tool_result(
            ToolRefusal.frame_expired(
                spec_id=frame_ref.spec_id, content_version=frame_ref.content_version
            )
        )
    if isinstance(exc, FrameIntegrityError):
        return to_tool_result(
            ToolRefusal.frame_integrity_failed(
                spec_id=frame_ref.spec_id,
                content_version=frame_ref.content_version,
                reason=exc.reason,
            )
        )
    return to_tool_result(ToolRefusal.frame_too_large_before_decode(reason=exc.reason))


# The row claim of the verb call running in this context (`holds_input_rows`).
_CALL_ROWS: ContextVar[RowClaim | None] = ContextVar("analysis_mcp_call_rows", default=None)


def holds_input_rows[**P](
    fn: Callable[P, Coroutine[Any, Any, ToolResult]],
) -> Callable[P, Coroutine[Any, Any, ToolResult]]:
    """Run verb `fn` under one claim on the process's frame row budget, opened
    empty and released when `fn` returns or raises, or later when work it ran
    through `RowClaim.hold_through` is still running. `resolve_input_frames`
    adds the inputs' rows to it. `functools.wraps` keeps the signature
    fastmcp builds the tool's schema from."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
        with row_budget().claim(0) as claim:
            token = _CALL_ROWS.set(claim)
            try:
                return await fn(*args, **kwargs)
            finally:
                _CALL_ROWS.reset(token)

    return wrapper


def call_row_claim() -> RowClaim:
    """The running verb call's row claim (`holds_input_rows`). Outside a
    `holds_input_rows` verb this raises `RuntimeError`, since nothing would
    release the rows."""
    claim = _CALL_ROWS.get()
    if claim is None:
        raise RuntimeError("frame rows claimed outside a holds_input_rows verb call")
    return claim


class AdmittedFrame(NamedTuple):
    """One input `resolve_input_frames` admitted: its frame, its stored trace,
    and `span`, the one span its `time_period` cells carry (`"2018/2022"`),
    or `None` when they are points."""

    frame: Frame
    trace: StoredFrameTrace
    span: str | None

    @property
    def time(self) -> Time:
        """The input's time kind. A toolkit result records its own
        (`Trace.time`). A raw materialisation records none, so its cells
        decide: one span is an `Interval` with unknown contiguity, since the
        trace cannot say which periods the source collapsed into it, and
        points are an `Observation`."""
        if isinstance(self.trace, Trace):
            return self.trace.time
        if self.span is None:
            return Observation()
        cell = parse_time_period(self.span)
        return Interval(start=cell.start, end=cell.end, contiguous=None)


async def resolve_input_frames(
    frame_refs: Sequence[FrameRef], *, settings: Settings
) -> list[AdmittedFrame] | ToolResult:
    """Resolve each of `frame_refs` to an `AdmittedFrame`, in order,
    or the first refusal.

    Every verb tool's own `frame_ref` resolution flow, in three passes. The
    first opens each input (`_open_or_refusal`) without reading it, checks the
    row count the store recorded against `settings.max_frame_rows`, and settles
    the rows it claims: that count, or `settings.max_frame_rows` for an object
    stored without one. The second adds every input's rows to the running
    verb's row claim (`holds_input_rows`) in one step, or refuses
    `service_unavailable` when the budget has no room for them. The third reads
    each input under that claim (`HeldFrame.read`), refused `frame_too_large`
    before decoding when it holds more rows than it claimed, then refuses
    `derived_input_unrecoverable` for a result an engine before 0.3.0
    computed (`_predates_span_labels`), and `mixed_time_grain` unless its `time_period`
    column holds points at one grain or exactly one span (`_period_refusal`).
    `Frame.build` and data-mcp's own materialisation each validate a
    `time_period` cell independently, so a stored frame can carry more than one grain (a string
    column passed through verbatim from an upstream artifact mixing an annual
    and a monthly series, say); every verb here compares periods as
    same-grain strings, so this refuses admission before any such comparison
    runs, rather than letting `filter`/`join` re-register the mixed column
    under a fresh `ocv_`.

    Outside a `holds_input_rows` verb this raises `RuntimeError`.
    """
    claim = call_row_claim()
    opened: list[tuple[FrameRef, HeldFrame, int]] = []
    for frame_ref in frame_refs:
        admitted = await _open_admissible(frame_ref, settings=settings)
        if isinstance(admitted, ToolResult):
            return admitted
        opened.append((frame_ref, *admitted))

    try:
        claim.extend(sum(rows for _, _, rows in opened))
    except RowBudgetExhausted as exc:
        return to_tool_result(
            ToolRefusal.row_budget_full(
                requested=exc.requested, free=exc.free, max_rows=exc.max_rows
            )
        )

    resolved: list[AdmittedFrame] = []
    for frame_ref, held, rows in opened:
        read = await _read_admissible(frame_ref, held, rows=rows)
        if isinstance(read, ToolResult):
            return read
        resolved.append(read)
    return resolved


async def _open_admissible(
    frame_ref: FrameRef, *, settings: Settings
) -> tuple[HeldFrame, int] | ToolResult:
    """`resolve_input_frames`'s first pass for one input: the opened input and the
    rows it claims."""
    held = await _open_or_refusal(frame_ref)
    if isinstance(held, ToolResult):
        return held
    if held.num_rows is None:
        return held, settings.max_frame_rows
    if held.num_rows > settings.max_frame_rows:
        return to_tool_result(
            ToolRefusal.frame_too_large(row_count=held.num_rows, ceiling=settings.max_frame_rows)
        )
    return held, held.num_rows


async def _read_admissible(
    frame_ref: FrameRef, held: HeldFrame, *, rows: int
) -> AdmittedFrame | ToolResult:
    """`resolve_input_frames`'s third pass for one input, which claimed `rows`."""
    try:
        frame, trace = await call_row_claim().hold_through(held.read(max_rows=rows))
    except (FrameExpired, FrameIntegrityError, FrameTooLarge) as exc:
        return _store_refusal(frame_ref, exc)
    if isinstance(trace, Trace) and _predates_span_labels(trace):
        return to_tool_result(
            ToolRefusal.derived_input_unrecoverable(
                engine_version=trace.execution_identity.engine_version
            )
        )
    periods = _distinct_periods(frame)
    if (refusal := _period_refusal(periods)) is not None:
        return to_tool_result(refusal)
    return AdmittedFrame(frame=frame, trace=trace, span=_one_span(periods))


# The first engine that labels a collapsed or grown row with the periods it covers.
_SPAN_LABEL_ENGINE = (0, 3, 0)


def _predates_span_labels(trace: Trace) -> bool:
    """Whether an engine before 0.3.0 computed `trace`'s result. Those engines
    labelled a collapsed row with one borrowed point (`2022` over a 2018-2022
    sum) and a growth row with its end year (`2023` over 2021-2023), and only
    the sum and rank cases carried a marker, so the engine version is the
    one reliable test."""
    engine = tuple(int(part) for part in trace.execution_identity.engine_version.split("."))
    return engine < _SPAN_LABEL_ENGINE


def _distinct_periods(frame: Frame) -> list[str]:
    """Every distinct `time_period` cell `frame` carries."""
    return cast("list[str]", frame.table.column(TIME_PERIOD_COLUMN).unique().to_pylist())


# How many spans `_period_refusal` names when a frame carries more than one.
_NAMED_SPANS = 2


def _period_refusal(periods: Sequence[str]) -> ToolRefusal | None:
    """`None` when `periods` (distinct `time_period` cells) are all points at
    one grain, or one span alone. Otherwise `mixed_time_grain`, naming one
    period per grain for a grain mix, or two spans and one point for a span
    beside other periods.

    Every verb reads a frame's periods as one set of points or one window, so
    a span beside a point, or two different spans, has no reading here.

    One pass keeps only the first period of each grain and the few periods a
    refusal names, so a frame of many distinct days holds no parsed cell per
    day.
    """
    first_by_grain: dict[TimeGrain, str] = {}
    spans: list[str] = []
    points: list[str] = []
    for period in sorted(periods):
        cell = parse_time_period(period)
        first_by_grain.setdefault(cell.grain, period)
        if cell.start != cell.end:
            if len(spans) < _NAMED_SPANS:
                spans.append(period)
        elif not points:
            points.append(period)
    grains = distinct_time_grains(first_by_grain.values())
    if len(grains) > 1:
        return ToolRefusal.mixed_time_grain(
            grains=list(grains), periods=[first_by_grain[grain] for grain in grains]
        )
    if not spans or len(periods) == 1:
        return None
    return ToolRefusal.mixed_time_grain(grains=list(grains), periods=[*spans, *points])


def _one_span(periods: Sequence[str]) -> str | None:
    """The one span admitted `periods` carry (`"2018/2022"`), or `None` when
    they are points. Admission leaves a span as a frame's only distinct
    period."""
    if len(periods) != 1:
        return None
    return periods[0] if is_interval(periods[0]) else None


async def resolve_input_frame(
    frame_ref: FrameRef, *, settings: Settings
) -> AdmittedFrame | ToolResult:
    """`resolve_input_frames` for one input."""
    resolved = await resolve_input_frames([frame_ref], settings=settings)
    return resolved if isinstance(resolved, ToolResult) else resolved[0]


def op_spec_hash_or_refusal(op_spec: OpSpecVariant) -> str | ToolResult:
    """`op_spec_hash_for(op_spec)`, with the two identity-minting failure
    modes caught and mapped to `op_spec_invalid` instead of propagating past
    the tool boundary as a raw, untyped error: `OpSpecIdentityError` (a
    duplicate entry in an array-shaped parameter such as `group_by` or
    `predicates`) and `rfc8785.CanonicalizationError` (a non-finite float
    that reached canonicalisation despite `op_spec.py`'s own field
    validators -- belt and suspenders, since a future array-shaped field
    could reintroduce the gap `OrderedThresholdPredicate.value`'s own field
    validator already closed). Every verb tool calls this immediately before
    minting identity, the same call site `aggregate_tool.aggregate`'s own
    inline copy of this guard occupies.
    """
    try:
        return op_spec_hash_for(op_spec)
    except (OpSpecIdentityError, rfc8785.CanonicalizationError) as exc:
        return to_tool_result(ToolRefusal.op_spec_invalid(detail=str(exc)))


def canonical_columns_or_refusal(
    columns: Sequence[str], *, description: str
) -> tuple[str, ...] | ToolResult:
    """`columns` -- a verb's own `group_by`/`entity_by`/`rank_by`/`join_keys`
    -- NFC-normalised, sorted, and deduplicated exactly as
    `identity._normalise_parameters` normalises the identical field before
    minting `op_spec_hash` (reusing that function's own private
    `_sorted_unique_strings` directly, not a reimplementation). Call this
    once, right after parsing the op-spec, and pass its result to the
    engine (and to the digest renderer) in place of the op-spec's own raw
    field -- otherwise two calls that mint the identical `op_spec_hash` from
    a differently-ordered `columns` can execute over, and display, a
    different column order (or, worse, a different sort priority for
    whichever truncation the digest applies).

    `description` is the field name a duplicate-entry refusal names, the
    same `description` `_sorted_unique_strings` itself takes.
    """
    try:
        return tuple(_sorted_unique_strings(columns, description=description))
    except OpSpecIdentityError as exc:
        return to_tool_result(ToolRefusal.op_spec_invalid(detail=str(exc)))


async def canonical_columns_and_resolved_frame(
    columns: Sequence[str],
    frame_ref: FrameRef,
    *,
    description: str,
    resolve: Callable[[FrameRef], Awaitable[AdmittedFrame | ToolResult]],
) -> tuple[tuple[str, ...], AdmittedFrame] | ToolResult:
    """`columns` canonicalised (`canonical_columns_or_refusal`), plus
    `frame_ref` resolved via `resolve` -- or the first refusal either step
    hits. Combined into one call so a verb's own body needs only
    one early-return check for both. `aggregate` and `rank` need that:
    splitting the two steps apart puts each of them at seven returns,
    one past the workspace's `PLR0911` gate. `growth` stays under the gate
    either way and uses this for consistency with the other two. A
    two-input verb calls `canonical_columns_or_refusal` and
    `resolve_input_frames` itself, so it claims both inputs' rows at once.

    `resolve` is `resolve_input_frame` bound to the caller's own `settings`,
    or for `rank_tool.rank` a wrapper around it that also refuses a rank
    position (`rank_tool._resolve_measured_frame`).
    """
    columns_result = canonical_columns_or_refusal(columns, description=description)
    if isinstance(columns_result, ToolResult):
        return columns_result
    resolved = await resolve(frame_ref)
    if isinstance(resolved, ToolResult):
        return resolved
    return columns_result, resolved


def canonical_predicates_or_refusal(
    predicates: Sequence[Predicate],
) -> tuple[Predicate, ...] | ToolResult:
    """`predicates` (`FilterOpSpec.predicates`), sorted and deduplicated
    exactly as `identity._normalise_parameters` normalises them before
    minting `op_spec_hash`. Reuses `identity._sorted_unique_predicates`
    directly (dict-shaped: each predicate is dumped to JSON, normalised,
    sorted by its own canonical JCS bytes, and checked for an exact
    duplicate), then rebuilds validated `Predicate` objects from the
    canonical dicts it returns, so `run_filter` executes over the identical
    order two op-specs sharing one `op_spec_hash` were minted from.
    """
    try:
        canonical = _sorted_unique_predicates(
            [predicate.model_dump(mode="json") for predicate in predicates]
        )
    except OpSpecIdentityError as exc:
        return to_tool_result(ToolRefusal.op_spec_invalid(detail=str(exc)))
    return tuple(PredicateAdapter.validate_python(entry) for entry in canonical)


def canonical_entity_binding(entity: Mapping[str, str]) -> dict[str, str]:
    """`entity` (a verb's own `{role_column: value}` binding -- `left_entity`/
    `right_entity`, `numerator_entity`/`denominator_entity`, or
    `ShareOfOpSpec.denominator_by`, the `{numerator column: denominator
    column}` name-keyed mapping), with its own keys sorted -- see this
    module's own docstring for why the array-field canonicalisation
    `canonical_columns_or_refusal` already applies isn't enough here. A
    plain `sorted()`, not `identity._sorted_unique_strings`: a `dict`'s own
    keys are already unique by construction, so there is no duplicate to
    refuse, and no NFC normalisation gap either -- two calls minting the
    identical `op_spec_hash` already agree on each key's exact bytes, byte
    order is all that differs between them. Call this once, right after
    parsing the op-spec, and pass its result to the engine in place of the
    op-spec's own raw binding, the same "canonicalise before executing"
    discipline `canonical_columns_or_refusal` already applies to every
    array-shaped parameter -- narrowing itself is order-independent
    (`narrow_and_resolve` filters one `{role_column: value}` pair at a time,
    and `denominator_by`'s own lookups are by key, not iteration order), so
    this changes no result, only which order a refusal exposes.
    """
    return dict(sorted(entity.items()))


@dataclass(frozen=True, kw_only=True)
class DerivedMarker:
    """What a producer verb recorded in `applied_policy.detail["derived"]`,
    naming which of a derived result's inherited fields (`unit`, `value`,
    `stat_var`) no longer mean what they meant on the input.

    `dimensionless`/`inherited_stat_var` -- `growth`'s and `share_of`'s
    output rows carry `unit=None` (a rate/ratio has none); `share_of`'s
    numerator's `stat_var` carries onto the output row, `growth`'s output
    inherits its input's `stat_var` unconditionally, and `compare` sets
    `inherited_stat_var` when its two sides' `stat_var`s differ. All are
    already-honest metadata: `unit=None` correctly says "no unit", and
    downstream unit resolution already refuses a genuine cross-unit
    combination without this marker's help. These two fields are carried
    for a human reading the trace; no verb refuses on them.

    `ordinal` -- a per-period `rank` with no `limit` carries the rank
    position as its output row's `value`, not the measured value the
    collapsing and limited modes emit (`engine/rank.py`). A ratio, a relative growth rate or a
    ranking over a position is not a meaningful number, so `share_of`, a
    relative `growth` and `rank` all refuse on this flag
    (`ToolRefusal.derived_input_ordinal`),
    independently of whatever the harness itself checks. Merged by OR like
    every other flag here -- a growth-over-a-rank output also carries
    `ordinal` onward, which over-claims (a difference of positions is a
    count of places, not itself an ordinal) but is correct in effect:
    nothing downstream of a rank position is a ratio-scale quantity either
    way, and a rank over a change in places refuses with the positions.
    """

    dimensionless: bool = False
    inherited_stat_var: bool = False
    ordinal: bool = False
    reason: str = ""


def read_derived_marker(trace: StoredFrameTrace) -> DerivedMarker | None:
    """`trace.applied_policy.detail["derived"]`, parsed into a `DerivedMarker`,
    or `None` if this input carries no such marker -- either because it
    wasn't itself a derived result, or because it's a bare `MaterialisationTrace`
    (a raw `get_data` frame nobody's verb has touched yet), which carries no
    `applied_policy` at all. Every verb tool reads its input's marker here to
    carry it onward via `derived_marker_detail`/`merge_derived_markers`.

    Keys outside `DerivedMarker`'s fields are ignored, so a stored trace
    carrying any other flag still resolves.
    """
    if not isinstance(trace, Trace):
        return None
    raw = trace.applied_policy.detail.get("derived")
    if raw is None:
        return None
    return DerivedMarker(
        dimensionless=bool(raw.get("dimensionless", False)),
        inherited_stat_var=bool(raw.get("inherited_stat_var", False)),
        ordinal=bool(raw.get("ordinal", False)),
        reason=str(raw.get("reason", "")),
    )


def derived_marker_detail(
    *,
    inherited: DerivedMarker | None,
    dimensionless: bool = False,
    inherited_stat_var: bool = False,
    ordinal: bool = False,
    reason: str = "",
) -> dict[str, object] | None:
    """The `applied_policy.detail["derived"]` payload for a producer verb
    that both reads (via `read_derived_marker`, or `merge_derived_markers`
    for a verb with more than one input) a marker its input(s) may already
    carry, and can set flags of its own: `rank` (`ordinal` when
    `per_period=True` with no `limit`), `growth`/`share_of` (`dimensionless`,
    both also `inherited_stat_var`, unconditional), `compare`
    (`inherited_stat_var`, only when its two sides' `stat_var`s differ).
    `aggregate`, `filter` and `join` call this with none of these, a pure
    passthrough of whatever `inherited` already carries.

    Merges rather than overwrites: every verb must carry its input's marker
    onward on its own output, merged with whatever flags it sets itself, or
    a single intervening hop silently launders a marker away; the laundering
    regression tests in `tests/test_derived_metadata.py` hold that. A growth
    reading a per-period rank carries `ordinal` onward this way, so a
    share_of two hops downstream still refuses. Returns `None` when neither
    the input(s) nor this call has anything to report.
    """
    merged_dimensionless = dimensionless or (inherited.dimensionless if inherited else False)
    merged_inherited_stat_var = inherited_stat_var or (
        inherited.inherited_stat_var if inherited else False
    )
    merged_ordinal = ordinal or (inherited.ordinal if inherited else False)
    if not (merged_dimensionless or merged_inherited_stat_var or merged_ordinal):
        return None

    detail: dict[str, object] = {}
    if merged_dimensionless:
        detail["dimensionless"] = True
    if merged_inherited_stat_var:
        detail["inherited_stat_var"] = True
    if merged_ordinal:
        detail["ordinal"] = True
    reasons = [r for r in (reason, inherited.reason if inherited else "") if r]
    if reasons:
        detail["reason"] = "; ".join(dict.fromkeys(reasons))
    return detail


def merge_derived_markers(*markers: DerivedMarker | None) -> DerivedMarker | None:
    """Combine markers read off more than one independently-resolved input
    (`compare`/`share_of`/`join`'s own two sides) into the single
    `inherited` a `derived_marker_detail` call can take -- OR-ing each flag
    and joining each side's own `reason`, the same "never drop a flag
    already set" discipline `derived_marker_detail` applies to one input,
    extended across more than one. `None` when every argument is `None`.
    """
    present = [marker for marker in markers if marker is not None]
    if not present:
        return None
    return DerivedMarker(
        dimensionless=any(marker.dimensionless for marker in present),
        inherited_stat_var=any(marker.inherited_stat_var for marker in present),
        ordinal=any(marker.ordinal for marker in present),
        reason="; ".join(dict.fromkeys(marker.reason for marker in present if marker.reason)),
    )


def _expected_period_count(*, start: str, end: str, grain: TimeGrain) -> int | None:
    """How many periods `start` to `end` inclusive spans at `grain`,
    computed arithmetically -- no period is ever materialised to count
    them, so this costs the same whether the span is a year or ten
    millennia.

    `None` at the day grain when `start` or `end` is shape-valid per
    `dfo_frame_grammar.grammar` (`YYYY-MM-DD`) but not a real calendar date
    -- the grammar deliberately admits a source's own out-of-range day
    (e.g. 30 February) or year 0, and this arithmetic has no ordinal to
    count from for a date that was never on the calendar.
    """
    if grain == "year":
        return int(end) - int(start) + 1
    if grain == "month":
        start_year, start_month = int(start[:4]), int(start[5:7])
        end_year, end_month = int(end[:4]), int(end[5:7])
        return (end_year - start_year) * 12 + (end_month - start_month) + 1
    try:
        return date.fromisoformat(end).toordinal() - date.fromisoformat(start).toordinal() + 1
    except ValueError:
        return None


def _iter_periods(*, start: str, end: str, grain: TimeGrain) -> Iterator[str]:
    """Every period from `start` to `end` inclusive, at `grain`, yielded
    one at a time so a caller can stop early rather than pay for the whole
    span. The day branch checks the loop bound *before* incrementing, so it
    never advances a `date` past `date.max` even when `end` is
    `"9999-12-31"` itself.
    """
    if grain == "year":
        year, end_year = int(start), int(end)
        while year <= end_year:
            yield str(year)
            year += 1
        return
    if grain == "month":
        year, month = int(start[:4]), int(start[5:7])
        end_year, end_month = int(end[:4]), int(end[5:7])
        while (year, month) <= (end_year, end_month):
            yield f"{year:04d}-{month:02d}"
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while True:
        yield current.isoformat()
        if current >= last:
            return
        current += timedelta(days=1)


# Bound on how many missing periods `Interval.absent` names, matching the
# fixed-cost caps `digest._truncation_note` and `analysis_mcp.artifact.
# ENTITY_CAP` apply elsewhere in this toolkit. `Interval.absent_count`
# carries the true total whenever this cap truncates the sample.
_ABSENT_CAP = 10


def build_interval(*, start: str, end: str, observed: Sequence[str]) -> Interval:
    """The `Interval` from `start` to `end`, with `contiguous`/`absent`/
    `absent_count` derived from `observed`: every real period at least one
    collapsed row carried.

    The true absent count is `_expected_period_count(start, end) -
    len(set(observed))` -- arithmetic, not a materialised span, since
    `observed` is already known to lie within `[start, end]` (both are
    derived from `observed`'s own min/max by every caller). `absent`
    itself still needs up to `_ABSENT_CAP` actual labels for the digest, so
    when the span isn't contiguous, this walks `_iter_periods` from
    `start` and collects labels not in `observed`, stopping as soon as
    `_ABSENT_CAP` are found. `observed` is bounded by `settings.
    max_frame_rows`, so that walk is bounded too: within any run of
    `len(observed) + _ABSENT_CAP` consecutive periods, at least
    `_ABSENT_CAP` must be absent, so the walk never has to materialise a
    wide span just to find its first ten gaps.

    `contiguous` is `None` when `_expected_period_count` can't compute a
    total (a calendar-invalid `start` or `end` the frame grammar admitted
    from its source) -- contiguity is unknown rather than false, and this
    returns without ever calling `_iter_periods`, whose own day-grain walk
    would hit the identical `date.fromisoformat` failure.
    """
    grain = time_grain(start)
    total = _expected_period_count(start=start, end=end, grain=grain)
    if total is None:
        return Interval(start=start, end=end, contiguous=None)
    observed_set = set(observed)
    absent_count = total - len(observed_set)
    if absent_count <= 0:
        return Interval(start=start, end=end, contiguous=True)

    absent: list[str] = []
    for period in _iter_periods(start=start, end=end, grain=grain):
        if len(absent) >= _ABSENT_CAP:
            break
        if period not in observed_set:
            absent.append(period)
    return Interval(
        start=start,
        end=end,
        contiguous=False,
        absent=tuple(absent),
        absent_count=absent_count,
    )


def merged_time(*, own_interval: Interval | None, inherited: Sequence[AdmittedFrame]) -> Time:
    """This call's time kind: `own_interval` when the verb's own output
    newly collapses a span of real periods into one label, otherwise the
    first `inherited` input's `Interval` (`AdmittedFrame.time`), if any
    carries one, otherwise `Observation()`.
    """
    if own_interval is not None:
        return own_interval
    for admitted in inherited:
        time = admitted.time
        if isinstance(time, Interval):
            return time
    return Observation()
