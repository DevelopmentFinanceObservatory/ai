"""The `growth` engine: growth-rate computation over a frame's table,
across each entity's calendar window, read from the window's two endpoint
years.

Growth is single-op arithmetic -- one subtraction, one division per entity
-- not a reduction over many rows, so `math.fsum` is never used here.
Single-op arithmetic (growth, compare, filter predicates) is IEEE-exact
as-is: IEEE-754 arithmetic on two already-finite floats is exact for that
purpose.

Declared policy: `change="relative"` (the default) computes
`growth_rate = (value_end - value_start) / value_start`, the total
percentage change across the window -- a plainer, directly
sanity-checkable number, not a compound annual rate. `change="absolute"`
computes `value_end - value_start` directly, the change in the input's own
unit. Both read only the entity's rows in `window_start` and `window_end`.
An entity with a row in each is computed, whatever years between them it
lacks; those interior gaps lower its `realised_coverage_fraction` and add a
warning counting the entities that have them. An entity missing either
endpoint is left out of the output and its in-window rows are recorded as an
`endpoint_missing` exclusion. When no entity has both endpoints, the call
refuses (`GrowthEndpointsMissingError`), counting how many entities miss
each endpoint and suggesting a window at least one entity holds both ends
of.

Under `change="relative"` only, a zero or sign-crossing base (`value_start`
or `value_end` zero, or the two differing in sign) makes a percentage change
undefined, so that entity is excluded as `zero_or_sign_crossing_base` before
any division is attempted, checked explicitly and not caught after a
`ZeroDivisionError` or a `**` on a negative base silently promoting to a
complex number. An absolute change over the same base is defined
(subtraction, unlike division, does not break at zero or across a sign
crossing), so that exclusion does not apply there. Either mode excludes an
entity whose computed value overflows to a non-finite float as
`non_finite`. The call refuses (`GrowthNonFiniteResultError`) only when
every entity holding both endpoints is excluded this way.

Each output row's `time_period` is the window as a span,
`span_label([window_start, window_end])` (`2018/2022`), so two growth results
over different windows carry different labels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal, cast

import pyarrow as pa
from common.time_grain import span_label
from dfo_frame_grammar import Frame, array_kernel
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitRequest,
)

from analysis_mcp.engine.columnar import (
    Groups,
    distinct,
    frame_from_columns,
    group_rows,
    in_periods,
    keys_unique,
    repeated,
    unit_column,
)
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    available_entity_columns,
    facet_columns,
    narrow_and_resolve,
    record_exclusion,
    require_single_stat_var,
    validate_entity_columns,
)


class GrowthEndpointsMissingError(Exception):
    """No entity has a row in both `window_start` and `window_end`, so no
    entity has the two values growth reads.

    The counts cover every entity the narrowed input holds, including one
    with no row inside the window: `missing_start` and `missing_end` count
    the entities lacking each endpoint, so an entity lacking both counts in
    each. `suggested_window` is a window inside the requested one that
    `suggested_window_entities` entities (at least one) hold both endpoints
    of. It spans the earliest and latest year every entity with an in-window
    row holds or, when those entities share fewer than two years, the widest
    span one entity holds (the first in sorted entity order on a tie). It is
    `()` with a count of 0 when no entity holds two years inside the window.
    These are what the `growth_gap_year` refusal's fix names.
    """

    def __init__(
        self,
        *,
        window_start: str,
        window_end: str,
        entity_count: int,
        missing_start: int,
        missing_end: int,
        suggested_window: tuple[str, str] | tuple[()],
        suggested_window_entities: int,
    ) -> None:
        self.window_start = window_start
        self.window_end = window_end
        self.entity_count = entity_count
        self.missing_start = missing_start
        self.missing_end = missing_end
        self.suggested_window = suggested_window
        self.suggested_window_entities = suggested_window_entities
        super().__init__(
            f"no entity of {entity_count} has rows in both {window_start} and {window_end}"
        )


class GrowthAmbiguousEntityError(Exception):
    """More than one row shares one entity's `time_period` inside the
    window -- `entity_by` doesn't uniquely identify one row per period, so
    there is no single value to call "the" observation for that year.
    """

    def __init__(self, *, entity_by: Sequence[str], available: Sequence[str]) -> None:
        self.entity_by = tuple(entity_by)
        self.available = tuple(available)
        super().__init__(
            f"entity_by {self.entity_by!r} does not uniquely identify one row per time_period"
        )


class GrowthNonFiniteResultError(Exception):
    """No entity holding both endpoints has a finite growth value.
    `zero_or_sign_crossing_base` counts the entities whose zero or
    sign-crossing base leaves a relative change undefined (only under
    `change="relative"`). `non_finite` counts those whose computed value
    overflowed to a non-finite float. Together they make `entity_count`."""

    def __init__(
        self, *, entity_count: int, zero_or_sign_crossing_base: int, non_finite: int
    ) -> None:
        self.entity_count = entity_count
        self.zero_or_sign_crossing_base = zero_or_sign_crossing_base
        self.non_finite = non_finite
        super().__init__(
            f"no finite growth for any of {entity_count} entities: "
            f"{zero_or_sign_crossing_base} zero or sign-crossing base, {non_finite} non-finite"
        )


# The closed two-member vocabulary `GrowthOpSpec.change` and `run_growth`
# both compare against -- see `op_spec.GrowthOpSpec`'s own docstring for why
# a `Literal` here rather than an engine-enforced open `str`.
GrowthChange = Literal["relative", "absolute"]


@dataclass(frozen=True, kw_only=True)
class EntityGrowth:
    """One entity's own growth result, addressed by its own `entity_by`-keyed
    identity -- the same "never trust row position" discipline
    `analysis_mcp.engine.aggregate.GroupSummary` follows.

    `growth_rate` is a relative rate (a dimensionless fraction) under
    `change="relative"`, and an absolute change in the input's own unit
    under `change="absolute"` -- the field keeps its established name across
    both, since which reading applies is `GrowthResult.change`'s own job to
    state, not a second field's.

    `realised_coverage_fraction` is the share of the window's calendar years
    this entity has a row in. It is below 1.0 when the entity lacks interior
    years.
    """

    entity_key: tuple[tuple[str, str], ...]
    value_start: float
    value_end: float
    growth_rate: float
    realised_coverage_fraction: float


@dataclass(frozen=True, kw_only=True)
class GrowthResult:
    """Everything `growth_tool.py` needs to mint identity, write the store,
    and render the digest.

    `observed_periods` is every distinct `time_period` the computed
    entities' rows carry inside the window, sorted. `growth_tool.py` passes
    them to `frame_ref_resolution.build_interval` to say whether the window
    is contiguous across them.
    """

    frame: Frame
    resolved_unit: str | UnitRequest | None
    facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    window_start: str
    window_end: str
    entity_growths: tuple[EntityGrowth, ...]
    observed_periods: tuple[str, ...] = ()
    change: GrowthChange = "relative"
    exclusions: Exclusions = field(default_factory=Exclusions)


def _indices(positions: Iterable[int]) -> pa.Array:
    """`positions` as an `int64` array `take` accepts, empty included."""
    return pa.array(positions, pa.int64())


def _group_rows(indices: Sequence[int], groups: Groups) -> pa.Array:
    """The row positions of every group in `indices`, for a table whose
    groups are contiguous runs in `groups.offsets` order."""
    return _indices(
        row for index in indices for row in range(groups.offsets[index], groups.offsets[index + 1])
    )


def _endpoints_missing(
    table: pa.Table,
    windowed: pa.Table,
    groups: Groups,
    *,
    entity_by: Sequence[str],
    window_start: str,
    window_end: str,
) -> GrowthEndpointsMissingError:
    """The refusal for a call where no group of `windowed` holds both
    endpoints, counted over every entity `table` (the narrowed input before
    windowing) holds."""
    entity_count = group_rows(table, entity_by).count
    periods = cast("list[str]", windowed.column(TIME_PERIOD_COLUMN).to_pylist())
    # One year set per entity with an in-window row, in sorted entity order.
    held = [set(periods[start:end]) for start, end in pairwise(groups.offsets)]
    shared = sorted(set.intersection(*held)) if held else []
    if len(shared) > 1:
        suggested: tuple[str, str] | tuple[()] = (shared[0], shared[-1])
    else:
        # `max` keeps the first of equally wide spans.
        first, last = max(
            ((min(years), max(years)) for years in held),
            key=lambda span: int(span[1]) - int(span[0]),
            default=("", ""),
        )
        suggested = (first, last) if first != last else ()
    return GrowthEndpointsMissingError(
        window_start=window_start,
        window_end=window_end,
        entity_count=entity_count,
        missing_start=entity_count - sum(window_start in years for years in held),
        missing_end=entity_count - sum(window_end in years for years in held),
        suggested_window=suggested,
        suggested_window_entities=sum(
            bool(suggested) and set(suggested) <= years for years in held
        ),
    )


def run_growth(
    frame: Frame,
    *,
    window_start: str,
    window_end: str,
    entity_by: Sequence[str],
    change: GrowthChange = "relative",
    pinned_unit: str | UnitRequest | None = None,
) -> GrowthResult:
    """Compute each entity's growth across `[window_start, window_end]`
    (whole calendar years) -- a relative rate or an absolute change per
    `change`. See the module docstring for the endpoint and non-finite-base
    policies; see each raised exception's own docstring for its refusal
    condition.
    """
    stat_var = require_single_stat_var(frame)
    entity_by = validate_entity_columns(frame, entity_by)
    # `exclude_unitless=True`: `resolved.unit` is one frame-wide label for
    # every entity's growth rate. An entity whose rows leave `unit` unbound
    # entirely is excluded; a wholly-unitless entity never becomes a group in
    # the output, only surfaced as a row-count warning. A partially-unitless
    # entity is computed when its unit-bound rows include both endpoints and
    # excluded as `endpoint_missing` otherwise.
    table, resolved, exclusions = narrow_and_resolve(
        frame, pinned_unit=pinned_unit, exclude_unitless=True
    )

    expected_years = [str(year) for year in range(int(window_start), int(window_end) + 1)]

    in_window = in_periods(table, expected_years)
    exclusions = record_exclusion(
        exclusions,
        frame=frame,
        reason="outside_window",
        dropped=table.filter(array_kernel("invert", in_window)),
    )
    # Sorted by entity then period, so groups come out in sorted entity order
    # and each group's rows in period order. Role values are never null, and
    # Arrow orders strings by UTF-8 bytes, which is code-point order.
    windowed = table.filter(in_window).sort_by(
        [(column, "ascending") for column in (*entity_by, TIME_PERIOD_COLUMN)]
    )

    if not keys_unique(windowed, (*entity_by, TIME_PERIOD_COLUMN)):
        raise GrowthAmbiguousEntityError(
            entity_by=entity_by, available=available_entity_columns(frame)
        )

    # With one row per period in period order, a group holds both endpoints
    # exactly when its first row is `window_start` and its last `window_end`.
    groups = group_rows(windowed, entity_by)
    periods = windowed.column(TIME_PERIOD_COLUMN)
    firsts = periods.take(_indices(groups.offsets[:-1])).to_pylist()
    lasts = periods.take(_indices(end - 1 for end in groups.offsets[1:])).to_pylist()
    kept = [
        index
        for index in range(groups.count)
        if firsts[index] == window_start and lasts[index] == window_end
    ]
    if not kept:
        raise _endpoints_missing(
            table,
            windowed,
            groups,
            entity_by=entity_by,
            window_start=window_start,
            window_end=window_end,
        )

    kept_set = set(kept)
    exclusions = record_exclusion(
        exclusions,
        frame=frame,
        reason="endpoint_missing",
        dropped=windowed.take(
            _group_rows([index for index in range(groups.count) if index not in kept_set], groups)
        ),
    )

    keys = (
        list(zip(*(groups.keys.column(column).to_pylist() for column in entity_by), strict=True))
        if entity_by
        else [()]
    )
    values = windowed.column(VALUE_COLUMN)
    starts_at = cast(
        "list[float]", values.take(_indices(groups.offsets[index] for index in kept)).to_pylist()
    )
    ends_at = cast(
        "list[float]",
        values.take(_indices(groups.offsets[index + 1] - 1 for index in kept)).to_pylist(),
    )

    entity_growths: list[EntityGrowth] = []
    computed: list[int] = []
    undefined_base: list[int] = []
    non_finite: list[int] = []
    for index, value_start, value_end in zip(kept, starts_at, ends_at, strict=True):
        if change == "relative":
            if value_start == 0.0 or value_end == 0.0 or (value_start < 0) != (value_end < 0):
                undefined_base.append(index)
                continue
            growth_rate = (value_end - value_start) / value_start
        else:
            # `change == "absolute"`: subtraction is defined over a zero or
            # sign-crossing base, so the relative-only check above does not
            # apply here.
            growth_rate = value_end - value_start
        if not math.isfinite(growth_rate):
            non_finite.append(index)
            continue
        # Normalised here, at the engine boundary, not left to the digest's
        # own render-time `+ 0.0` -- signed zero is an engine-level
        # invariant this toolkit states for every verb, not a render patch.
        growth_rate += 0.0

        computed.append(index)
        size = groups.offsets[index + 1] - groups.offsets[index]
        entity_growths.append(
            EntityGrowth(
                entity_key=tuple(zip(entity_by, cast("tuple[str, ...]", keys[index]), strict=True)),
                value_start=value_start,
                value_end=value_end,
                growth_rate=growth_rate,
                realised_coverage_fraction=size / len(expected_years),
            )
        )

    if not computed:
        raise GrowthNonFiniteResultError(
            entity_count=len(kept),
            zero_or_sign_crossing_base=len(undefined_base),
            non_finite=len(non_finite),
        )
    exclusions = record_exclusion(
        exclusions,
        frame=frame,
        reason="zero_or_sign_crossing_base",
        dropped=windowed.take(_group_rows(undefined_base, groups)),
    )
    exclusions = record_exclusion(
        exclusions,
        frame=frame,
        reason="non_finite",
        dropped=windowed.take(_group_rows(non_finite, groups)),
    )
    computed_rows = windowed.take(_group_rows(computed, groups))

    warnings = resolved.warnings
    gapped = sum(growth.realised_coverage_fraction < 1.0 for growth in entity_growths)
    if gapped:
        warnings = (
            *warnings,
            f"{gapped} of {len(entity_growths)} entit(y/ies) have no row in some year between "
            f"{window_start} and {window_end}; growth reads only the two endpoint years, "
            "so those gaps do not change their values",
        )

    count = len(entity_growths)
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        STAT_VAR_COLUMN: repeated(stat_var, count),
        TIME_PERIOD_COLUMN: repeated(span_label([window_start, window_end]), count),
        VALUE_COLUMN: pa.array([growth.growth_rate for growth in entity_growths], pa.float64()),
        **{column: groups.keys.column(column).take(computed) for column in entity_by},
        **facet_columns(resolved.facet_selections, count=count),
    }
    # A relative growth rate is dimensionless -- `engine/share_of.py`'s own
    # output row writes `None` for the same reason, and a missing `unit`
    # column builds as nulls. `resolved.unit` still selected which input rows
    # this entity's `value_start`/`value_end` were read from; it just doesn't
    # denominate the ratio those two values produced. An absolute change
    # keeps the input's own unit -- it is that unit's own value, subtracted
    # -- which is `None` too when the input itself was already unitless (e.g.
    # a prior relative-growth or share_of result).
    if change == "absolute":
        columns[UNIT_COLUMN] = unit_column(computed_rows, count=count)
    output_frame = frame_from_columns(columns, role_keys=entity_by)

    return GrowthResult(
        frame=output_frame,
        resolved_unit=resolved.unit,
        facet_selections=resolved.facet_selections,
        warnings=warnings,
        window_start=window_start,
        window_end=window_end,
        entity_growths=tuple(entity_growths),
        observed_periods=tuple(distinct(computed_rows, TIME_PERIOD_COLUMN)),
        change=change,
        exclusions=exclusions,
    )
