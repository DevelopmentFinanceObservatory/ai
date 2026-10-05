"""The `growth` engine: growth-rate computation over a frame's table,
across each entity's calendar window, refusing on any gap year inside the window
as a break; the refusal names the gap and a gap-free window.

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
unit. Every calendar year between `window_start` and `window_end`
(inclusive) must carry a row for the entity, or this refuses
(`GrowthGapYearError`) naming the missing years and the longest gap-free
sub-window, rather than skipping the gap or computing across it regardless.
This is a "refuse" instance of `analysis_mcp.engine.common`'s own
partial-overlap policy -- see that module's docstring for the reasoning,
including why growth counts here despite taking only one input. Under
`change="relative"` only, a zero or sign-crossing base (`value_start` zero,
or `value_start`/`value_end` differing in sign) makes a percentage change
undefined and refuses (`GrowthNonFiniteResultError`) before any division is
attempted -- checked explicitly, not caught after a `ZeroDivisionError` or a
`**` on a negative base silently promoting to a complex number. An absolute
change over the same base is defined (subtraction, unlike division, does
not break at zero or across a sign crossing), so that refusal does not
apply there; either mode still refuses a computed value that overflows to a
non-finite float.

Each output row's `time_period` is the window as a span,
`span_label([window_start, window_end])` (`2018/2022`), so two growth results
over different windows carry different labels.
"""

from __future__ import annotations

import math
from collections.abc import Container, Mapping, Sequence
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


class GrowthGapYearError(Exception):
    """A calendar year inside `[window_start, window_end]` has no row for
    this entity. `gap_years` is every missing year, sorted;
    `longest_gap_free_window` is the longest run of consecutive present
    years inside the window, `()` if the entity has no data in the window
    at all -- both are what the `growth_gap_year` refusal's fix names.
    """

    def __init__(
        self,
        *,
        entity_key: Mapping[str, str],
        gap_years: Sequence[str],
        longest_gap_free_window: tuple[str, str] | tuple[()],
    ) -> None:
        self.entity_key = dict(entity_key)
        self.gap_years = tuple(gap_years)
        self.longest_gap_free_window = longest_gap_free_window
        super().__init__(
            f"gap year(s) {self.gap_years!r} inside the window for entity {self.entity_key!r}"
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
    """Under `change="relative"`, a zero or sign-crossing base makes this
    entity's growth rate undefined. Under either mode, the computed value
    overflowed to a non-finite float."""

    def __init__(self, *, entity_key: Mapping[str, str], reason: str) -> None:
        self.entity_key = dict(entity_key)
        self.reason = reason
        super().__init__(f"non-finite growth rate for entity {self.entity_key!r}: {reason}")


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
    """

    entity_key: tuple[tuple[str, str], ...]
    value_start: float
    value_end: float
    growth_rate: float
    realised_coverage_fraction: float


@dataclass(frozen=True, kw_only=True)
class GrowthResult:
    """Everything `growth_tool.py` needs to mint identity, write the store,
    and render the digest."""

    frame: Frame
    resolved_unit: str | UnitRequest | None
    facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    window_start: str
    window_end: str
    entity_growths: tuple[EntityGrowth, ...]
    change: GrowthChange = "relative"
    exclusions: Exclusions = field(default_factory=Exclusions)


def _longest_gap_free_window(
    expected_years: Sequence[str], present: Container[str]
) -> tuple[str, str] | tuple[()]:
    """The longest run of consecutive `expected_years` entries found in
    `present`, as `(first, last)`; `()` if none are present at all."""
    best: tuple[str, str] | tuple[()] = ()
    best_len = 0
    run_start: str | None = None
    run_len = 0
    for year in expected_years:
        if year in present:
            if run_start is None:
                run_start = year
            run_len += 1
            if run_len > best_len:
                best_len = run_len
                best = (run_start, year)
        else:
            run_start = None
            run_len = 0
    return best


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
    `change`. See the module docstring for the gap-year and non-finite-base
    policies; see each raised exception's own docstring for its refusal
    condition.
    """
    stat_var = require_single_stat_var(frame)
    entity_by = validate_entity_columns(frame, entity_by)
    # `exclude_unitless=True`: `resolved.unit` is one frame-wide label for
    # every entity's growth rate. An entity whose rows leave `unit` unbound
    # entirely is excluded; a wholly-unitless entity never becomes a group in
    # the output, only surfaced as a row-count warning. Partially-unitless
    # entities are caught by `GrowthGapYearError` (missing years in the window).
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

    # An empty window reports one all-years gap with no entity named, whether
    # `entity_by` is empty (one implicit entity) or not (no entity at all).
    if not windowed.num_rows:
        raise GrowthGapYearError(
            entity_key={}, gap_years=tuple(expected_years), longest_gap_free_window=()
        )

    if not keys_unique(windowed, (*entity_by, TIME_PERIOD_COLUMN)):
        raise GrowthAmbiguousEntityError(
            entity_by=entity_by, available=available_entity_columns(frame)
        )

    groups = group_rows(windowed, entity_by)
    keys = (
        list(zip(*(groups.keys.column(column).to_pylist() for column in entity_by), strict=True))
        if entity_by
        else [()]
    )
    # With one row per period, a group is gap-free exactly when it holds every
    # expected year; its rows then run through `sorted(expected_years)`.
    sizes = [end - start for start, end in pairwise(groups.offsets)]
    complete = next(
        (index for index, size in enumerate(sizes) if size != len(expected_years)), groups.count
    )
    starts = groups.offsets[:complete]
    sorted_years = sorted(expected_years)
    values = windowed.column(VALUE_COLUMN)
    starts_at = cast(
        "list[float]",
        values.take([start + sorted_years.index(window_start) for start in starts]).to_pylist()
        if complete
        else [],
    )
    ends_at = cast(
        "list[float]",
        values.take([start + sorted_years.index(window_end) for start in starts]).to_pylist()
        if complete
        else [],
    )

    entity_growths: list[EntityGrowth] = []
    for key, size, value_start, value_end in zip(
        keys[:complete], sizes[:complete], starts_at, ends_at, strict=True
    ):
        entity_key_map = dict(zip(entity_by, cast("tuple[str, ...]", key), strict=True))
        if change == "relative":
            if value_start == 0.0 or value_end == 0.0 or (value_start < 0) != (value_end < 0):
                raise GrowthNonFiniteResultError(
                    entity_key=entity_key_map, reason="zero or sign-crossing base"
                )
            growth_rate = (value_end - value_start) / value_start
        else:
            # `change == "absolute"`: subtraction is defined over a zero or
            # sign-crossing base, so the relative-only refusal above does
            # not apply here.
            growth_rate = value_end - value_start
        if not math.isfinite(growth_rate):
            raise GrowthNonFiniteResultError(entity_key=entity_key_map, reason="overflow")
        # Normalised here, at the engine boundary, not left to the digest's
        # own render-time `+ 0.0` -- signed zero is an engine-level
        # invariant this toolkit states for every verb, not a render patch.
        growth_rate += 0.0

        entity_growths.append(
            EntityGrowth(
                entity_key=tuple(entity_key_map.items()),
                value_start=value_start,
                value_end=value_end,
                growth_rate=growth_rate,
                # Computed from the group's row count, not a bare `1.0`
                # literal. Any gap anywhere in the window causes the whole
                # call to refuse, so a returned entity's window is by
                # construction gap-free and carries 1.0.
                realised_coverage_fraction=size / len(expected_years),
            )
        )

    if complete < groups.count:
        present = set(
            cast(
                "list[str]",
                windowed.column(TIME_PERIOD_COLUMN)
                .slice(groups.offsets[complete], sizes[complete])
                .to_pylist(),
            )
        )
        raise GrowthGapYearError(
            entity_key=dict(zip(entity_by, cast("tuple[str, ...]", keys[complete]), strict=True)),
            gap_years=[year for year in expected_years if year not in present],
            longest_gap_free_window=_longest_gap_free_window(expected_years, present),
        )

    count = len(entity_growths)
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        STAT_VAR_COLUMN: repeated(stat_var, count),
        TIME_PERIOD_COLUMN: repeated(span_label([window_start, window_end]), count),
        VALUE_COLUMN: pa.array([growth.growth_rate for growth in entity_growths], pa.float64()),
        **{column: groups.keys.column(column) for column in entity_by},
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
        columns[UNIT_COLUMN] = unit_column(windowed, count=count)
    output_frame = frame_from_columns(columns, role_keys=entity_by)

    return GrowthResult(
        frame=output_frame,
        resolved_unit=resolved.unit,
        facet_selections=resolved.facet_selections,
        warnings=resolved.warnings,
        window_start=window_start,
        window_end=window_end,
        entity_growths=tuple(entity_growths),
        change=change,
        exclusions=exclusions,
    )
