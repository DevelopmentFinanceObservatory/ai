"""The `share_of` engine: numerator/denominator ratio over two slices.
`share_of` refuses on a denominator/numerator grain mismatch, with both
coverages in the refusal's fix payload.

A "slice" is `frame_ref` (or `denominator_frame_ref`) plus an `entity`
binding, the same `{role_column: value}` narrowing shape
`analysis_mcp.engine.compare` uses. The one place `share_of` diverges from
`compare`: an `entity` binding that leaves more than one row per period on a
side is not ambiguous here, it is summed -- `math.fsum` is used for any
summed side -- because a share's denominator is very often a
deliberate total ("USA's ODA as a share of *all* DAC donors' ODA" binds
nothing on the denominator side, summing every donor's own row per period).

Each side resolves its own unit independently, and the two sides' units
must match -- a ratio of two different units (dollars over a headcount, say)
is silently wrong, so `run_share_of` refuses (`ShareOfUnitMismatchError`) if
the units differ. One side's rows leaving `unit` unbound entirely is a
separate case (`ShareOfUnitUnboundError`): one side has no declared unit at
all, so there is no second unit to compare. Facets are checked the same way
(`ShareOfFacetMismatchError`): each side resolves its own facet selections
independently; nothing else catches a numerator scoped to `provenance=dcid:P1`
paired against a denominator scoped to `dcid:P2`. Neither check runs when
either side's slice is empty; `ShareOfGrainMismatchError` names that real
problem once the period sets are compared.

The grain-match policy: each side's own summed `{time_period: value}` series
is computed independently, and the two sides' period sets must be identical
(a stricter bar than intersecting as `compare` does). Any mismatch refuses
with `ShareOfGrainMismatchError`, naming both sides' periods and coverage.
This is the refusing side of the partial-overlap policy in
`analysis_mcp.engine.common`, whose docstring carries the reasoning. A
denominator period whose summed value is exactly zero makes the ratio
undefined and refuses (`ShareOfNonFiniteResultError`) before any division
is attempted.

`numerator_by`/`denominator_by` key the sum: `numerator_by` names role
column(s) on the numerator's own frame, so the sum runs once per (key,
period) instead of collapsing every row into one period-level total, and
the output frame carries one row per (key, period), keyed by `numerator_by`
itself. `denominator_by` pairs each `numerator_by` column with the
denominator's own column carrying the same role -- **by name, never by
position**: two independently sorted column lists silently pair the wrong
columns (`identity._normalise_parameters` and
`frame_ref_resolution.canonical_columns_or_refusal` both sort every
array-shaped parameter, so two callers naming
`numerator_by=["donor","sector"]` and `denominator_by` as a positional
`["provider","area"]` list would see that sort to `["area","provider"]`
against `["donor","sector"]` -- donor paired with area). `denominator_by`
left empty means a pooled denominator: one total per period, shared by
every numerator key -- the same total an un-keyed call already computes.
When non-empty, `denominator_by` must be **total** over `numerator_by`
(every numerator_by column has an entry): a partial mapping has no defined
pairing for the column(s) it leaves out (`ShareOfKeyUnknownError`). A
numerator key with no matching denominator group under this mapping
refuses distinctly (`ShareOfKeyUnmatchedError`). `numerator_by=()` (the
default) computes a single total, with no key.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal, cast

import pyarrow as pa
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    VALUE_COLUMN,
    UnitRequest,
)

from analysis_mcp.engine.columnar import frame_from_columns, group_rows, repeated
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    combine_exclusions,
    cross_side_one_sided_facet_axes,
    first_cross_side_facet_mismatch,
    narrow_and_resolve,
    require_one_place_level,
    require_single_stat_var,
    role_columns,
    validate_entity_columns,
)


class ShareOfGrainMismatchError(Exception):
    """The numerator's and denominator's own summed period sets differ.

    `numerator_periods`/`denominator_periods` are each side's own periods,
    sorted; `numerator_coverage_fraction`/`denominator_coverage_fraction` are
    each side's own "how much of my periods does the other side also cover"
    fraction; `matched_periods` is the sorted intersection -- every period
    both sides actually share, if any -- naming the exact `time_period`
    values a `filter` retry (`category_match` on `time_period`) would narrow
    each side down to, since narrowing both sides to `matched_periods` and
    retrying `share_of` on the results is a repair fully inside this
    toolkit. All four land in the refusal's `fix` payload together, per the
    declared policy: never report one side's numbers and proceed on the
    other's say-so.

    `key` names the offending `numerator_by` key -- empty (`()`) for an
    un-keyed call, where this whole call has exactly one implicit key. The
    identical-period-set rule applies per key, so a multi-key call reports
    the first key (in sorted order) whose own periods disagree, never a
    pooled figure across every key.
    """

    def __init__(
        self,
        *,
        numerator_periods: Sequence[str],
        denominator_periods: Sequence[str],
        numerator_coverage_fraction: float,
        denominator_coverage_fraction: float,
        matched_periods: Sequence[str],
        key: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.numerator_periods = tuple(numerator_periods)
        self.denominator_periods = tuple(denominator_periods)
        self.numerator_coverage_fraction = numerator_coverage_fraction
        self.denominator_coverage_fraction = denominator_coverage_fraction
        self.matched_periods = tuple(matched_periods)
        self.key = tuple(key)
        super().__init__(
            f"numerator periods {self.numerator_periods!r} do not match denominator "
            f"periods {self.denominator_periods!r} for key {dict(self.key)!r}"
        )


class ShareOfNonFiniteResultError(Exception):
    """A period's denominator summed to exactly zero, or the ratio overflowed."""

    def __init__(self, *, time_period: str, reason: str) -> None:
        self.time_period = time_period
        self.reason = reason
        super().__init__(f"non-finite share for period {time_period!r}: {reason}")


class ShareOfUnitMismatchError(Exception):
    """The numerator's and denominator's independently-resolved units differ.

    A share is a ratio, but the two sides are summed and compared before
    that division happens, and a numerator/denominator pair reported in two
    different units (USD over a headcount, say) is exactly the same
    "silently wrong" combination `compare`'s own `CompareUnitMismatchError`
    refuses -- checked before either side's periods are compared, let alone
    divided.
    """

    def __init__(
        self, *, numerator_unit: str | UnitRequest, denominator_unit: str | UnitRequest
    ) -> None:
        self.numerator_unit = numerator_unit
        self.denominator_unit = denominator_unit
        super().__init__(
            f"numerator resolved to unit {numerator_unit!r}, denominator to {denominator_unit!r}"
        )


class ShareOfUnitUnboundError(Exception):
    """One side's own rows leave `unit` unbound entirely, while the other
    side resolved to a real declared unit. Distinct from
    `ShareOfUnitMismatchError`, which fires when both sides resolve to a
    real, differing unit -- here there is no second resolved unit to
    disagree with, only a fact about `unbound_side`'s own data: none of its
    own rows ever declared one. Silently inheriting `bound_side`'s own
    resolved unit would state something `unbound_side`'s own rows never
    declared -- the same distinction `analysis_mcp.engine.compare.
    CompareUnitUnboundError` already draws for `compare`.
    """

    def __init__(
        self, *, unbound_side: str, bound_side: str, bound_unit: str | UnitRequest
    ) -> None:
        self.unbound_side = unbound_side
        self.bound_side = bound_side
        self.bound_unit = bound_unit
        super().__init__(
            f"{unbound_side} side's own rows leave `unit` unbound; {bound_side} resolved "
            f"to {bound_unit!r}"
        )


class ShareOfFacetMismatchError(Exception):
    """Each side resolved its own facet selections independently, but they
    disagree on at least one axis -- a `provenance=dcid:P1` numerator
    compared against a `provenance=dcid:P2` denominator, say. Distinct from
    `analysis_mcp.engine.common.FacetHeterogeneousError`, which fires when
    *one* slice itself carries more than one declared value on an axis;
    here each side is internally uniform (or unbound) on its own, and it's
    the pairing across sides that's unsafe -- the same "each side resolves
    independently, the pairing is what's checked" shape
    `ShareOfUnitMismatchError` already applies to `unit`.
    """

    def __init__(self, *, axis: str, numerator_ref: str, denominator_ref: str) -> None:
        self.axis = axis
        self.numerator_ref = numerator_ref
        self.denominator_ref = denominator_ref
        super().__init__(
            f"numerator resolved facet {axis!r} to {numerator_ref!r}, denominator to "
            f"{denominator_ref!r}"
        )


class ShareOfKeyUnknownError(Exception):
    """`denominator_by` is non-empty but does not cover every `numerator_by`
    column -- a partial mapping has no defined pairing for the column(s) it
    leaves out. `column` is the first `numerator_by` column with no
    `denominator_by` entry (sorted, for a deterministic message);
    `denominator_by_keys` are the keys `denominator_by` does carry.
    """

    def __init__(
        self, *, column: str, numerator_by: Sequence[str], denominator_by_keys: Sequence[str]
    ) -> None:
        self.column = column
        self.numerator_by = tuple(numerator_by)
        self.denominator_by_keys = tuple(denominator_by_keys)
        super().__init__(
            f"denominator_by does not cover numerator_by column {column!r}; "
            f"denominator_by names: {self.denominator_by_keys!r}"
        )


class ShareOfKeyUnmatchedError(Exception):
    """A numerator key, realised from its own frame's rows, has no matching
    denominator group under `denominator_by`'s own mapping -- every row on
    the denominator side was grouped by the columns `denominator_by` maps
    `numerator_by` to, and none of those groups carries this particular
    combination of values. `available` lists every denominator key this
    call did realise, sorted, for a retry to compare against.
    """

    def __init__(
        self,
        *,
        key: Sequence[tuple[str, str]],
        available: Sequence[tuple[tuple[str, str], ...]],
    ) -> None:
        self.key = tuple(key)
        self.available = tuple(available)
        super().__init__(f"no denominator group matches numerator key {dict(self.key)!r}")


@dataclass(frozen=True, kw_only=True)
class PeriodShare:
    """One matched (key, period)'s own numerator/denominator values and
    their ratio, addressed by `time_period` and `key` -- the same "never
    trust row order" discipline `analysis_mcp.engine.compare.
    PeriodComparison` follows. `key` is `()` for an un-keyed call, where
    every row shares the one implicit key."""

    time_period: str
    numerator_value: float
    denominator_value: float
    share: float
    key: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, kw_only=True)
class ShareOfResult:
    """Everything `share_of_tool.py` needs to mint identity, write the
    store, and render the digest."""

    frame: Frame
    numerator_resolved_unit: str | UnitRequest | None
    denominator_resolved_unit: str | UnitRequest | None
    numerator_facet_selections: tuple[FacetSelection, ...]
    denominator_facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    numerator_periods: tuple[str, ...]
    denominator_periods: tuple[str, ...]
    numerator_coverage_fraction: float
    denominator_coverage_fraction: float
    period_shares: tuple[PeriodShare, ...]
    exclusions: Exclusions = field(default_factory=Exclusions)


def _one_sided_facet_warning(
    axis: str, numerator_ref: str | None, denominator_ref: str | None
) -> str:
    """One `cross_side_one_sided_facet_axes` finding, worded for share_of's
    numerator/denominator vocabulary. The result carries only the numerator's
    facet selections, so the warning states not just which side left the axis
    unbound, but whether the other side's value actually reaches the result.
    """
    if numerator_ref is not None:
        return (
            f"facet {axis!r} was left unbound on the denominator's own rows; the "
            f"numerator's own declared value {numerator_ref!r} is what the trace's own "
            "resolved_facets record."
        )
    return (
        f"facet {axis!r} was left unbound on the numerator's own rows; the "
        f"denominator's own declared value {denominator_ref!r} exists but the trace's "
        "own resolved_facets record carries no value for this axis (share_of's own "
        "resolved_facets always reflects the numerator's own facet selections)."
    )


def _summed_by_key_and_period(
    frame: Frame,
    *,
    side: Literal["numerator", "denominator"],
    entity: Mapping[str, str],
    by: Sequence[str],
    pinned_unit: str | UnitRequest | None,
) -> tuple[
    dict[tuple[str, ...], dict[str, float]],
    str | UnitRequest | None,
    tuple[FacetSelection, ...],
    tuple[str, ...],
    Exclusions,
]:
    """Narrow `frame` to `entity`'s own rows, resolve unit/facets over
    exactly that slice, then sum (`math.fsum`) every remaining row sharing
    both `by`'s own key values and a `time_period` -- one summed value per
    (key, period), never a per-row value, so more than one matching row in
    one cell is a deliberate total, not an ambiguity. `by=()` computes one
    key, the empty tuple, covering every remaining row. Refuses
    (`UnknownEntityColumnError`) if `entity` or `by` names a column this
    frame has no grain for, and (`PlaceLevelsMixedError`) if `frame`'s
    places share no level, whether this side sums across them or keys by them,
    as the harness's `planner.shapes` refuses either side of a share over such a frame.

    Returns a mapping keyed by `by`'s own key tuple to that key's own
    `{time_period: summed_value}` series -- a key with no rows at all is
    simply absent, never present with an empty series; the caller decides
    what an absent key means for its own side.
    """
    require_single_stat_var(frame)
    validate_entity_columns(frame, (*entity, *by))
    require_one_place_level(frame, role_columns(frame))
    # `exclude_unitless=True`: a row leaving `unit` unbound is excluded from
    # this side's sum, consistent with the policy `aggregate` applies.
    table, resolved, exclusions = narrow_and_resolve(
        frame, entity=entity, pinned_unit=pinned_unit, exclude_unitless=True, side=side
    )

    # Groups come in order of their first row, so keys land in this dict in
    # order of their first row and each key's periods in order of theirs:
    # the order an overflowing sum below is found in.
    groups = group_rows(table, (*by, TIME_PERIOD_COLUMN))
    values = cast("list[float]", table.column(VALUE_COLUMN).take(groups.indices).to_pylist())
    key_columns = [cast("list[str]", groups.keys.column(column).to_pylist()) for column in by]
    periods = cast("list[str]", groups.keys.column(TIME_PERIOD_COLUMN).to_pylist())
    by_key_and_period: dict[tuple[str, ...], dict[str, list[float]]] = {}
    for group, (start, end) in enumerate(pairwise(groups.offsets)):
        key = tuple(column[group] for column in key_columns)
        by_key_and_period.setdefault(key, {})[periods[group]] = values[start:end]

    summed: dict[tuple[str, ...], dict[str, float]] = {}
    for key, by_period in by_key_and_period.items():
        summed_periods: dict[str, float] = {}
        for period, values in by_period.items():
            try:
                # `math.fsum` raises `OverflowError` for a sum whose exact
                # value exceeds the finite double range, rather than
                # returning `inf` the way naive `+` accumulation would --
                # the same guard `aggregate.run_aggregate` and
                # `rank.run_rank` already apply to their own `math.fsum`
                # reduction.
                summed_periods[period] = math.fsum(values)
            except OverflowError as exc:
                raise ShareOfNonFiniteResultError(time_period=period, reason="overflow") from exc
        summed[key] = summed_periods

    return summed, resolved.unit, resolved.facet_selections, tuple(resolved.warnings), exclusions


def run_share_of(
    numerator_frame: Frame,
    denominator_frame: Frame,
    *,
    numerator_entity: Mapping[str, str],
    denominator_entity: Mapping[str, str],
    numerator_by: Sequence[str] = (),
    denominator_by: Mapping[str, str] | None = None,
    numerator_pinned_unit: str | UnitRequest | None = None,
    denominator_pinned_unit: str | UnitRequest | None = None,
) -> ShareOfResult:
    """Compute `numerator_frame`'s (narrowed by `numerator_entity`) share of
    `denominator_frame`'s (narrowed by `denominator_entity`) own value, per
    matched (key, period). See the module docstring for the keying,
    grain-match and zero-denominator policies; see each raised exception's
    own docstring for its refusal condition. `numerator_by=()` (the
    default) computes a single total, with no key.
    """
    numerator_by = tuple(numerator_by)
    denominator_by = dict(denominator_by or {})

    if denominator_by:
        missing = sorted(set(numerator_by) - set(denominator_by))
        if missing:
            raise ShareOfKeyUnknownError(
                column=missing[0],
                numerator_by=numerator_by,
                denominator_by_keys=tuple(sorted(denominator_by)),
            )
        denominator_columns: tuple[str, ...] = tuple(
            denominator_by[column] for column in numerator_by
        )
    else:
        denominator_columns = ()

    (
        numerator_grouped,
        numerator_unit,
        numerator_facets,
        numerator_warnings,
        numerator_exclusions,
    ) = _summed_by_key_and_period(
        numerator_frame,
        side="numerator",
        entity=numerator_entity,
        by=numerator_by,
        pinned_unit=numerator_pinned_unit,
    )
    (
        denominator_grouped,
        denominator_unit,
        denominator_facets,
        denominator_warnings,
        denominator_exclusions,
    ) = _summed_by_key_and_period(
        denominator_frame,
        side="denominator",
        entity=denominator_entity,
        by=denominator_columns,
        pinned_unit=denominator_pinned_unit,
    )

    # An un-keyed call (`by=()`) has exactly one implicit key, `()`,
    # whatever it summed to -- including empty. `_summed_by_key_and_period`
    # only ever records a key some row actually produced, so an entity
    # binding matching zero rows leaves the `()` key entirely absent rather
    # than present with an empty series. Reinstating it here keeps the
    # per-key loop below running its grain-mismatch check exactly once for
    # the un-keyed case.
    if not numerator_by and () not in numerator_grouped:
        numerator_grouped[()] = {}
    if not denominator_columns and () not in denominator_grouped:
        denominator_grouped[()] = {}

    # A keyed numerator whose own entity binding matched zero rows at all
    # has no key to report a per-key mismatch against -- refuse the same
    # "matched nothing" fault the un-keyed case already refuses, rather
    # than an empty per-key loop silently returning a vacuous success.
    if numerator_by and not numerator_grouped:
        raise ShareOfGrainMismatchError(
            numerator_periods=(),
            denominator_periods=(),
            numerator_coverage_fraction=0.0,
            denominator_coverage_fraction=0.0,
            matched_periods=(),
        )

    # Both checks only run when both sides actually have rows -- an empty
    # slice (an `entity` binding matching nothing) resolves to `unit=None`
    # with no facet selections (`narrow_and_resolve`'s documented
    # behaviour), which would otherwise misread here as a mismatch against
    # whatever the other, populated side resolved to. The real problem an
    # empty side has is that it has no periods at all, which the
    # `ShareOfGrainMismatchError` check below already names.
    numerator_has_rows = any(numerator_grouped.values())
    denominator_has_rows = any(denominator_grouped.values())
    cross_side_facet_warnings: tuple[str, ...] = ()
    if numerator_has_rows and denominator_has_rows:
        if numerator_unit is None or denominator_unit is None:
            if denominator_unit is not None:
                raise ShareOfUnitUnboundError(
                    unbound_side="numerator",
                    bound_side="denominator",
                    bound_unit=denominator_unit,
                )
            if numerator_unit is not None:
                raise ShareOfUnitUnboundError(
                    unbound_side="denominator", bound_side="numerator", bound_unit=numerator_unit
                )
        elif numerator_unit != denominator_unit:
            raise ShareOfUnitMismatchError(
                numerator_unit=numerator_unit, denominator_unit=denominator_unit
            )

        facet_mismatch = first_cross_side_facet_mismatch(numerator_facets, denominator_facets)
        if facet_mismatch is not None:
            axis, numerator_ref, denominator_ref = facet_mismatch
            raise ShareOfFacetMismatchError(
                axis=axis, numerator_ref=numerator_ref, denominator_ref=denominator_ref
            )

        # A one-sided axis (see `cross_side_one_sided_facet_axes`'s own
        # docstring) doesn't refuse, but it still deserves the same
        # visibility a single slice's own auto-selection already gives --
        # worded here, not in the shared helper, because only this verb
        # knows its own rule for which side's facets a caller actually
        # sees again: `share_of_tool.py`'s own trace records the
        # numerator's own facet selections as `resolved_facets`, never the
        # denominator's, regardless of which side happened to declare the
        # axis (`share_of`'s own output rows carry no facet columns at all
        # -- see `run_share_of`'s output frame below).
        cross_side_facet_warnings = tuple(
            _one_sided_facet_warning(axis, numerator_ref, denominator_ref)
            for axis, numerator_ref, denominator_ref in cross_side_one_sided_facet_axes(
                numerator_facets, denominator_facets
            )
        )

    # In per-axis mode (`denominator_columns` non-empty) every numerator key
    # must find its own denominator group, by construction one-to-one
    # (`denominator_columns` is `denominator_by` mapped through
    # `numerator_by`'s own order, so a numerator key tuple and its matching
    # denominator key tuple carry identical values). In pooled mode
    # (`denominator_columns` empty) every numerator key shares the single
    # `()` group.
    pooled_denominator = denominator_grouped.get(()) if not denominator_columns else None

    period_shares: list[PeriodShare] = []
    all_numerator_periods: set[str] = set()
    all_denominator_periods: set[str] = set()
    for key in sorted(numerator_grouped):
        numerator_by_period = numerator_grouped[key]
        if denominator_columns:
            if key not in denominator_grouped:
                raise ShareOfKeyUnmatchedError(
                    key=tuple(zip(numerator_by, key, strict=True)),
                    available=tuple(
                        tuple(zip(numerator_by, other_key, strict=True))
                        for other_key in sorted(denominator_grouped)
                    ),
                )
            denominator_by_period = denominator_grouped[key]
        else:
            denominator_by_period = pooled_denominator if pooled_denominator is not None else {}

        numerator_periods = tuple(sorted(numerator_by_period))
        denominator_periods = tuple(sorted(denominator_by_period))
        key_pairs = tuple(zip(numerator_by, key, strict=True))

        # `or not numerator_periods`: two *empty* period sets satisfy
        # `set(numerator_periods) != set(denominator_periods)` trivially
        # (both equal the empty set), which would otherwise fall through to
        # the empty `period_shares` loop below and return a vacuous success
        # -- `is_error=False`, a registered pair, "0 matched period(s)" --
        # for, say, a typo'd role value on both `numerator_entity` and
        # `denominator_entity` that matches no rows on either side. Refuse
        # the same as any other grain mismatch, per key; two keys with
        # genuinely matching, non-empty period sets proceed to the
        # per-period loop below.
        if set(numerator_periods) != set(denominator_periods) or not numerator_periods:
            matched = set(numerator_periods) & set(denominator_periods)
            numerator_coverage_fraction = (
                len(matched) / len(numerator_periods) if numerator_periods else 0.0
            )
            denominator_coverage_fraction = (
                len(matched) / len(denominator_periods) if denominator_periods else 0.0
            )
            raise ShareOfGrainMismatchError(
                numerator_periods=numerator_periods,
                denominator_periods=denominator_periods,
                numerator_coverage_fraction=numerator_coverage_fraction,
                denominator_coverage_fraction=denominator_coverage_fraction,
                matched_periods=sorted(matched),
                key=key_pairs,
            )

        for period in numerator_periods:
            numerator_value = numerator_by_period[period]
            denominator_value = denominator_by_period[period]
            if denominator_value == 0.0:
                raise ShareOfNonFiniteResultError(time_period=period, reason="zero denominator")
            share = numerator_value / denominator_value
            if not math.isfinite(share):
                raise ShareOfNonFiniteResultError(time_period=period, reason="overflow")
            # Signed zero is normalized at the engine boundary. This is an
            # engine-level invariant this toolkit states for every verb
            # (see `engine/growth.py`'s own `growth_rate += 0.0`), applied
            # so a `-0.0` cannot persist into the stored output frame where
            # it might be re-chained to another verb.
            share += 0.0
            period_shares.append(
                PeriodShare(
                    time_period=period,
                    numerator_value=numerator_value,
                    denominator_value=denominator_value,
                    share=share,
                    key=key_pairs,
                )
            )
        all_numerator_periods.update(numerator_periods)
        all_denominator_periods.update(denominator_periods)

    # Computed from the period sets every key's own `ShareOfGrainMismatchError`
    # check above already passed, not a bare `1.0` literal. Each contributing
    # key's own numerator periods equal its own denominator periods exactly
    # (the per-key check above refuses otherwise), so the union across every
    # key is equal on both sides too, and `matched_periods` here equals the
    # whole of both.
    numerator_periods = tuple(sorted(all_numerator_periods))
    denominator_periods = tuple(sorted(all_denominator_periods))
    matched_periods = set(numerator_periods) & set(denominator_periods)
    numerator_coverage_fraction = len(matched_periods) / len(numerator_periods)
    denominator_coverage_fraction = len(matched_periods) / len(denominator_periods)

    numerator_stat_var = require_single_stat_var(numerator_frame)
    # Inherits the numerator's own `stat_var` rather than minting a new one:
    # a share is a ratio, not a newly measured quantity, so there is no
    # real-world dcid this toolkit could mint for it. The output frame
    # requires `stat_var` to be `dcid:`-prefixed, so the only alternative is
    # fabricating a string that *looks* like a Data Commons reference and
    # resolves to nothing -- worse than inheriting the numerator's, since a
    # reader trusting `stat_var` would fail visibly against a real dcid and
    # silently against a fake one. The inherited value no longer describes
    # what this row's own `value` is measured in; `frame_ref_resolution`'s
    # derived-ness marker (`applied_policy.detail["derived"]`, populated in
    # `share_of_tool.py`) is what tells a downstream verb that.
    output_frame = frame_from_columns(
        {
            STAT_VAR_COLUMN: repeated(numerator_stat_var, len(period_shares)),
            TIME_PERIOD_COLUMN: pa.array(
                [ps.time_period for ps in period_shares], type=pa.string()
            ),
            VALUE_COLUMN: pa.array([ps.share for ps in period_shares], type=pa.float64()),
            **{
                column: pa.array([ps.key[index][1] for ps in period_shares], type=pa.string())
                for index, column in enumerate(numerator_by)
            },
        },
        role_keys=numerator_by,
    )

    return ShareOfResult(
        frame=output_frame,
        numerator_resolved_unit=numerator_unit,
        denominator_resolved_unit=denominator_unit,
        numerator_facet_selections=numerator_facets,
        denominator_facet_selections=denominator_facets,
        warnings=tuple(
            dict.fromkeys((*numerator_warnings, *denominator_warnings, *cross_side_facet_warnings))
        ),
        numerator_periods=numerator_periods,
        denominator_periods=denominator_periods,
        numerator_coverage_fraction=numerator_coverage_fraction,
        denominator_coverage_fraction=denominator_coverage_fraction,
        period_shares=tuple(period_shares),
        exclusions=combine_exclusions(numerator_exclusions, denominator_exclusions),
    )
