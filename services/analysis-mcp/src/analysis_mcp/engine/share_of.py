"""The `share_of` engine: numerator/denominator ratio over two slices.
`share_of` divides over the periods both sides report and records the rest.

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
all, so there is no second unit to compare. Facets are checked across the
sides too: each side resolves its own facet selections independently, and
when both sides hold the same StatVar, a numerator scoped to
`provenance=dcid:P1` paired against a denominator scoped to `dcid:P2` refuses
(`ShareOfFacetMismatchError`), since that is one measure from two sources.
Two different StatVars (debt service over GDP) divide with a warning naming
both selections. Neither check runs when
either side's slice is empty; `ShareOfGrainMismatchError` names that real
problem once the period sets are compared.

The period policy is `quotient_of`'s, through the same
`analysis_mcp.engine.two_sided.pair_periods`: each side's own summed
`{time_period: value}` series is computed independently, each numerator key
divides over the periods it shares with its denominator group, and every
(key, period) either side holds outside them is recorded as a
`period_unmatched` exclusion with its side. A key sharing no period at all
refuses (`ShareOfGrainMismatchError`). This is the proceeding side of the
partial-overlap policy in `analysis_mcp.engine.common`. A denominator
period whose summed value is exactly zero makes the ratio undefined and
refuses (`ShareOfNonFiniteResultError`) before any division is attempted.

A side in a composed unit (dollars per person, a `quotient_of` result) is
one figure per (key, period) or nothing: a cell holding more than one such
row refuses (`analysis_mcp.engine.aggregate.ComposedUnitSumError`, naming
the side), since figures per head have no total. Two sides of one row per
cell divide as any two figures in one unit do.

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

Every result row states what it holds. Its `value` is the share in per cent
(numerator / denominator x 100), its `unit` is
`common.unit_refs.PERCENT_UNIT`, the ref dfo-apis writes on a published
share such as a "% of GDP" series, and its `stat_var` is the caller-supplied
`stat_var`, the operation's own minted ref, as `quotient_of` labels its
rows. A computed share and a published one therefore compare and plot
together.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import pyarrow as pa
from common.unit_refs import PERCENT_UNIT
from dfo_frame_grammar import Frame, unit_cell_array
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    UnitCell,
)

from analysis_mcp.engine.columnar import repeated
from analysis_mcp.engine.common import (
    Exclusions,
    FacetSelection,
    combine_exclusions,
    cross_side_differing_facet_axes,
    cross_side_one_sided_facet_axes,
    first_cross_side_facet_mismatch,
    frame_keeping_role_types,
)
from analysis_mcp.engine.two_sided import (
    DenominatorByPartialError,
    KeyPairs,
    KeyPeriods,
    NonFiniteCellError,
    denominator_key_columns,
    divide_cell,
    pair_periods,
    record_unmatched_periods,
    sum_side,
)


class ShareOfGrainMismatchError(Exception):
    """A numerator key shares no period with its denominator group, so it
    has no share to report.

    `numerator_periods`/`denominator_periods` are that key's own periods on
    each side, sorted, and `key` names it -- empty (`()`) for an un-keyed
    call, where this whole call has exactly one implicit key. A multi-key
    call reports the first key, in sorted order, sharing no period. Both
    tuples empty means a side's entity binding matched no row: both sides
    for an un-keyed call, the numerator for a keyed one.
    """

    def __init__(
        self,
        *,
        numerator_periods: Sequence[str],
        denominator_periods: Sequence[str],
        key: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.numerator_periods = tuple(numerator_periods)
        self.denominator_periods = tuple(denominator_periods)
        self.key = tuple(key)
        super().__init__(
            f"numerator periods {self.numerator_periods!r} share none with denominator "
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

    def __init__(self, *, numerator_unit: UnitCell, denominator_unit: UnitCell) -> None:
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

    def __init__(self, *, unbound_side: str, bound_side: str, bound_unit: UnitCell) -> None:
        self.unbound_side = unbound_side
        self.bound_side = bound_side
        self.bound_unit = bound_unit
        super().__init__(
            f"{unbound_side} side's own rows leave `unit` unbound; {bound_side} resolved "
            f"to {bound_unit!r}"
        )


class ShareOfFacetMismatchError(Exception):
    """Both sides hold the same `stat_var` and resolved their own facet
    selections independently, but they disagree on at least one axis, for
    example a `provenance=dcid:P1` numerator compared against a
    `provenance=dcid:P2` denominator. Distinct from
    `analysis_mcp.engine.common.FacetHeterogeneousError`, which fires when
    *one* slice itself carries more than one declared value on an axis;
    here each side is internally uniform (or unbound) on its own, and it's
    the pairing across sides that's unsafe -- the same "each side resolves
    independently, the pairing is what's checked" shape
    `ShareOfUnitMismatchError` already applies to `unit`.
    """

    def __init__(
        self, *, axis: str, numerator_ref: str, denominator_ref: str, stat_var: str
    ) -> None:
        self.axis = axis
        self.numerator_ref = numerator_ref
        self.denominator_ref = denominator_ref
        self.stat_var = stat_var
        super().__init__(
            f"both sides hold {stat_var!r}; numerator resolved facet {axis!r} to "
            f"{numerator_ref!r}, denominator to {denominator_ref!r}"
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
    their share in per cent, the `value` its output row holds, addressed by
    `time_period` and `key` -- the same "never trust row order" discipline
    `analysis_mcp.engine.compare.PeriodComparison` follows. `key` is `()`
    for an un-keyed call, where every row shares the one implicit key."""

    time_period: str
    numerator_value: float
    denominator_value: float
    percent: float
    key: KeyPairs = ()


@dataclass(frozen=True, kw_only=True)
class ShareOfResult:
    """Everything `share_of_tool.py` needs to mint identity, write the
    store, and render the digest.

    Periods, coverage and the periods one side holds alone read as
    `analysis_mcp.engine.quotient_of.QuotientOfResult`'s do
    (`analysis_mcp.engine.two_sided.PeriodPairing`): `matched_periods` are
    the periods some key shares with its denominator group, and each side's
    coverage fraction counts the (key, period) cells it shares over those it
    holds.

    `stat_var` labels every output row; `numerator_stat_var` and
    `denominator_stat_var` are the StatVars each side resolved to.
    """

    frame: Frame
    stat_var: str
    numerator_stat_var: str
    denominator_stat_var: str
    numerator_resolved_unit: UnitCell | None
    denominator_resolved_unit: UnitCell | None
    numerator_facet_selections: tuple[FacetSelection, ...]
    denominator_facet_selections: tuple[FacetSelection, ...]
    warnings: tuple[str, ...]
    numerator_periods: tuple[str, ...]
    denominator_periods: tuple[str, ...]
    matched_periods: tuple[str, ...]
    numerator_coverage_fraction: float
    denominator_coverage_fraction: float
    period_shares: tuple[PeriodShare, ...]
    numerator_only_periods: tuple[KeyPeriods, ...] = ()
    denominator_only_periods: tuple[KeyPeriods, ...] = ()
    exclusions: Exclusions = field(default_factory=Exclusions)


def _exceeding_whole_warnings(period_shares: Sequence[PeriodShare]) -> tuple[str, ...]:
    """A `share_exceeds_whole` warning when any share is above 100 per cent:
    the numerator is then no part of the denominator, as a gross figure over a
    net one can be, or the two count something twice. Empty otherwise."""
    above = [share for share in period_shares if share.percent > 100.0]
    if not above:
        return ()
    largest = max(above, key=lambda share: share.percent)
    return (
        f"share_exceeds_whole: {len(above)} share(s) above 100%, the largest "
        f"{largest.percent:.1f}% in {largest.time_period}; the numerator is not a part of "
        "the denominator",
    )


def _differing_facet_warning(axis: str, numerator_ref: str, denominator_ref: str) -> str:
    """One `cross_side_differing_facet_axes` finding, worded for share_of's
    numerator/denominator vocabulary. It reaches a result only when the two
    sides hold different StatVars, each read under its own selection. Short
    enough to fit `analysis_mcp.disambiguation.MAX_LABEL_CHARS` with refs of
    45 characters each, the digest's cap on one warning line.
    """
    return (
        f"facet {axis!r} differs: numerator {numerator_ref!r}, denominator "
        f"{denominator_ref!r}; two stat_vars, each keeps its own"
    )


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


@contextmanager
def _share_of_refusals() -> Iterator[None]:
    """Re-raise `analysis_mcp.engine.two_sided`'s own exceptions as the
    `share_of` exceptions carrying the same fields."""
    try:
        yield
    except NonFiniteCellError as exc:
        raise ShareOfNonFiniteResultError(time_period=exc.time_period, reason=exc.reason) from exc
    except DenominatorByPartialError as exc:
        raise ShareOfKeyUnknownError(
            column=exc.column,
            numerator_by=exc.numerator_by,
            denominator_by_keys=exc.denominator_by_keys,
        ) from exc


def run_share_of(
    numerator_frame: Frame,
    denominator_frame: Frame,
    *,
    stat_var: str,
    numerator_entity: Mapping[str, str],
    denominator_entity: Mapping[str, str],
    numerator_by: Sequence[str] = (),
    denominator_by: Mapping[str, str] | None = None,
    numerator_pinned_unit: UnitCell | None = None,
    denominator_pinned_unit: UnitCell | None = None,
) -> ShareOfResult:
    """Compute `numerator_frame`'s (narrowed by `numerator_entity`) share of
    `denominator_frame`'s (narrowed by `denominator_entity`) own value, per
    matched (key, period), in per cent, labelling every result row
    `stat_var`. See the module docstring for the keying, grain-match and
    zero-denominator policies; see each raised exception's own docstring for
    its refusal condition. `numerator_by=()` (the default) computes a single
    total, with no key.
    """
    numerator_by = tuple(numerator_by)
    denominator_by = dict(denominator_by or {})

    with _share_of_refusals():
        denominator_columns = denominator_key_columns(
            numerator_by=numerator_by, denominator_by=denominator_by
        )
        numerator = sum_side(
            numerator_frame,
            side="numerator",
            entity=numerator_entity,
            by=numerator_by,
            pinned_unit=numerator_pinned_unit,
        )
        denominator = sum_side(
            denominator_frame,
            side="denominator",
            entity=denominator_entity,
            by=denominator_columns,
            pinned_unit=denominator_pinned_unit,
        )
    numerator_grouped, denominator_grouped = numerator.by_key, denominator.by_key
    numerator_unit, denominator_unit = numerator.unit, denominator.unit
    numerator_facets = numerator.facet_selections
    denominator_facets = denominator.facet_selections

    # An un-keyed call (`by=()`) has exactly one implicit key, `()`,
    # whatever it summed to -- including empty. `two_sided.sum_side`
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
        raise ShareOfGrainMismatchError(numerator_periods=(), denominator_periods=())

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

        facet_mismatch = first_cross_side_facet_mismatch(
            numerator_facets,
            denominator_facets,
            left_stat_vars=(numerator.stat_var,),
            right_stat_vars=(denominator.stat_var,),
        )
        if facet_mismatch is not None:
            axis, numerator_ref, denominator_ref = facet_mismatch
            raise ShareOfFacetMismatchError(
                axis=axis,
                numerator_ref=numerator_ref,
                denominator_ref=denominator_ref,
                stat_var=numerator.stat_var,
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
        # An axis both sides declare to different values reaches here only
        # when the two sides hold different StatVars, each with its own
        # selection (see `first_cross_side_facet_mismatch`).
        cross_side_facet_warnings = (
            *(
                _differing_facet_warning(axis, numerator_ref, denominator_ref)
                for axis, numerator_ref, denominator_ref in cross_side_differing_facet_axes(
                    numerator_facets, denominator_facets
                )
            ),
            *(
                _one_sided_facet_warning(axis, numerator_ref, denominator_ref)
                for axis, numerator_ref, denominator_ref in cross_side_one_sided_facet_axes(
                    numerator_facets, denominator_facets
                )
            ),
        )

    # In per-axis mode (`denominator_columns` non-empty) every numerator key
    # must find its own denominator group, by construction one-to-one
    # (`denominator_columns` is `denominator_by` mapped through
    # `numerator_by`'s own order, so a numerator key tuple and its matching
    # denominator key tuple carry identical values). In pooled mode
    # (`denominator_columns` empty) every numerator key shares the single
    # `()` group.
    pooled = not denominator_columns
    numerator_keys = sorted(numerator_grouped)
    for key in numerator_keys:
        if not pooled and key not in denominator_grouped:
            raise ShareOfKeyUnmatchedError(
                key=tuple(zip(numerator_by, key, strict=True)),
                available=tuple(
                    tuple(zip(numerator_by, other_key, strict=True))
                    for other_key in sorted(denominator_grouped)
                ),
            )
    pairing = pair_periods(
        numerator, denominator, numerator_by=numerator_by, paired_keys=numerator_keys, pooled=pooled
    )

    period_shares: list[PeriodShare] = []
    for key, shared in pairing.shared.items():
        numerator_by_period = numerator_grouped[key]
        denominator_by_period = denominator_grouped[() if pooled else key]
        key_pairs = tuple(zip(numerator_by, key, strict=True))
        # A key sharing no period has no share to report, including a key
        # whose two sides both hold no row (a typo'd entity binding on each
        # side); the call refuses at the first such key rather than return
        # a result missing it.
        if not shared:
            raise ShareOfGrainMismatchError(
                numerator_periods=sorted(numerator_by_period),
                denominator_periods=sorted(denominator_by_period),
                key=key_pairs,
            )
        for period in shared:
            numerator_value = numerator_by_period[period]
            denominator_value = denominator_by_period[period]
            with _share_of_refusals():
                share = divide_cell(
                    numerator_value=numerator_value,
                    denominator_value=denominator_value,
                    time_period=period,
                )
            percent = share * 100.0
            if not math.isfinite(percent):
                raise ShareOfNonFiniteResultError(time_period=period, reason="overflow")
            period_shares.append(
                PeriodShare(
                    time_period=period,
                    numerator_value=numerator_value,
                    denominator_value=denominator_value,
                    percent=percent,
                    key=key_pairs,
                )
            )

    output_frame = frame_keeping_role_types(
        {
            STAT_VAR_COLUMN: repeated(stat_var, len(period_shares)),
            TIME_PERIOD_COLUMN: pa.array(
                [ps.time_period for ps in period_shares], type=pa.string()
            ),
            VALUE_COLUMN: pa.array([ps.percent for ps in period_shares], type=pa.float64()),
            UNIT_COLUMN: unit_cell_array(PERCENT_UNIT, len(period_shares)),
            **{
                column: pa.array([ps.key[index][1] for ps in period_shares], type=pa.string())
                for index, column in enumerate(numerator_by)
            },
        },
        role_keys=numerator_by,
        source=numerator_frame,
    )

    return ShareOfResult(
        frame=output_frame,
        stat_var=stat_var,
        numerator_stat_var=numerator.stat_var,
        denominator_stat_var=denominator.stat_var,
        numerator_resolved_unit=numerator_unit,
        denominator_resolved_unit=denominator_unit,
        numerator_facet_selections=numerator_facets,
        denominator_facet_selections=denominator_facets,
        warnings=tuple(
            dict.fromkeys(
                (
                    *numerator.warnings,
                    *denominator.warnings,
                    *cross_side_facet_warnings,
                    *_exceeding_whole_warnings(period_shares),
                )
            )
        ),
        numerator_periods=pairing.numerator_periods,
        denominator_periods=pairing.denominator_periods,
        matched_periods=pairing.matched_periods,
        numerator_coverage_fraction=pairing.numerator_coverage_fraction,
        denominator_coverage_fraction=pairing.denominator_coverage_fraction,
        period_shares=tuple(period_shares),
        numerator_only_periods=pairing.numerator_only_periods,
        denominator_only_periods=pairing.denominator_only_periods,
        exclusions=record_unmatched_periods(
            combine_exclusions(numerator.exclusions, denominator.exclusions),
            pairing,
            numerator=numerator,
            denominator=denominator,
            numerator_by=numerator_by,
            denominator_columns=denominator_columns,
        ),
    )
