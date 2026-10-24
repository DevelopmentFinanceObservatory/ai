"""A small, committed corpus of every verb's digest-rendering scenarios for
the budget test (`test_digest_budget.py`), mirroring `data_mcp.tests.
digest_corpus`'s own shape: named cases, a size ceiling, and named pairs
proving specific size-regression properties -- scaled down to what nine
digest shapes (one tool per verb, no status discriminator) actually need.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pyarrow as pa
from analysis_mcp.artifact import Exclusion
from analysis_mcp.digest import (
    render_aggregate_digest,
    render_compare_digest,
    render_filter_digest,
    render_growth_digest,
    render_join_digest,
    render_quotient_of_digest,
    render_rank_digest,
    render_share_of_digest,
    render_sum_of_digest,
)
from analysis_mcp.engine.aggregate import AggregateResult, GroupSummary
from analysis_mcp.engine.compare import CompareResult, PeriodComparison
from analysis_mcp.engine.filter import FilterResult, FilterRowSummary
from analysis_mcp.engine.growth import EntityGrowth, GrowthResult
from analysis_mcp.engine.join import JoinResult, MatchedKey, MatchedKeys
from analysis_mcp.engine.quotient_of import PeriodQuotient, QuotientOfResult, ZeroDenominatorCell
from analysis_mcp.engine.rank import EntityRank, RankResult
from analysis_mcp.engine.share_of import PeriodShare, ShareOfResult
from analysis_mcp.engine.sum_of import PeriodSum, SumOfResult
from analysis_mcp.execution_identity import ExecutionIdentity
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart

_EI = ExecutionIdentity(
    engine_version="0.1.0", frame_grammar_version="1.0.0", reduction_policy_version="0.1.0"
)

# The single ceiling every corpus case's digest must stay under. Generous
# relative to the committed baseline's actual measurements -- this catches a
# gross regression (a frame leak, an uncapped loop), not a few-character
# wording change.
CEILING = 4000


@dataclass(frozen=True, kw_only=True)
class CorpusCase:
    id: str
    kwargs: dict
    render: Callable[..., str] = field(default=render_aggregate_digest)


# aggregate ---------------------------------------------------------------------


def _summary(
    key: str, value: float, *, year: str = "2021", value_period: str | None = None
) -> GroupSummary:
    return GroupSummary(
        group_key=(("reporter", key),),
        value=value,
        row_count=1,
        time_period_min=year,
        time_period_max=year,
        value_period=value_period,
    )


def _result(group_summaries: tuple[GroupSummary, ...], **overrides: object) -> AggregateResult:
    defaults: dict[str, object] = {
        "frame": None,
        "resolved_unit": "dcid:USDollar",
        "facet_selections": (),
        "warnings": (),
        "realised_coverage_fraction": 1.0,
        "skipped_cell_count": 0,
        "expected_cell_count": len(group_summaries) or 1,
        "present_cell_count": len(group_summaries),
        "collapsed_periods": ("2021",),
        "period_label": "2021",
    }
    defaults.update(overrides)
    return AggregateResult(group_summaries=group_summaries, **defaults)


def _digest_kwargs(
    result: AggregateResult, *, group_by: tuple[str, ...] = ("reporter",), aggregation: str = "sum"
) -> dict:
    return {
        "group_by": group_by,
        "aggregation": aggregation,
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _groups(count: int, *, offset: int = 0) -> tuple[GroupSummary, ...]:
    return tuple(_summary(f"dcid:{i:03d}", float(i)) for i in range(offset, offset + count))


CASES: tuple[CorpusCase, ...] = (
    CorpusCase(id="single_group", kwargs=_digest_kwargs(_result(_groups(1)))),
    CorpusCase(id="five_groups", kwargs=_digest_kwargs(_result(_groups(5)))),
    CorpusCase(
        id="with_warnings",
        kwargs=_digest_kwargs(
            _result(_groups(1), warnings=("facet 'provenance' auto-selected 'dcid:X'.",))
        ),
    ),
    CorpusCase(
        id="sparse_coverage",
        kwargs=_digest_kwargs(
            _result(
                _groups(9),
                realised_coverage_fraction=0.75,
                present_cell_count=9,
                expected_cell_count=12,
                skipped_cell_count=3,
            )
        ),
    ),
    # `offset=100`: every candidate value in both cases stays a 3-digit
    # number, whichever 20 `_capped` (`analysis_mcp.digest`) actually picks
    # by magnitude -- keeps the two cases' digest sizes comparable purely by
    # entry *count*, not by which digit-widths the magnitude selection
    # happened to keep (see `test_content_past_the_group_cap_costs_at_most_
    # the_headers_own_digit_growth`).
    CorpusCase(id="capped_25_groups", kwargs=_digest_kwargs(_result(_groups(25, offset=100)))),
    CorpusCase(id="capped_30_groups", kwargs=_digest_kwargs(_result(_groups(30, offset=100)))),
    CorpusCase(id="empty_group_by", kwargs=_digest_kwargs(_result(_groups(1)), group_by=())),
    # `offset=0` (the default), no digit-width trick: as `count` grows, the
    # top-20-by-magnitude entries `_capped` actually picks carry more
    # digits (5..24 at count=25, 980..999 at count=1000, 99980..99999 at
    # count=100000), so these three exercise the real growth mode
    # `capped_25_groups`/`capped_30_groups` above are deliberately built to
    # avoid -- see `test_digest_budget.py`'s own
    # `test_content_past_the_group_cap_grows_by_a_bounded_amount`.
    CorpusCase(id="wide_band_25_groups", kwargs=_digest_kwargs(_result(_groups(25)))),
    CorpusCase(id="wide_band_1000_groups", kwargs=_digest_kwargs(_result(_groups(1000)))),
    CorpusCase(id="wide_band_100000_groups", kwargs=_digest_kwargs(_result(_groups(100_000)))),
    # `max`/`min`: every group line carries a value_period alongside the
    # value, the one shape the other cases above never exercise.
    CorpusCase(
        id="max_min_groups",
        kwargs=_digest_kwargs(
            _result((_summary("dcid:001", 1.0, value_period="2019"),), period_label=None),
            aggregation="max",
        ),
    ),
)


# growth --------------------------------------------------------------------------


def _entity_growth(key: str, rate: float) -> EntityGrowth:
    return EntityGrowth(
        entity_key=(("reporter", key),),
        value_start=100.0,
        value_end=100.0 * (1 + rate),
        growth_rate=rate,
        realised_coverage_fraction=1.0,
    )


def _growth_result(entity_growths: tuple[EntityGrowth, ...], **overrides: object) -> GrowthResult:
    defaults: dict[str, object] = {
        "frame": None,
        "resolved_unit": "dcid:USDollar",
        "facet_selections": (),
        "warnings": (),
        "window_start": "2020",
        "window_end": "2023",
    }
    defaults.update(overrides)
    return GrowthResult(entity_growths=entity_growths, **defaults)


def _growth_digest_kwargs(result: GrowthResult) -> dict:
    return {
        "window_start": result.window_start,
        "window_end": result.window_end,
        "entity_by": ("reporter",),
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _entity_growths(count: int) -> tuple[EntityGrowth, ...]:
    return tuple(_entity_growth(f"dcid:{i:03d}", 0.01 * i) for i in range(count))


GROWTH_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="growth_single_entity",
        kwargs=_growth_digest_kwargs(_growth_result(_entity_growths(1))),
        render=render_growth_digest,
    ),
    CorpusCase(
        id="growth_capped_25_entities",
        kwargs=_growth_digest_kwargs(_growth_result(_entity_growths(25))),
        render=render_growth_digest,
    ),
)


# compare -------------------------------------------------------------------------


def _period_comparison(period: str, left: float, right: float) -> PeriodComparison:
    return PeriodComparison(
        time_period=period,
        left_value=left,
        right_value=right,
        delta=right - left,
    )


def _compare_result(
    period_comparisons: tuple[PeriodComparison, ...], **overrides: object
) -> CompareResult:
    defaults: dict[str, object] = {
        "frame": None,
        "stat_var": "dcid:ops_" + "a" * 32,
        "left_stat_vars": ("dcid:DebtService",),
        "right_stat_vars": ("dcid:Disbursements",),
        "left_entities": {},
        "right_entities": {},
        "left_resolved_unit": "dcid:USDollar",
        "right_resolved_unit": "dcid:USDollar",
        "left_facet_selections": (),
        "right_facet_selections": (),
        "warnings": (),
        "left_periods": tuple(pc.time_period for pc in period_comparisons),
        "right_periods": tuple(pc.time_period for pc in period_comparisons),
        "intersected_periods": tuple(pc.time_period for pc in period_comparisons),
        "left_coverage_fraction": 1.0,
        "right_coverage_fraction": 1.0,
    }
    defaults.update(overrides)
    return CompareResult(period_comparisons=period_comparisons, **defaults)


def _compare_digest_kwargs(result: CompareResult) -> dict:
    return {
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _many_stat_vars(side: str) -> tuple[str, ...]:
    """2,000 StatVar refs for one side, a family summed by its retrieval at
    the scale of CRS's purpose codes."""
    return tuple(f"dcid:{side}_stat_var_{index:04d}" for index in range(2000))


def _period_comparisons(count: int) -> tuple[PeriodComparison, ...]:
    return tuple(_period_comparison(f"20{i:02d}", float(i), float(i) * 2) for i in range(count))


COMPARE_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="compare_single_period",
        kwargs=_compare_digest_kwargs(_compare_result(_period_comparisons(1))),
        render=render_compare_digest,
    ),
    CorpusCase(
        id="compare_capped_25_periods",
        kwargs=_compare_digest_kwargs(_compare_result(_period_comparisons(25))),
        render=render_compare_digest,
    ),
    CorpusCase(
        id="compare_2000_stat_vars_per_side",
        kwargs=_compare_digest_kwargs(
            _compare_result(
                _period_comparisons(1),
                left_stat_vars=_many_stat_vars("left"),
                right_stat_vars=_many_stat_vars("right"),
            )
        ),
        render=render_compare_digest,
    ),
)


# sum_of --------------------------------------------------------------------------


def _sum_of_digest_kwargs(
    count: int,
    *,
    left_only: int = 0,
    stat_vars: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
) -> dict:
    """`count` summed periods; the left side also offers `left_only` later
    periods the right lacks, which the coverage line lists. `stat_vars`
    replaces each side's one StatVar."""
    periods = tuple(f"20{i:02d}" for i in range(count))
    left_periods = periods + tuple(f"21{i:02d}" for i in range(left_only))
    left_stat_vars, right_stat_vars = stat_vars or (
        ("dcid:test_stat_var_left",),
        ("dcid:test_stat_var_right",),
    )
    fields: dict[str, object] = {
        "frame": None,
        "stat_var": "dcid:ops_" + "a" * 32,
        "left_stat_vars": left_stat_vars,
        "right_stat_vars": right_stat_vars,
        "unit": "dcid:USDollar",
        "facet_selections": (),
        "left_facet_selections": (),
        "right_facet_selections": (),
        "warnings": (),
        "left_periods": left_periods,
        "right_periods": periods,
        "intersected_periods": periods,
        "left_coverage_fraction": count / len(left_periods),
        "right_coverage_fraction": 1.0,
    }
    sums = tuple(
        PeriodSum(time_period=period, left_value=float(i), right_value=2.0 * i, value=3.0 * i)
        for i, period in enumerate(periods)
    )
    return {
        "result": SumOfResult(period_sums=sums, **fields),
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


SUM_OF_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="sum_of_single_period",
        kwargs=_sum_of_digest_kwargs(1),
        render=render_sum_of_digest,
    ),
    CorpusCase(
        id="sum_of_capped_25_periods_25_left_only",
        kwargs=_sum_of_digest_kwargs(25, left_only=25),
        render=render_sum_of_digest,
    ),
    CorpusCase(
        id="sum_of_2000_stat_vars_per_side",
        kwargs=_sum_of_digest_kwargs(
            1, stat_vars=(_many_stat_vars("left"), _many_stat_vars("right"))
        ),
        render=render_sum_of_digest,
    ),
)


# quotient_of -----------------------------------------------------------------


def quotient_of_result(
    period_quotients: tuple[PeriodQuotient, ...],
    *,
    matched_periods: tuple[str, ...] | None = None,
    numerator_periods: tuple[str, ...] | None = None,
    denominator_periods: tuple[str, ...] | None = None,
    **overrides: object,
) -> QuotientOfResult:
    matched = matched_periods or tuple(sorted({entry.time_period for entry in period_quotients}))
    numerator_periods = matched if numerator_periods is None else numerator_periods
    denominator_periods = matched if denominator_periods is None else denominator_periods
    defaults: dict[str, object] = {
        "frame": None,
        "stat_var": "dcid:ops_" + "a" * 32,
        "numerator_stat_var": "dcid:test_stat_var_numerator",
        "denominator_stat_var": "dcid:Count_Person",
        "numerator_resolved_unit": "dcid:USDollar",
        "denominator_resolved_unit": None,
        "unit": ComposedUnit(
            numerator="dcid:USDollar", denominator=StatVarPart("dcid:Count_Person")
        ),
        "facet_selections": (),
        "numerator_facet_selections": (),
        "denominator_facet_selections": (),
        "warnings": (),
        "numerator_periods": numerator_periods,
        "denominator_periods": denominator_periods,
        "matched_periods": matched,
        "numerator_coverage_fraction": len(matched) / len(numerator_periods),
        "denominator_coverage_fraction": len(matched) / len(denominator_periods),
    }
    defaults.update(overrides)
    return QuotientOfResult(period_quotients=period_quotients, **defaults)


def _quotient_of_digest_kwargs(result: QuotientOfResult) -> dict:
    return {
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _period_quotients(count: int, *, key: str | None = None) -> tuple[PeriodQuotient, ...]:
    return tuple(
        PeriodQuotient(
            time_period=f"20{i:02d}",
            numerator_value=float(i) + 1.0,
            denominator_value=(float(i) + 1.0) * 50.0,
            quotient=0.02,
            key=(("recipient", key),) if key else (),
        )
        for i in range(count)
    )


def _keyed_period_quotients(count: int) -> tuple[PeriodQuotient, ...]:
    return tuple(
        PeriodQuotient(
            time_period="2020",
            numerator_value=float(i) + 1.0,
            denominator_value=50.0,
            quotient=(float(i) + 1.0) / 50.0,
            key=(("recipient", f"dcid:R{i:03d}"),),
        )
        for i in range(count)
    )


QUOTIENT_OF_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="quotient_of_unkeyed_periods_left_out",
        kwargs=_quotient_of_digest_kwargs(
            quotient_of_result(
                _period_quotients(3),
                numerator_periods=("2000", "2001", "2002", "2003", "2004"),
                denominator_periods=("1999", "2000", "2001", "2002"),
            )
        ),
        render=render_quotient_of_digest,
    ),
    CorpusCase(
        id="quotient_of_keyed_dropped_key_and_zero_denominator",
        kwargs=_quotient_of_digest_kwargs(
            quotient_of_result(
                _period_quotients(2, key="dcid:KEN"),
                unmatched_keys=((("recipient", "dcid:GTM"),),),
                keys_with_zero_denominator=((("recipient", "dcid:TZA"),),),
                zero_denominator_cells=(
                    ZeroDenominatorCell(
                        time_period="2002",
                        numerator_value=4.0,
                        key=(("recipient", "dcid:KEN"),),
                    ),
                    ZeroDenominatorCell(
                        time_period="2002",
                        numerator_value=9.0,
                        key=(("recipient", "dcid:TZA"),),
                    ),
                ),
                matched_periods=("2000", "2001", "2002"),
            )
        ),
        render=render_quotient_of_digest,
    ),
    CorpusCase(
        id="quotient_of_capped_25_keys",
        kwargs=_quotient_of_digest_kwargs(quotient_of_result(_keyed_period_quotients(25))),
        render=render_quotient_of_digest,
    ),
)


# rank --------------------------------------------------------------------------


def _entity_rank(key: str, value: float, rank: int) -> EntityRank:
    return EntityRank(
        entity_key=(("reporter", key),), value=value, rank=rank, row_count=1, coverage_fraction=1.0
    )


def _rank_result(entity_ranks: tuple[EntityRank, ...], **overrides: object) -> RankResult:
    defaults: dict[str, object] = {
        "frame": None,
        "resolved_unit": "dcid:USDollar",
        "facet_selections": (),
        "warnings": (),
        "aggregation": "sum",
        "window_start": "2021",
        "window_end": "2023",
        "periods_in_window": ("2021", "2022", "2023"),
        "expected_cell_count": len(entity_ranks) * 3 or 1,
        "present_cell_count": len(entity_ranks) * 3,
        "overall_coverage_fraction": 1.0,
    }
    defaults.update(overrides)
    return RankResult(entity_ranks=entity_ranks, **defaults)


def _rank_digest_kwargs(result: RankResult) -> dict:
    return {
        "rank_by": ("reporter",),
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _entity_ranks(count: int) -> tuple[EntityRank, ...]:
    return tuple(_entity_rank(f"dcid:{i:03d}", float(count - i), i + 1) for i in range(count))


RANK_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="rank_single_entity",
        kwargs=_rank_digest_kwargs(_rank_result(_entity_ranks(1))),
        render=render_rank_digest,
    ),
    CorpusCase(
        id="rank_capped_25_entities",
        kwargs=_rank_digest_kwargs(_rank_result(_entity_ranks(25))),
        render=render_rank_digest,
    ),
    CorpusCase(
        id="rank_limit_10_of_25_entities",
        kwargs={
            **_rank_digest_kwargs(_rank_result(_entity_ranks(25), limit=10)),
            "exclusions": [
                Exclusion(
                    reason="beyond_limit",
                    rows=45,
                    entities={"reporter": tuple(f"dcid:{i:03d}" for i in range(10, 25))},
                )
            ],
        },
        render=render_rank_digest,
    ),
)


# share_of --------------------------------------------------------------------


def _period_share(period: str, numerator: float, denominator: float) -> PeriodShare:
    return PeriodShare(
        time_period=period,
        numerator_value=numerator,
        denominator_value=denominator,
        percent=numerator / denominator * 100.0,
    )


def share_of_result(period_shares: tuple[PeriodShare, ...], **overrides: object) -> ShareOfResult:
    defaults: dict[str, object] = {
        "frame": None,
        "stat_var": "dcid:ops_" + "a" * 32,
        "numerator_stat_var": "dcid:test_stat_var_numerator",
        "denominator_stat_var": "dcid:test_stat_var_denominator",
        "numerator_resolved_unit": "dcid:USDollar",
        "denominator_resolved_unit": "dcid:USDollar",
        "numerator_facet_selections": (),
        "denominator_facet_selections": (),
        "warnings": (),
        "numerator_periods": tuple(ps.time_period for ps in period_shares),
        "denominator_periods": tuple(ps.time_period for ps in period_shares),
        "matched_periods": tuple(ps.time_period for ps in period_shares),
        "numerator_coverage_fraction": 1.0,
        "denominator_coverage_fraction": 1.0,
    }
    defaults.update(overrides)
    return ShareOfResult(period_shares=period_shares, **defaults)


def _share_of_digest_kwargs(result: ShareOfResult) -> dict:
    return {
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _period_shares(count: int) -> tuple[PeriodShare, ...]:
    return tuple(
        _period_share(f"20{i:02d}", float(i) + 1.0, (float(i) + 1.0) * 100.0) for i in range(count)
    )


SHARE_OF_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="share_of_single_period",
        kwargs=_share_of_digest_kwargs(share_of_result(_period_shares(1))),
        render=render_share_of_digest,
    ),
    CorpusCase(
        id="share_of_capped_25_periods",
        kwargs=_share_of_digest_kwargs(share_of_result(_period_shares(25))),
        render=render_share_of_digest,
    ),
)


# join --------------------------------------------------------------------------


def _matched_key(key: str, value: float) -> MatchedKey:
    return MatchedKey(
        key=(("reporter", key),), time_period="2021", left_value=value, right_value=value * 10.0
    )


def matched_keys_of(rows: Sequence[MatchedKey]) -> MatchedKeys:
    """`rows` as the columnar `MatchedKeys` a `JoinResult` carries. Every
    row names the same key columns, in the same order."""
    columns = [column for column, _ in rows[0].key] if rows else []
    return MatchedKeys(
        role_cells={
            column: pa.array([dict(row.key)[column] for row in rows], type=pa.string())
            for column in columns
        },
        time_periods=pa.array([row.time_period for row in rows], type=pa.string()),
        left_values=pa.array([row.left_value for row in rows], type=pa.float64()),
        right_values=pa.array([row.right_value for row in rows], type=pa.float64()),
    )


def _join_result(matched_keys: tuple[MatchedKey, ...], **overrides: object) -> JoinResult:
    defaults: dict[str, object] = {
        "frame": None,
        "left_resolved_unit": "dcid:USDollar",
        "right_resolved_unit": "dcid:USDollar",
        "left_facet_selections": (),
        "right_facet_selections": (),
        "warnings": (),
        "join_keys": ("reporter",),
        "right_key_columns": ("reporter",),
        "left_dropped_key_count": 0,
        "right_dropped_key_count": 0,
    }
    defaults.update(overrides)
    return JoinResult(matched_keys=matched_keys_of(matched_keys), **defaults)


def _join_digest_kwargs(result: JoinResult) -> dict:
    return {
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _matched_keys(count: int) -> tuple[MatchedKey, ...]:
    return tuple(_matched_key(f"dcid:{i:03d}", float(i)) for i in range(count))


JOIN_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="join_single_key",
        kwargs=_join_digest_kwargs(_join_result(_matched_keys(1))),
        render=render_join_digest,
    ),
    CorpusCase(
        id="join_capped_25_keys",
        kwargs=_join_digest_kwargs(_join_result(_matched_keys(25))),
        render=render_join_digest,
    ),
    CorpusCase(
        id="join_with_dropped_keys",
        kwargs=_join_digest_kwargs(
            _join_result(
                _matched_keys(1),
                left_dropped_key_count=3,
                right_dropped_key_count=2,
                warnings=("3 left-side and 2 right-side key(s) had no match on the other side.",),
            )
        ),
        render=render_join_digest,
    ),
)


# filter --------------------------------------------------------------------------


def _filter_row_summary(key: str, value: float) -> FilterRowSummary:
    return FilterRowSummary(
        row_key=(("stat_var", "dcid:X"), ("reporter", key), ("time_period", "2021")),
        value=value,
        unit="dcid:USDollar",
    )


def _filter_result(
    kept_row_summaries: tuple[FilterRowSummary, ...], **overrides: object
) -> FilterResult:
    defaults: dict[str, object] = {
        "frame": None,
        "total_row_count": len(kept_row_summaries) + 1,
        "kept_row_count": len(kept_row_summaries),
        "dropped_row_count": 1,
    }
    defaults.update(overrides)
    return FilterResult(kept_row_summaries=kept_row_summaries, **defaults)


def _filter_digest_kwargs(result: FilterResult) -> dict:
    return {
        "predicate_count": 1,
        "result": result,
        "op_spec_hash": "ops_" + "a" * 32,
        "content_version": "ocv_" + "b" * 32,
        "execution_identity": _EI,
    }


def _filter_row_summaries(count: int) -> tuple[FilterRowSummary, ...]:
    return tuple(_filter_row_summary(f"dcid:{i:03d}", float(i)) for i in range(count))


FILTER_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="filter_single_row",
        kwargs=_filter_digest_kwargs(_filter_result(_filter_row_summaries(1))),
        render=render_filter_digest,
    ),
    CorpusCase(
        id="filter_capped_25_rows",
        kwargs=_filter_digest_kwargs(_filter_result(_filter_row_summaries(25))),
        render=render_filter_digest,
    ),
    CorpusCase(
        id="filter_empty_result",
        kwargs=_filter_digest_kwargs(
            _filter_result((), total_row_count=15, kept_row_count=0, dropped_row_count=15)
        ),
        render=render_filter_digest,
    ),
)


# Frame-pair regression: identical summary tuples, wildly different `frame`
# sizes behind them -- the digest must render identically either way, since
# no `render_*_digest` function ever reads its result's own `frame` field.
@dataclass(frozen=True, kw_only=True)
class FramePair:
    lean_id: str
    fat_id: str


def _lean_frame() -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0}],
    )


def _fat_frame() -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": f"dcid:R{i:04d}",
                "time_period": "2020",
                "value": float(i),
            }
            for i in range(500)
        ],
    )


FRAME_PAIR_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        id="frame_pair_lean", kwargs=_digest_kwargs(_result(_groups(1), frame=_lean_frame()))
    ),
    CorpusCase(id="frame_pair_fat", kwargs=_digest_kwargs(_result(_groups(1), frame=_fat_frame()))),
)

FRAME_PAIRS: tuple[FramePair, ...] = (
    FramePair(lean_id="frame_pair_lean", fat_id="frame_pair_fat"),
)
