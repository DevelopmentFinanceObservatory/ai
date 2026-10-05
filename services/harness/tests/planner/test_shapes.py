"""Tests for `planner.shapes`: the frame-shape algebra's transforms and `rejects` rules."""

from __future__ import annotations

import pytest
from common.place_type import PLACE_LEVELS
from planner.records import (
    DenominatorScope,
    GrowthChange,
    Period,
    PeriodThreshold,
    PlaceType,
    RankAmong,
    Reduction,
)
from planner.shapes import (
    SEVERAL,
    FrameShape,
    RoleDimension,
    ShapeVerb,
    TimeLabel,
    ValueKind,
    YearFilter,
    aggregated,
    compared,
    filtered,
    grown,
    joined,
    ranked,
    rejects,
    retrieved,
    shared,
    year_filter,
)

_PERIOD = Period(start=2015, end=2020)


def _role(
    role: str | None = "donor", requirement_id: str = "req_a", narrowed: bool = False
) -> RoleDimension:
    return RoleDimension(role=role, requirement_id=requirement_id, narrowed=narrowed)


def _shape(
    *,
    roles: tuple[RoleDimension, ...] = (),
    measures: int | None = 1,
    value: ValueKind = ValueKind.MEASURED,
    time: TimeLabel = TimeLabel.OBSERVATION,
    period: Period | None = _PERIOD,
    one_row_per_entity: bool = False,
    cut_per_period: bool = False,
    limited: bool = False,
    roots: tuple[str, ...] = ("req_a",),
    unsummable_axes: tuple[str, ...] | None = None,
) -> FrameShape:
    return FrameShape(
        roles=roles,
        measures=measures,
        value=value,
        time=time,
        period=period,
        one_row_per_entity=one_row_per_entity,
        cut_per_period=cut_per_period,
        limited=limited,
        roots=roots,
        unsummable_axes=unsummable_axes,
    )


# --- retrieved ---------------------------------------------------------


def test_retrieved_stamps_every_role_with_the_requirement_and_leaves_it_unnarrowed():
    # A stale requirement id and a stale narrowed=True prove `retrieved` re-stamps
    # both rather than trusting what the caller happened to pass in.
    stale = RoleDimension(role="donor", requirement_id="whatever-stale", narrowed=True)
    shape = retrieved(requirement_id="req_a", roles=(stale,), period=_PERIOD, measures=1)
    assert shape.roles == (_role(role="donor", requirement_id="req_a", narrowed=False),)
    assert shape.measures == 1
    assert shape.value is ValueKind.MEASURED
    assert shape.time is TimeLabel.OBSERVATION
    assert shape.period == _PERIOD
    assert shape.one_row_per_entity is False
    assert shape.roots == ("req_a",)


# --- aggregated ----------------------------------------------------------


def test_aggregated_by_entity_and_period_keeps_roles_and_stays_observation():
    shape = _shape(roles=(_role(),))
    out = aggregated(shape, reduction=Reduction.SUM, by_entity=True, by_period=True)
    assert out.roles == shape.roles
    assert out.measures == shape.measures
    assert out.value == shape.value
    assert out.time is TimeLabel.OBSERVATION
    assert out.period == shape.period
    assert out.one_row_per_entity is False
    assert out.roots == shape.roots


@pytest.mark.parametrize(
    ("reduction", "by_entity", "shape", "time"),
    [
        pytest.param(
            Reduction.SUM, False, _shape(roles=(_role(),)), TimeLabel.SPAN, id="sum-pooled"
        ),
        pytest.param(
            Reduction.MEAN, True, _shape(roles=(_role(),)), TimeLabel.SPAN, id="mean-by-entity"
        ),
        pytest.param(
            Reduction.MAX, True, _shape(roles=(_role(),)), TimeLabel.PICKED, id="max-by-entity"
        ),
        pytest.param(
            Reduction.MIN, False, _shape(roles=(_role(),)), TimeLabel.PICKED, id="min-pooled"
        ),
        pytest.param(
            Reduction.MAX,
            True,
            _shape(roles=(_role(),), time=TimeLabel.SPAN),
            TimeLabel.SPAN,
            id="max-over-a-span-keeps-the-span",
        ),
        pytest.param(
            Reduction.MAX,
            True,
            _shape(roles=(_role(),), period=Period(start=2020, end=2020)),
            TimeLabel.OBSERVATION,
            id="max-over-one-year-keeps-that-year",
        ),
    ],
)
def test_aggregated_without_period_collapses_a_sum_or_mean_and_picks_an_extreme_s_year(
    reduction: Reduction, by_entity: bool, shape: FrameShape, time: TimeLabel
):
    # analysis-mcp labels a max or min row with the period its value came
    # from: one year per entity over a series, the span itself over a span.
    out = aggregated(shape, reduction=reduction, by_entity=by_entity, by_period=False)
    assert out.time is time
    assert out.period == shape.period
    assert out.one_row_per_entity is True


# --- grown ---------------------------------------------------------------


def test_grown_relative_reads_a_dimensionless_ratio_labelled_the_window_span():
    shape = _shape(roles=(_role(),))
    out = grown(shape, change=GrowthChange.RELATIVE, window=Period(start=2016, end=2020))
    assert out.roles == shape.roles
    assert out.measures == shape.measures
    assert out.value is ValueKind.RATIO
    assert out.time is TimeLabel.SPAN
    assert out.period == Period(start=2016, end=2020)
    assert out.one_row_per_entity is True
    assert out.roots == shape.roots


def test_grown_absolute_reads_a_difference_in_the_inputs_own_unit():
    shape = _shape(roles=(_role(),))
    out = grown(shape, change=GrowthChange.ABSOLUTE, window=Period(start=2016, end=2020))
    assert out.value is ValueKind.DIFFERENCE
    assert out.period == Period(start=2016, end=2020)
    assert out.one_row_per_entity is True


# --- ranked ----------------------------------------------------------------


def test_ranked_collapse_mode_carries_the_measured_value_as_a_span():
    shape = _shape(roles=(_role(role="recipient"),))
    out = ranked(shape, axis=None, window=None, per_period=False, limit=None)
    assert out.roles == (_role(role="recipient"),)
    assert out.measures == shape.measures
    assert out.value is ValueKind.MEASURED
    assert out.time is TimeLabel.SPAN
    assert out.period == shape.period
    assert out.one_row_per_entity is True


def test_ranked_per_period_mode_carries_the_rank_position_as_an_ordinal_observation():
    role = _role(role="recipient", narrowed=True)
    shape = _shape(roles=(role,))
    out = ranked(shape, axis="recipient", window=None, per_period=True, limit=None)
    # The axis role's own narrowed flag travels forward unchanged.
    assert out.roles == (role,)
    assert out.measures == 1
    assert out.value is ValueKind.ORDINAL
    assert out.time is TimeLabel.OBSERVATION
    assert out.one_row_per_entity is False


def test_ranked_uses_the_lone_role_when_axis_is_none_and_carries_the_ranks_own_window():
    shape = _shape(roles=(_role(role="recipient"),))
    window = Period(start=2018, end=2020)
    out = ranked(shape, axis=None, window=window, per_period=False, limit=None)
    assert out.roles == (_role(role="recipient"),)
    assert out.period == window


def test_ranked_over_an_axis_naming_no_role_carries_no_role_forward():
    # A defensive case: `rejects` is what turns an unknown axis into a
    # refusal before a transform ever runs against it. The transform itself
    # degrades to no role rather than raising.
    shape = _shape(roles=(_role(role="recipient"),))
    out = ranked(shape, axis="creditor", window=None, per_period=False, limit=None)
    assert out.roles == ()


# --- filtered ----------------------------------------------------------------


def test_filtered_marks_the_matched_role_narrowed_and_keeps_everything_else():
    role = _role(role="donor", narrowed=False)
    other = _role(role="sector", narrowed=False)
    shape = _shape(roles=(role, other), value=ValueKind.RATIO, time=TimeLabel.SPAN)
    narrowed_period = Period(start=2018, end=2018)
    out = filtered(shape, narrowed_roles=("donor",), period=narrowed_period, pins_one_year=True)
    assert out.roles == (_role(role="donor", narrowed=True), other)
    assert out.period == narrowed_period
    assert out.one_row_per_entity is True
    assert out.value is shape.value
    assert out.time is shape.time
    assert out.measures == shape.measures


def test_filtered_with_no_narrowing_keeps_the_period_the_caller_passed_back():
    shape = _shape(roles=(_role(),), one_row_per_entity=True)
    out = filtered(shape, narrowed_roles=(), period=shape.period, pins_one_year=False)
    assert out.roles == shape.roles
    assert out.period == shape.period
    assert out.one_row_per_entity is True


def test_filtered_with_an_unqualified_predicate_narrows_every_role():
    # An `EntityIn` with no `role` stated -- the common case, since a
    # requirement usually names its lone entity with no role phrase --
    # carries `None` in `narrowed_roles`. The primitives here cannot tell
    # which single dimension it resolved against, so it narrows every role
    # the input carries rather than none of them.
    donor = _role(role="donor", narrowed=False)
    sector = _role(role="sector", requirement_id="req_b", narrowed=False)
    shape = _shape(roles=(donor, sector))
    out = filtered(shape, narrowed_roles=(None,), period=shape.period, pins_one_year=False)
    assert out.roles == (
        _role(role="donor", narrowed=True),
        _role(role="sector", requirement_id="req_b", narrowed=True),
    )


def test_filtered_with_an_unqualified_predicate_alongside_a_named_one_still_narrows_both():
    donor = _role(role="donor", narrowed=False)
    sector = _role(role="sector", requirement_id="req_b", narrowed=False)
    shape = _shape(roles=(donor, sector))
    out = filtered(shape, narrowed_roles=("sector", None), period=shape.period, pins_one_year=False)
    assert all(role.narrowed for role in out.roles)


# --- shared ------------------------------------------------------------------


@pytest.mark.parametrize(
    "one_row_per_entity",
    [
        pytest.param(False, id="a-series-per-entity"),
        pytest.param(True, id="one-total-per-entity"),
    ],
)
def test_shared_keys_the_output_on_the_numerators_own_axis_role_alone(one_row_per_entity: bool):
    # run_share_of writes one ratio per numerator key per period, so a share
    # of each entity's total holds one row per entity, as its numerator does.
    numerator = _shape(
        roles=(_role(role="donor", requirement_id="req_num"),),
        period=Period(start=2015, end=2020),
        one_row_per_entity=one_row_per_entity,
        roots=("req_num",),
    )
    denominator = _shape(
        roles=(_role(role="donor", requirement_id="req_den"),),
        period=Period(start=2016, end=2022),
        roots=("req_den",),
    )
    out = shared(numerator, denominator, axis="donor", scope=DenominatorScope.PER_AXIS)
    assert out.roles == (_role(role="donor", requirement_id="req_num"),)
    assert out.measures == 1
    assert out.value is ValueKind.RATIO
    assert out.time is TimeLabel.OBSERVATION
    assert out.period == Period(start=2016, end=2020)
    assert out.one_row_per_entity is one_row_per_entity
    assert out.roots == ("req_den", "req_num")


def test_shared_unkeyed_carries_no_role():
    numerator = _shape(roles=(_role(),))
    denominator = _shape(roles=())
    out = shared(numerator, denominator, axis=None, scope=None)
    assert out.roles == ()


def test_shared_period_is_unbounded_when_either_side_is():
    numerator = _shape(roles=(_role(),), period=None)
    denominator = _shape(roles=())
    out = shared(numerator, denominator, axis=None, scope=None)
    assert out.period is None


# --- compared ------------------------------------------------------------------


def test_compared_shape_for_completeness():
    left = _shape(roots=("req_a",), period=Period(start=2015, end=2020))
    right = _shape(roots=("req_b",), period=Period(start=2016, end=2022))
    out = compared(left, right)
    assert out.roles == ()
    assert out.measures == 1
    assert out.value is ValueKind.DIFFERENCE
    assert out.time is TimeLabel.OBSERVATION
    assert out.period == Period(start=2016, end=2020)
    assert out.roots == ("req_a", "req_b")


# --- joined ------------------------------------------------------------------


def test_joined_keeps_only_the_shared_role_dimensions_from_the_left_side():
    left = _shape(
        roles=(
            _role(role="donor", requirement_id="req_a"),
            _role(role="sector", requirement_id="req_a"),
        ),
        roots=("req_a",),
        period=Period(start=2015, end=2020),
    )
    right = _shape(
        roles=(_role(role="donor", requirement_id="req_b"),),
        roots=("req_b",),
        period=Period(start=2016, end=2022),
    )
    out = joined(left, right)
    assert out.roles == (_role(role="donor", requirement_id="req_a"),)
    assert out.measures == 2
    assert out.value is ValueKind.MEASURED
    assert out.time is TimeLabel.OBSERVATION
    assert out.roots == ("req_a", "req_b")


# --- the narrow-lineage invariant ---------------------------------------------


def test_every_non_terminal_transforms_output_roles_name_one_requirement():
    """Every role dimension a non-terminal verb's output carries names one requirement.

    `shared` is the case that matters: its output could plausibly mix the
    numerator's and the denominator's requirements. The output names only
    the numerator's, the narrow-lineage rule that keeps every non-terminal
    verb's role dimensions tied to one requirement.
    """
    numerator = _shape(roles=(_role(role="donor", requirement_id="req_num"),), roots=("req_num",))
    denominator = _shape(roles=(_role(role="donor", requirement_id="req_den"),), roots=("req_den",))
    share = shared(numerator, denominator, axis="donor", scope=DenominatorScope.PER_AXIS)
    assert {role.requirement_id for role in share.roles} == {"req_num"}

    rank = ranked(share, axis="donor", window=None, per_period=False, limit=None)
    assert {role.requirement_id for role in rank.roles} == {"req_num"}

    grow = grown(rank, change=GrowthChange.ABSOLUTE, window=Period(start=2019, end=2020))
    assert {role.requirement_id for role in grow.roles} == {"req_num"}


# --- rejects: base rules -------------------------------------------------------


def test_rejects_rank_with_no_roles():
    shape = _shape(roles=())
    reason = rejects(ShapeVerb.RANK, shape, axis=None)
    assert reason is not None
    assert "role dimension" in reason


@pytest.mark.parametrize(
    ("verb", "shape"),
    [
        pytest.param(
            ShapeVerb.RANK,
            _shape(roles=(_role(),), time=TimeLabel.SPAN),
            id="rank-over-a-span",
        ),
        pytest.param(
            ShapeVerb.RANK,
            grown(
                _shape(roles=(_role(),)),
                change=GrowthChange.RELATIVE,
                window=Period(start=2016, end=2020),
            ),
            id="rank-over-a-growth",
        ),
        pytest.param(
            ShapeVerb.SHARE_OF,
            _shape(roles=(_role(),), time=TimeLabel.SPAN),
            id="share-of-over-a-span",
        ),
    ],
)
def test_rank_and_share_of_read_a_span_as_one_period(verb: ShapeVerb, shape: FrameShape):
    # analysis-mcp labels a collapsed frame with one interval, which a rank
    # orders entities within and a share matches on its other side's label.
    assert rejects(verb, shape, axis="donor") is None


def test_rejects_rank_over_more_than_one_role_with_no_axis():
    shape = _shape(roles=(_role(role="donor"), _role(role="sector")))
    reason = rejects(ShapeVerb.RANK, shape, axis=None)
    assert reason is not None
    assert "axis" in reason


def test_rejects_rank_axis_naming_an_unknown_role():
    shape = _shape(roles=(_role(role="donor"),))
    reason = rejects(ShapeVerb.RANK, shape, axis="creditor")
    assert reason is not None
    assert "creditor" in reason


def test_rank_with_a_single_role_and_no_axis_is_not_refused():
    shape = _shape(roles=(_role(role="donor"),))
    assert rejects(ShapeVerb.RANK, shape, axis=None) is None


def test_rejects_growth_relative_over_a_ratio():
    shape = _shape(roles=(_role(),), value=ValueKind.RATIO)
    reason = rejects(ShapeVerb.GROWTH, shape, change=GrowthChange.RELATIVE)
    assert reason is not None
    assert "absolute" in reason


_SPAN = _shape(roles=(_role(),), time=TimeLabel.SPAN)
_PICKED = _shape(roles=(_role(),), time=TimeLabel.PICKED, one_row_per_entity=True)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(_SPAN, id="span"),
        pytest.param(
            grown(_shape(roles=(_role(),)), change=GrowthChange.ABSOLUTE, window=_PERIOD),
            id="growth",
        ),
        pytest.param(
            aggregated(_SPAN, reduction=Reduction.SUM, by_entity=True, by_period=True),
            id="aggregate-by-period",
        ),
        pytest.param(
            ranked(_SPAN, axis=None, window=None, per_period=True, limit=5),
            id="per-period-rank",
        ),
        pytest.param(
            shared(_SPAN, _SPAN, axis="donor", scope=DenominatorScope.POOLED),
            id="share-of",
        ),
        pytest.param(
            aggregated(
                _shape(roles=(_role(),)), reduction=Reduction.MAX, by_entity=True, by_period=False
            ),
            id="growth-after-max",
        ),
    ],
)
def test_rejects_growth_over_a_span_or_a_picked_year_regardless_of_mode(shape: FrameShape):
    # Every verb that reads a span keeps its label, so growth still finds no
    # per-period series behind it; a max or min holds one year per entity,
    # and analysis-mcp refuses growth_gap_year over it.
    assert rejects(ShapeVerb.GROWTH, shape, change=GrowthChange.RELATIVE) is not None
    assert rejects(ShapeVerb.GROWTH, shape, change=GrowthChange.ABSOLUTE) is not None


def test_rejects_two_measure_input_to_every_verb():
    shape = _shape(roles=(), measures=2, roots=("req_a", "req_b"))
    assert rejects(ShapeVerb.AGGREGATE, shape, by_entity=False, by_period=True) is not None
    assert rejects(ShapeVerb.GROWTH, shape, change=GrowthChange.RELATIVE) is not None
    assert rejects(ShapeVerb.RANK, shape, axis=None) is not None
    assert rejects(ShapeVerb.SHARE_OF, shape, axis=None) is not None


def test_rejects_aggregate_grouping_by_entity_with_no_roles():
    shape = _shape(roles=())
    reason = rejects(ShapeVerb.AGGREGATE, shape, by_entity=True, by_period=False)
    assert reason is not None
    assert "entity" in reason


def test_aggregate_grouping_by_period_alone_with_no_roles_is_not_refused():
    shape = _shape(roles=())
    assert rejects(ShapeVerb.AGGREGATE, shape, by_entity=False, by_period=True) is None


def test_rejects_share_of_axis_naming_an_unknown_role():
    shape = _shape(roles=(_role(role="donor"),))
    reason = rejects(ShapeVerb.SHARE_OF, shape, axis="creditor")
    assert reason is not None
    assert "creditor" in reason


def test_rejects_share_of_per_axis_when_the_other_side_lacks_the_role():
    shape = _shape(roles=(_role(role="donor"),))
    other = _shape(roles=(_role(role="sector"),))
    reason = rejects(
        ShapeVerb.SHARE_OF, shape, axis="donor", scope=DenominatorScope.PER_AXIS, other=other
    )
    assert reason is not None
    assert "donor" in reason


def test_share_of_per_axis_when_both_sides_carry_the_role_is_not_refused():
    shape = _shape(roles=(_role(role="donor"),))
    other = _shape(roles=(_role(role="donor"),))
    assert (
        rejects(
            ShapeVerb.SHARE_OF, shape, axis="donor", scope=DenominatorScope.PER_AXIS, other=other
        )
        is None
    )


def test_retrieved_carries_the_measure_count_it_is_given():
    # A family read separately: counted from its source, several where the
    # source's list is empty or capped at one, or uncounted (`None`) where the plan has no
    # source to count from.
    for measures in (3, SEVERAL, None):
        shape = retrieved(
            requirement_id="req_a", roles=(_role(),), period=_PERIOD, measures=measures
        )
        assert shape.measures == measures


@pytest.mark.parametrize(
    ("time", "period", "period_threshold", "refused"),
    [
        (TimeLabel.SPAN, _PERIOD, False, False),
        (TimeLabel.OBSERVATION, _PERIOD, True, False),
        (TimeLabel.SPAN, _PERIOD, True, True),
        pytest.param(
            TimeLabel.SPAN, Period(start=2020, end=2020), True, False, id="span-over-one-year"
        ),
        pytest.param(TimeLabel.PICKED, _PERIOD, True, False, id="picked-years"),
    ],
)
def test_filter_refuses_only_a_year_threshold_over_a_span(
    time: TimeLabel, period: Period, period_threshold: bool, refused: bool
):
    # A span label such as 2018/2022 has no year a threshold orders against;
    # a span over one year is labelled with that year's point.
    shape = _shape(roles=(_role(),), measures=2, value=ValueKind.ORDINAL, time=time, period=period)
    reason = rejects(ShapeVerb.FILTER, shape, period_threshold=period_threshold)
    assert (reason is not None) is refused


def test_compare_and_join_accept_any_single_measure_shape():
    shape = _shape(roles=(_role(),), value=ValueKind.ORDINAL, time=TimeLabel.SPAN)
    assert rejects(ShapeVerb.COMPARE, shape) is None
    assert rejects(ShapeVerb.JOIN, shape) is None


_SPAN_2015_TO_2020 = _shape(roles=(_role(),), time=TimeLabel.SPAN)
_POINTS = _shape(roles=(_role(),))
_ONE_YEAR_SPAN = _shape(roles=(_role(),), time=TimeLabel.SPAN, period=Period(start=2020, end=2020))
_UNKNOWN_SPAN = _shape(roles=(_role(),), time=TimeLabel.SPAN, period=None)
_PICKED_TOTAL = aggregated(_PICKED, reduction=Reduction.SUM, by_entity=True, by_period=False)
_ALL_PAIRED = frozenset({ShapeVerb.SHARE_OF, ShapeVerb.COMPARE, ShapeVerb.JOIN})


@pytest.mark.parametrize(
    ("verb", "options"),
    [
        pytest.param(ShapeVerb.SHARE_OF, {"axis": None}, id="share_of"),
        pytest.param(ShapeVerb.COMPARE, {}, id="compare"),
        pytest.param(ShapeVerb.JOIN, {}, id="join"),
    ],
)
@pytest.mark.parametrize(
    ("shape", "other", "refused"),
    [
        pytest.param(_POINTS, _POINTS, False, id="points-against-points"),
        pytest.param(_SPAN_2015_TO_2020, _SPAN_2015_TO_2020, False, id="the-same-span"),
        pytest.param(_SPAN_2015_TO_2020, _POINTS, True, id="span-against-points"),
        pytest.param(_POINTS, _SPAN_2015_TO_2020, True, id="points-against-span"),
        pytest.param(
            _SPAN_2015_TO_2020,
            _shape(roles=(_role(),), time=TimeLabel.SPAN, period=Period(start=2016, end=2020)),
            True,
            id="spans-over-different-years",
        ),
        pytest.param(_ONE_YEAR_SPAN, _POINTS, False, id="a-one-year-span-reads-as-its-point"),
        pytest.param(_PICKED, _POINTS, frozenset({ShapeVerb.SHARE_OF}), id="picked-against-points"),
        pytest.param(_POINTS, _PICKED, frozenset({ShapeVerb.SHARE_OF}), id="points-against-picked"),
        pytest.param(_PICKED, _PICKED, _ALL_PAIRED, id="picked-against-picked"),
        pytest.param(_PICKED, _SPAN_2015_TO_2020, _ALL_PAIRED, id="picked-against-span"),
        pytest.param(_ONE_YEAR_SPAN, _PICKED, _ALL_PAIRED, id="one-year-span-against-picked"),
        pytest.param(
            _shape(roles=(_role(),), period=Period(start=2020, end=2020)),
            _PICKED,
            _ALL_PAIRED,
            id="one-year-series-against-picked",
        ),
        pytest.param(
            _PICKED,
            _shape(roles=(_role(),), period=Period(start=2016, end=2020)),
            _ALL_PAIRED,
            id="picked-against-a-narrower-series",
        ),
        pytest.param(
            _PICKED,
            _shape(roles=(_role(),), period=Period(start=2010, end=2025)),
            frozenset({ShapeVerb.SHARE_OF}),
            id="picked-against-a-wider-series",
        ),
        pytest.param(
            _PICKED_TOTAL, _SPAN_2015_TO_2020, _ALL_PAIRED, id="total-of-picked-against-span"
        ),
        pytest.param(_UNKNOWN_SPAN, _UNKNOWN_SPAN, _ALL_PAIRED, id="two-unknown-spans"),
    ],
)
def test_a_two_sided_verb_refuses_sides_whose_period_labels_cannot_match(
    verb: ShapeVerb,
    options: dict,
    shape: FrameShape,
    other: FrameShape,
    refused: bool | frozenset[ShapeVerb],
):
    # analysis-mcp matches the two sides on the time_period string, so a span
    # matches only the same span: share_of refuses share_of_grain_mismatch,
    # compare comparison_periods_disjoint, and a join matches no row. A
    # picked year matches a series on each entity's own year in compare and
    # join, so the series has to cover every picked year; share_of needs each
    # key's years equal on both sides. A total of picked years is labelled
    # with the years the data picked, which no stated span is known to match.
    reason = rejects(verb, shape, other=other, **options)
    assert (reason is not None) is (refused if isinstance(refused, bool) else verb in refused)


@pytest.mark.parametrize(
    ("shape", "window", "refused"),
    [
        pytest.param(_SPAN_2015_TO_2020, None, False, id="span-no-window"),
        pytest.param(_SPAN_2015_TO_2020, _PERIOD, False, id="span-the-same-window"),
        pytest.param(
            _SPAN_2015_TO_2020, Period(start=2010, end=2025), False, id="span-a-wider-window"
        ),
        pytest.param(
            _SPAN_2015_TO_2020, Period(start=2017, end=2020), True, id="span-cut-at-the-start"
        ),
        pytest.param(
            _SPAN_2015_TO_2020, Period(start=2015, end=2018), True, id="span-cut-at-the-end"
        ),
        pytest.param(
            _SPAN_2015_TO_2020, Period(start=2021, end=2025), True, id="span-outside-the-window"
        ),
        pytest.param(_POINTS, Period(start=2017, end=2020), False, id="points-any-window"),
        pytest.param(_POINTS, Period(start=2021, end=2025), True, id="points-outside-the-window"),
        pytest.param(_UNKNOWN_SPAN, None, False, id="unknown-span-no-window"),
        pytest.param(
            _UNKNOWN_SPAN, Period(start=2010, end=2025), True, id="unknown-span-any-window"
        ),
    ],
)
def test_a_rank_window_must_keep_a_row_of_its_input(
    shape: FrameShape, window: Period | None, refused: bool
):
    # analysis-mcp keeps a span row only when the window holds both its start
    # and end years; a window cutting the span leaves no row to rank. A span
    # whose years only the data decides fits no window known here. A window
    # sharing no year with the input keeps no row, and run_rank refuses it.
    reason = rejects(ShapeVerb.RANK, shape, axis="donor", window=window)
    assert (reason is not None) is refused


@pytest.mark.parametrize(
    ("verb", "options", "refused"),
    [
        pytest.param(
            ShapeVerb.RANK, {"axis": "donor", "per_period": True}, True, id="per-period-rank"
        ),
        pytest.param(ShapeVerb.RANK, {"axis": "donor"}, True, id="collapsing-rank"),
        pytest.param(
            ShapeVerb.AGGREGATE,
            {"by_entity": True, "by_period": True},
            True,
            id="aggregate-by-period",
        ),
        pytest.param(
            ShapeVerb.AGGREGATE,
            {"by_entity": False, "by_period": False},
            False,
            id="aggregate-across-entities",
        ),
    ],
)
def test_a_reader_within_one_year_refuses_a_picked_frame(
    verb: ShapeVerb, options: dict, refused: bool
):
    # A picked frame holds each entity in the year its max or min fell in,
    # so a per-year reader would set each entity only beside those that
    # peaked in the same year, and a collapsing rank finds one year of each
    # entity's window covered.
    reason = rejects(verb, _PICKED, **options)
    assert (reason is not None) is refused


@pytest.mark.parametrize(
    ("period", "time"),
    [
        pytest.param(Period(start=2016, end=2020), TimeLabel.PICKED, id="several-years"),
        pytest.param(Period(start=2020, end=2020), TimeLabel.OBSERVATION, id="one-year"),
    ],
)
def test_a_year_filter_over_a_picked_frame_leaves_one_year_as_an_observation(
    period: Period, time: TimeLabel
):
    out = filtered(_PICKED, narrowed_roles=(), period=period, pins_one_year=False)
    assert out.time is time


@pytest.mark.parametrize(
    ("shape", "window", "per_period", "period"),
    [
        pytest.param(
            _SPAN_2015_TO_2020, Period(start=2010, end=2025), False, _PERIOD, id="span-wider-window"
        ),
        pytest.param(
            _POINTS, Period(start=2010, end=2025), False, _PERIOD, id="points-wider-window"
        ),
        pytest.param(
            _POINTS, Period(start=2010, end=2025), True, _PERIOD, id="per-period-wider-window"
        ),
        pytest.param(_POINTS, _PERIOD, False, _PERIOD, id="points-equal-window"),
        pytest.param(
            _POINTS,
            Period(start=2016, end=2019),
            False,
            Period(start=2016, end=2019),
            id="points-window-inside-the-period",
        ),
        pytest.param(
            _POINTS,
            Period(start=2017, end=2025),
            False,
            Period(start=2017, end=2020),
            id="points-window-overlapping-the-end",
        ),
        pytest.param(
            _shape(roles=(_role(),), period=None),
            Period(start=2017, end=2020),
            False,
            Period(start=2017, end=2020),
            id="unbounded-points",
        ),
    ],
)
def test_a_rank_holds_the_years_of_its_input_its_window_keeps(
    shape: FrameShape, window: Period, per_period: bool, period: Period
):
    # run_rank labels a collapsed row with the extent of the periods it
    # ranked, so a 2010-2025 window over 2015-2020 labels it 2015/2020, the
    # label a total over 2015-2020 carries.
    out = ranked(shape, axis=None, window=window, per_period=per_period, limit=None)
    assert out.period == period


_SINGLE_MEASURE_VERBS = pytest.mark.parametrize(
    ("verb", "options"),
    [
        (ShapeVerb.AGGREGATE, {"by_entity": False, "by_period": True}),
        (ShapeVerb.GROWTH, {"change": GrowthChange.RELATIVE}),
        (ShapeVerb.RANK, {"axis": "donor"}),
        (ShapeVerb.SHARE_OF, {"axis": None}),
        (ShapeVerb.COMPARE, {}),
        (ShapeVerb.JOIN, {}),
    ],
)


@_SINGLE_MEASURE_VERBS
def test_every_single_measure_verb_refuses_a_family_read_separately(
    verb: ShapeVerb, options: dict[str, object]
):
    # analysis-mcp refuses each of these over a frame holding several StatVars
    # (`require_single_stat_var`; compare's slice needs one row per period).
    shape = _shape(roles=(_role(),), measures=3)
    reason = rejects(verb, shape, **options)
    assert reason is not None
    assert f"{verb.value} reads a single measure" in reason
    # No source says whether the family sums, so the advice is a condition.
    assert "read the family summed (stat_vars='sum') to give it one if it can be summed" in reason


@_SINGLE_MEASURE_VERBS
def test_no_single_measure_verb_refuses_an_uncounted_reading(
    verb: ShapeVerb, options: dict[str, object]
):
    # A plan's separate reading, before any source counts it: a source binding
    # one StatVar gives it one measure, so only the compiler can refuse it.
    shape = _shape(roles=(_role(),), measures=None)
    assert rejects(verb, shape, **options) is None


def test_a_family_summable_over_every_axis_is_told_to_read_it_summed():
    shape = _shape(roles=(_role(),), measures=3, unsummable_axes=())
    reason = rejects(ShapeVerb.AGGREGATE, shape, by_entity=False, by_period=True)
    assert reason == (
        "aggregate reads a single measure; got 3, one per StatVar of a family read separately; "
        "read the family summed (stat_vars='sum') to give it one"
    )


def test_a_family_with_an_unsummable_axis_is_told_to_name_one_value_for_it():
    shape = _shape(roles=(_role(),), measures=2, unsummable_axes=("gender",))
    reason = rejects(ShapeVerb.AGGREGATE, shape, by_entity=False, by_period=True)
    assert reason == (
        "aggregate reads a single measure; got 2, one per StatVar of a family read separately; "
        "it cannot be summed over gender, so the question has to name one value for it"
    )


_MINIMAL_OPTIONS: dict[ShapeVerb, dict[str, object]] = {
    ShapeVerb.AGGREGATE: {"by_entity": False, "by_period": True},
    ShapeVerb.GROWTH: {"change": GrowthChange.RELATIVE},
    ShapeVerb.RANK: {"axis": "donor"},
    ShapeVerb.FILTER: {},
    ShapeVerb.SHARE_OF: {"axis": None},
    ShapeVerb.COMPARE: {},
    ShapeVerb.JOIN: {},
}


def test_every_shape_verb_has_a_rejects_branch():
    shape = _shape(roles=(_role(),))
    for verb in ShapeVerb:
        result = rejects(verb, shape, **_MINIMAL_OPTIONS[verb])
        assert result is None or isinstance(result, str)


# --- rejects: the per-period rank additions -------------------------------------


def test_rejects_share_of_over_an_ordinal_side():
    ordinal = ranked(
        _shape(roles=(_role(role="recipient"),)),
        axis="recipient",
        window=None,
        per_period=True,
        limit=None,
    )
    reason = rejects(ShapeVerb.SHARE_OF, ordinal, axis=None)
    assert reason is not None
    assert "rank position" in reason


def test_a_limited_per_period_rank_carries_the_measured_value_and_share_of_reads_it():
    """`run_rank` writes the measured value for a limited per-period rank, so
    "the top 5 donors each year" plots amounts and a share over it is legal."""
    limited = ranked(
        _shape(roles=(_role(role="recipient"),)),
        axis="recipient",
        window=None,
        per_period=True,
        limit=5,
    )
    assert limited.value is ValueKind.MEASURED
    assert limited.time is TimeLabel.OBSERVATION
    assert limited.one_row_per_entity is False
    assert rejects(ShapeVerb.SHARE_OF, limited, axis=None) is None


def test_rejects_relative_growth_over_a_per_period_rank():
    ordinal = ranked(
        _shape(roles=(_role(role="recipient", narrowed=True),)),
        axis="recipient",
        window=None,
        per_period=True,
        limit=None,
    )
    reason = rejects(ShapeVerb.GROWTH, ordinal, change=GrowthChange.RELATIVE)
    assert reason is not None
    assert "absolute" in reason


def test_absolute_growth_over_a_collapsing_rank_is_refused_as_a_span():
    collapsed = ranked(
        _shape(roles=(_role(role="recipient"),)),
        axis="recipient",
        window=None,
        per_period=False,
        limit=None,
    )
    reason = rejects(ShapeVerb.GROWTH, collapsed, change=GrowthChange.ABSOLUTE)
    assert reason is not None
    assert "span" in reason


def test_absolute_growth_over_an_unnarrowed_per_period_rank_names_the_filter_as_the_fix():
    ordinal = ranked(
        _shape(roles=(_role(role="recipient", narrowed=False),)),
        axis="recipient",
        window=None,
        per_period=True,
        limit=None,
    )
    reason = rejects(ShapeVerb.GROWTH, ordinal, change=GrowthChange.ABSOLUTE)
    assert reason is not None
    assert "filter" in reason
    assert "recipient" in reason


def test_absolute_growth_over_a_narrowed_per_period_rank_is_accepted():
    """A growth over a collapsing rank always refuses (span), and a growth over
    a per-period rank refuses until a filter narrows it. Both refusals read
    as plain error messages, the safe default that keeps an ambiguous
    growth from producing a wrong number.
    """
    ordinal = ranked(
        _shape(roles=(_role(role="recipient", narrowed=True),)),
        axis="recipient",
        window=None,
        per_period=True,
        limit=None,
    )
    assert rejects(ShapeVerb.GROWTH, ordinal, change=GrowthChange.ABSOLUTE) is None


def test_rejects_population_rank_over_an_already_narrowed_axis():
    shape = _shape(roles=(_role(role="recipient", narrowed=True),))
    reason = rejects(ShapeVerb.RANK, shape, axis="recipient", among=RankAmong.POPULATION)
    assert reason is not None
    assert "population" in reason.lower()
    assert "named" in reason.lower()


def test_named_rank_over_a_narrowed_axis_is_not_refused_by_the_population_rule():
    shape = _shape(roles=(_role(role="recipient", narrowed=True),))
    assert rejects(ShapeVerb.RANK, shape, axis="recipient", among=RankAmong.NAMED) is None


def test_population_rank_over_an_unnarrowed_axis_is_not_refused():
    shape = _shape(roles=(_role(role="recipient", narrowed=False),))
    assert rejects(ShapeVerb.RANK, shape, axis="recipient", among=RankAmong.POPULATION) is None


# --- a per-period rank with a limit leaves gaps by construction ------------------


def _top_five_each_year(period: Period | None = _PERIOD) -> FrameShape:
    return ranked(
        _shape(roles=(_role(role="donor"),), period=period),
        axis="donor",
        window=None,
        per_period=True,
        limit=5,
    )


def test_only_a_limited_per_period_rank_over_several_years_is_cut():
    assert _top_five_each_year().cut_per_period is True
    # One year has no gap between years to leave.
    assert _top_five_each_year(Period(start=2020, end=2020)).cut_per_period is False
    source = _shape(roles=(_role(role="donor"),))
    unlimited = ranked(source, axis="donor", window=None, per_period=True, limit=None)
    collapsed = ranked(source, axis="donor", window=None, per_period=False, limit=5)
    assert unlimited.cut_per_period is False
    assert collapsed.cut_per_period is False
    # Re-ranking each period of a cut frame keeps its gaps.
    reranked = ranked(_top_five_each_year(), axis="donor", window=None, per_period=True, limit=None)
    assert reranked.cut_per_period is True


def test_a_filter_carries_the_cut_only_while_its_period_spans_years():
    cut = _top_five_each_year()
    kept = filtered(cut, narrowed_roles=("donor",), period=cut.period, pins_one_year=False)
    one_year = filtered(
        cut, narrowed_roles=(), period=Period(start=2020, end=2020), pins_one_year=True
    )
    # A lone `year >= 2020` narrows the period to its last year without pinning it.
    last_year = filtered(
        cut, narrowed_roles=(), period=Period(start=2020, end=2020), pins_one_year=False
    )
    last_two = filtered(
        cut, narrowed_roles=(), period=Period(start=2019, end=2020), pins_one_year=False
    )
    assert kept.cut_per_period is True
    assert one_year.cut_per_period is False
    assert last_year.cut_per_period is False
    assert last_two.cut_per_period is True
    assert rejects(ShapeVerb.AGGREGATE, last_year, by_entity=True) is None
    assert rejects(ShapeVerb.AGGREGATE, last_two, by_entity=True) is not None


def test_an_aggregate_keeps_the_cut_only_when_it_keeps_entity_and_period_apart():
    cut = _top_five_each_year()
    assert (
        aggregated(cut, reduction=Reduction.SUM, by_entity=True, by_period=True).cut_per_period
        is True
    )
    assert (
        aggregated(cut, reduction=Reduction.SUM, by_entity=False, by_period=True).cut_per_period
        is False
    )


def test_every_other_transform_clears_the_cut():
    cut = _top_five_each_year()
    whole = _shape(roles=(_role(role="donor"),))
    window = Period(start=2016, end=2020)
    assert grown(cut, change=GrowthChange.ABSOLUTE, window=window).cut_per_period is False
    assert shared(cut, whole, axis=None, scope=None).cut_per_period is False
    assert compared(cut, whole).cut_per_period is False
    assert joined(cut, whole).cut_per_period is False


def test_growth_over_a_limited_per_period_rank_is_refused_even_when_narrowed():
    """`run_growth` refuses an entity missing a year of its window, and the
    limit removes the years an entity missed the cut, so no narrowing makes
    the series whole."""
    cut = _top_five_each_year()
    narrowed = filtered(cut, narrowed_roles=("donor",), period=cut.period, pins_one_year=False)
    for shape in (cut, narrowed):
        for change in GrowthChange:
            reason = rejects(ShapeVerb.GROWTH, shape, change=change)
            assert reason is not None
            assert "compute the growth first, then rank its result with the limit" in reason


def test_a_share_of_per_entity_over_a_limited_per_period_rank_is_refused():
    cut = _top_five_each_year()
    whole = _shape(roles=(_role(role="donor"),))
    order = "compute the share_of first, then rank the shares per period with the limit"
    keyed = rejects(
        ShapeVerb.SHARE_OF, cut, axis="donor", scope=DenominatorScope.POOLED, other=whole
    )
    bound = rejects(ShapeVerb.SHARE_OF, cut, axis=None, entity="Germany")
    # A per-axis denominator divides each entity by its own series.
    per_axis = rejects(
        ShapeVerb.SHARE_OF, whole, axis="donor", scope=DenominatorScope.PER_AXIS, other=cut
    )
    for reason in (keyed, per_axis):
        assert reason is not None
        assert order in reason
    # A share narrowed to one entity carries no role for a rank to read, so
    # its advice is the frame the rank reads.
    assert bound is not None
    assert order not in bound
    assert "read the share_of's side from the frame that rank reads, keeping its entity" in bound
    assert rejects(ShapeVerb.SHARE_OF, whole, axis=None, entity="Germany") is None
    one_entity_share = shared(whole, whole, axis=None, scope=None)
    assert rejects(ShapeVerb.RANK, one_entity_share, axis=None) is not None
    # Ranking per-axis shares keeps the top entities by share; ranking pooled
    # ones keeps those the measure ranked first.
    by_share = "which keeps the top entities by share, not by the measure the limit ranked"
    assert keyed is not None and by_share not in keyed
    assert per_axis is not None and by_share in per_axis


def test_a_share_of_reading_a_limited_per_period_rank_as_one_total_is_accepted():
    """The top 5's share of the total each year sums each period's kept
    rows, and every period keeps some."""
    cut = _top_five_each_year()
    whole = _shape(roles=(_role(role="donor"),))
    assert rejects(ShapeVerb.SHARE_OF, cut, axis=None) is None
    # A pooled denominator is one total per period, so a cut one has no gaps.
    assert (
        rejects(ShapeVerb.SHARE_OF, whole, axis="donor", scope=DenominatorScope.POOLED, other=cut)
        is None
    )


def test_an_aggregate_by_entity_alone_over_a_limited_per_period_rank_is_refused():
    """Each entity's total would cover only the periods it made the cut."""
    cut = _top_five_each_year()
    reason = rejects(ShapeVerb.AGGREGATE, cut, by_entity=True, by_period=False)
    assert reason is not None
    assert "rank the source once, with the reduction, the limit and per_period=false" in reason
    assert rejects(ShapeVerb.AGGREGATE, cut, by_entity=False, by_period=True) is None
    assert rejects(ShapeVerb.AGGREGATE, cut, by_entity=True, by_period=True) is None


def test_the_plan_the_aggregate_refusal_advises_is_one_rank_reads():
    """A collapsing rank over the source holds each entity's whole series
    reduced to one value, then the limit."""
    source = _shape(roles=(_role(role="donor"),))
    assert rejects(ShapeVerb.RANK, source, axis="donor", among=RankAmong.POPULATION) is None
    top = ranked(source, axis="donor", window=None, per_period=False, limit=5)
    assert top.one_row_per_entity is True
    assert top.cut_per_period is False


# --- a rank limit keeps entities neither named nor the source's population ---------


def test_a_limit_marks_the_frame_limited_in_either_mode_and_over_one_year():
    source = _shape(roles=(_role(role="donor"),))
    one_year = _top_five_each_year(Period(start=2020, end=2020))
    collapsed = ranked(source, axis="donor", window=None, per_period=False, limit=5)
    unlimited = ranked(source, axis="donor", window=None, per_period=True, limit=None)
    assert _top_five_each_year().limited is True
    assert one_year.limited is True
    assert one_year.cut_per_period is False
    # A span is one period, so a limit over it leaves no gap between periods.
    over_a_span = ranked(_SPAN, axis=None, window=None, per_period=True, limit=5)
    assert (over_a_span.limited, over_a_span.cut_per_period) == (True, False)
    assert collapsed.limited is True
    assert unlimited.limited is False


def test_every_transform_that_keeps_the_kept_entities_carries_limited():
    limited = _top_five_each_year()
    whole = _shape(roles=(_role(role="donor"),))
    window = Period(start=2016, end=2020)
    one_year = Period(start=2020, end=2020)
    assert filtered(limited, narrowed_roles=(), period=one_year, pins_one_year=True).limited
    assert aggregated(limited, reduction=Reduction.SUM, by_entity=True, by_period=True).limited
    assert aggregated(limited, reduction=Reduction.SUM, by_entity=False, by_period=True).limited
    assert grown(limited, change=GrowthChange.ABSOLUTE, window=window).limited
    assert shared(limited, whole, axis="donor", scope=DenominatorScope.POOLED).limited
    # A share keeps every numerator entity, so a limited denominator does not
    # limit it.
    assert not shared(whole, limited, axis="donor", scope=DenominatorScope.POOLED).limited


def test_a_per_axis_share_of_over_a_limited_denominator_is_refused():
    """`run_share_of` refuses a per-axis call at the first numerator entity
    with no denominator group, and a limit drops some. One year proves the
    refusal rests on the limit, not on the cut."""
    one_year = Period(start=2022, end=2022)
    top = _top_five_each_year(one_year)
    assert top.cut_per_period is False
    whole = _shape(roles=(_role(role="donor"),), period=one_year)
    order = (
        "compute the share_of over the unlimited denominator, then rank the shares with the "
        "limit, which keeps the top entities by share, not by the measure the limit ranked"
    )
    for numerator in (whole, top):
        reason = rejects(
            ShapeVerb.SHARE_OF, numerator, axis="donor", scope=DenominatorScope.PER_AXIS, other=top
        )
        assert reason is not None
        assert order in reason
    # A pooled total and a limited numerator over a whole denominator both hold.
    assert (
        rejects(ShapeVerb.SHARE_OF, whole, axis="donor", scope=DenominatorScope.POOLED, other=top)
        is None
    )
    assert (
        rejects(ShapeVerb.SHARE_OF, top, axis="donor", scope=DenominatorScope.PER_AXIS, other=whole)
        is None
    )


def test_any_rank_over_a_limited_rank_is_refused():
    """The review's case: among each year's top 10, which 3 received least.
    The frame holds the 8th to 10th largest, so neither `among` is true of it."""
    source = _shape(roles=(_role(role="donor"),))
    top_ten = ranked(source, axis="donor", window=None, per_period=True, limit=10)
    single_year = ranked(
        _shape(roles=(_role(role="donor"),), period=Period(start=2022, end=2022)),
        axis="donor",
        window=None,
        per_period=True,
        limit=10,
    )
    last_year = filtered(
        top_ten, narrowed_roles=(), period=Period(start=2020, end=2020), pins_one_year=True
    )
    named = filtered(top_ten, narrowed_roles=("donor",), period=top_ten.period, pins_one_year=False)
    order = "rank the source once with the limit the question needs"
    for shape, among in (
        (top_ten, RankAmong.POPULATION),
        (single_year, RankAmong.POPULATION),
        (last_year, RankAmong.POPULATION),
        (named, RankAmong.NAMED),
    ):
        reason = rejects(ShapeVerb.RANK, shape, axis="donor", among=among)
        assert reason is not None
        assert order in reason


def test_any_rank_over_rank_positions_is_refused():
    """With ODA of 100 and 10, an unlimited per-period rank writes positions
    1 and 2, and a highest-first rank over those keeps the donor at position
    2 as "the top donor". A filter and a group that keeps both entity and
    period carry the positions through."""
    source = _shape(roles=(_role(role="donor"),))
    each_year = ranked(source, axis="donor", window=None, per_period=True, limit=None)
    assert each_year.value is ValueKind.ORDINAL
    named = filtered(
        each_year, narrowed_roles=("donor",), period=each_year.period, pins_one_year=False
    )
    one_year = filtered(
        each_year, narrowed_roles=(), period=Period(start=2020, end=2020), pins_one_year=True
    )
    regrouped = aggregated(each_year, reduction=Reduction.SUM, by_entity=True, by_period=True)
    order = "rank the source once with the limit the question needs"
    for shape, among in (
        (each_year, RankAmong.POPULATION),
        (named, RankAmong.NAMED),
        (one_year, RankAmong.POPULATION),
        (regrouped, RankAmong.POPULATION),
    ):
        reason = rejects(ShapeVerb.RANK, shape, axis="donor", among=among)
        assert reason is not None
        assert "rank position" in reason
        assert order in reason


def test_a_change_in_rank_position_stays_ordinal():
    """analysis-mcp carries its ordinal marker through a growth, so `rank`,
    `share_of` and a relative `growth` refuse a change in places as they
    refuse the positions."""
    positions = ranked(
        _shape(roles=(_role(role="donor", narrowed=True),)),
        axis="donor",
        window=None,
        per_period=True,
        limit=None,
    )
    places = grown(positions, change=GrowthChange.ABSOLUTE, window=Period(start=2015, end=2020))
    assert places.value is ValueKind.ORDINAL
    assert rejects(ShapeVerb.RANK, places, axis="donor", among=RankAmong.NAMED) is not None
    assert rejects(ShapeVerb.SHARE_OF, places, axis=None) is not None
    assert rejects(ShapeVerb.GROWTH, places, change=GrowthChange.RELATIVE) is not None


def test_a_relative_growth_over_unnarrowed_rank_positions_names_both_corrections():
    """`change='absolute'` alone meets the narrowing rule next, so the reason
    states the filter with it."""
    positions = ranked(
        _shape(roles=(_role(role="recipient"),)),
        axis="recipient",
        window=None,
        per_period=True,
        limit=None,
    )
    reason = rejects(ShapeVerb.GROWTH, positions, change=GrowthChange.RELATIVE)
    assert reason is not None
    assert "change='absolute'" in reason
    assert "filter" in reason
    assert "recipient" in reason
    narrowed = filtered(
        positions, narrowed_roles=("recipient",), period=positions.period, pins_one_year=False
    )
    assert rejects(ShapeVerb.GROWTH, narrowed, change=GrowthChange.ABSOLUTE) is None


def _years(*thresholds: tuple[str, int]) -> tuple[PeriodThreshold, ...]:
    return tuple(
        PeriodThreshold.model_validate({"op": op, "year": year}) for op, year in thresholds
    )


@pytest.mark.parametrize(
    ("period", "predicates", "left"),
    [
        # An exclusion that leaves one year.
        ((2020, 2021), _years(("ne", 2020)), (2021, 2021)),
        ((2020, 2021), _years(("ne", 2021)), (2020, 2020)),
        ((2020, 2022), _years(("ne", 2020), ("ne", 2022)), (2021, 2021)),
        ((2015, 2022), _years(("gte", 2021), ("ne", 2022)), (2021, 2021)),
        ((2020, 2020), _years(("ne", 2019)), (2020, 2020)),
        # An exclusion that leaves several: an end moves, the middle does not.
        ((2020, 2022), _years(("ne", 2020)), (2021, 2022)),
        ((2020, 2022), _years(("ne", 2022)), (2020, 2021)),
        ((2020, 2022), _years(("ne", 2021)), (2020, 2022)),
        ((2020, 2022), _years(("ne", 2019)), (2020, 2022)),
        # Exclusions that remove every year.
        ((2020, 2020), _years(("ne", 2020)), None),
        ((2020, 2021), _years(("ne", 2020), ("ne", 2021)), None),
        ((2015, 2022), _years(("gt", 2020), ("ne", 2021), ("ne", 2022)), None),
        # An exclusion beside an `eq`.
        ((2020, 2022), _years(("eq", 2021), ("ne", 2020)), (2021, 2021)),
        ((2020, 2022), _years(("eq", 2021), ("ne", 2021)), None),
        # A bound that meets the period's own end.
        ((2015, 2022), _years(("gt", 2021)), (2022, 2022)),
        ((2015, 2022), _years(("lt", 2016)), (2015, 2015)),
        ((2015, 2022), _years(("gte", 2022)), (2022, 2022)),
        # No exclusion.
        ((2020, 2022), _years(("eq", 2022)), (2022, 2022)),
        ((2015, 2022), _years(("gt", 2017), ("lt", 2019)), (2018, 2018)),
        ((2015, 2022), _years(("gt", 2017), ("lt", 2020)), (2018, 2019)),
        ((2020, 2022), _years(("eq", 2030)), None),
    ],
)
def test_a_year_filter_narrows_the_period_and_says_whether_one_year_is_left(
    period: tuple[int, int],
    predicates: tuple[PeriodThreshold, ...],
    left: tuple[int, int] | None,
) -> None:
    start, end = period

    out = year_filter(Period(start=start, end=end), predicates)

    if left is None:
        assert out == f"the year filter leaves no year of {start}-{end}"
    else:
        assert out == YearFilter(
            period=Period(start=left[0], end=left[1]), leaves_one_year=left[0] == left[1]
        )


@pytest.mark.parametrize("period", [Period(start=2020, end=2022), Period(start=2020, end=2020)])
def test_a_filter_with_no_year_predicate_leaves_the_period_and_no_one_year(period: Period) -> None:
    assert year_filter(period, ()) == YearFilter(period=period, leaves_one_year=False)


@pytest.mark.parametrize(
    ("predicates", "leaves_one_year"),
    [
        (_years(("eq", 2020)), True),
        (_years(("gte", 2020), ("lte", 2020)), True),
        (_years(("ne", 2020)), False),
        (_years(("gte", 2020)), False),
        ((), False),
    ],
)
def test_an_unbounded_period_stays_unbounded_under_a_year_filter(
    predicates: tuple[PeriodThreshold, ...], leaves_one_year: bool
) -> None:
    assert year_filter(None, predicates) == YearFilter(period=None, leaves_one_year=leaves_one_year)


def test_a_year_exclusion_leaving_one_year_of_a_limited_rank_clears_its_cut() -> None:
    cut = _top_five_each_year(Period(start=2020, end=2021))
    years = year_filter(cut.period, _years(("ne", 2020)))
    assert isinstance(years, YearFilter)

    one_year = filtered(
        cut, narrowed_roles=(), period=years.period, pins_one_year=years.leaves_one_year
    )

    assert cut.cut_per_period is True
    assert cut.one_row_per_entity is False
    assert one_year.period == Period(start=2021, end=2021)
    assert one_year.cut_per_period is False
    assert one_year.one_row_per_entity is True
    assert rejects(ShapeVerb.AGGREGATE, one_year, by_entity=True, by_period=False) is None


# --- place levels: a retrieval that names no place holds every level ------------------


def _every_place() -> FrameShape:
    return retrieved(
        requirement_id="req_a",
        roles=(_role(role="country"),),
        period=_PERIOD,
        measures=1,
        names_no_place=True,
    )


def _countries(shape: FrameShape) -> FrameShape:
    return filtered(
        shape, narrowed_roles=(), period=shape.period, pins_one_year=False, keeps_one_level=True
    )


def test_a_retrieval_naming_no_place_mixes_levels_and_one_naming_places_does_not():
    named = retrieved(requirement_id="req_a", roles=(_role(),), period=_PERIOD, measures=1)

    assert (_every_place().mixed_levels, _every_place().place_typed) == (True, True)
    assert (named.mixed_levels, named.place_typed) == (False, False)


def test_a_rank_a_total_and_a_share_over_mixed_levels_are_refused():
    shape = _every_place()
    every_level = "holds every place with data at every level"

    for reason in (
        rejects(ShapeVerb.RANK, shape, axis="country"),
        rejects(ShapeVerb.AGGREGATE, shape, by_entity=False, by_period=True),
        rejects(ShapeVerb.SHARE_OF, shape, axis="country", scope=DenominatorScope.POOLED),
    ):
        assert reason is not None
        assert every_level in reason
        assert "place_level" in reason


def test_a_step_that_keeps_each_place_apart_reads_mixed_levels_and_carries_them():
    shape = _every_place()

    assert rejects(ShapeVerb.AGGREGATE, shape, by_entity=True, by_period=False) is None
    assert rejects(ShapeVerb.GROWTH, shape, change=GrowthChange.RELATIVE) is None
    per_place = aggregated(shape, reduction=Reduction.SUM, by_entity=True, by_period=True)
    growth = grown(shape, change=GrowthChange.RELATIVE, window=_PERIOD)

    assert (per_place.mixed_levels, per_place.place_typed) == (True, False)
    assert (growth.mixed_levels, growth.place_typed) == (True, False)
    assert rejects(ShapeVerb.RANK, growth, axis="country") is not None


def test_a_level_filter_leaves_one_level_and_any_other_filter_leaves_them_mixed():
    one_year = filtered(
        _every_place(), narrowed_roles=(), period=Period(start=2020, end=2020), pins_one_year=True
    )
    countries = _countries(one_year)

    assert (one_year.mixed_levels, one_year.place_typed) == (True, True)
    assert (countries.mixed_levels, countries.place_typed) == (False, True)
    assert rejects(ShapeVerb.RANK, countries, axis="country") is None
    assert rejects(ShapeVerb.AGGREGATE, countries, by_entity=False, by_period=True) is None


def test_a_level_filter_reads_only_a_frame_that_still_carries_place_types():
    named = retrieved(requirement_id="req_a", roles=(_role(),), period=_PERIOD, measures=1)
    per_place = aggregated(_every_place(), reduction=Reduction.SUM, by_entity=True, by_period=True)

    assert rejects(ShapeVerb.FILTER, _every_place(), place_level=True) is None
    assert rejects(ShapeVerb.FILTER, named) is None
    for shape in (named, per_place):
        reason = rejects(ShapeVerb.FILTER, shape, place_level=True)
        assert reason is not None
        assert "a place_level filter reads a requirement that names no place" in reason


def test_a_plan_states_exactly_the_levels_analysis_mcp_keeps():
    # A PlaceLevel compiles to analysis-mcp's place_level predicate, whose level
    # is one of common.place_type.PLACE_LEVELS.
    assert {level.value for level in PlaceType} == PLACE_LEVELS
