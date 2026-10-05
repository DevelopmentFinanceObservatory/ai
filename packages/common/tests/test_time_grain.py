"""`time_grain`/`distinct_time_grains`: grain classification through the frame grammar's parser;
`span_label`: the one cell covering a set of periods."""

from __future__ import annotations

import pytest
from common.time_grain import distinct_time_grains, span_label, time_grain


@pytest.mark.parametrize(
    ("period", "grain"),
    [
        ("2020", "year"),
        ("2020-01", "month"),
        ("2020-01-15", "day"),
        ("2018/2022", "year"),
        ("2018-01/2022-12", "month"),
        ("2018-01-01/2018-01-31", "day"),
    ],
)
def test_time_grain_classifies_each_format(period: str, grain: str) -> None:
    assert time_grain(period) == grain


def test_distinct_time_grains_is_empty_for_no_periods() -> None:
    assert distinct_time_grains([]) == ()


def test_distinct_time_grains_reports_one_entry_for_a_single_grain() -> None:
    assert distinct_time_grains(["2020", "2021", "2022"]) == ("year",)


def test_distinct_time_grains_reports_every_grain_present_in_year_month_day_order() -> None:
    assert distinct_time_grains(["2021-01", "2020", "2020-01-15"]) == ("year", "month", "day")


@pytest.mark.parametrize(
    ("periods", "label"),
    [
        pytest.param(["2022", "2018", "2020"], "2018/2022", id="points-give-their-extent"),
        pytest.param(["2018/2022"], "2018/2022", id="a-span-is-unchanged"),
        pytest.param(["2022"], "2022", id="one-point-stays-a-point"),
        pytest.param(["2018/2020", "2019/2022"], "2018/2022", id="spans-give-their-union"),
        pytest.param(["2018-11", "2019-02"], "2018-11/2019-02", id="month-grain"),
    ],
)
def test_span_label_writes_the_window_the_periods_cover(periods: list[str], label: str) -> None:
    assert span_label(periods) == label


@pytest.mark.parametrize("periods", [[], ["2018", "2018-01"]], ids=["empty", "mixed-grain"])
def test_span_label_refuses_periods_with_no_single_window(periods: list[str]) -> None:
    with pytest.raises(ValueError, match="span_label"):
        span_label(periods)
