"""Time-period grain classification: `year`, `month`, or `day`, read from a
`time_period` cell by `dfo_frame_grammar.grammar.parse_time_period`, so a
point (`2018`) and an interval (`2018/2022`) at one grain classify alike.
`is_interval` tells the two apart, and `span_label` writes the one cell that
covers a set of periods.

Lives here rather than in `analysis_mcp` so a materialisation-time check
(data-mcp, coercing an upstream artifact's own `time_period` column) can
reuse the identical classifier without a move.
"""

from __future__ import annotations

from collections.abc import Iterable

from dfo_frame_grammar import TimeGrain, parse_time_period

__all__ = ["TimeGrain", "distinct_time_grains", "is_interval", "span_label", "time_grain"]

_GRAIN_ORDER: tuple[TimeGrain, ...] = ("year", "month", "day")


def time_grain(period: str) -> TimeGrain:
    """`period`'s own grain.

    Raises:
        FrameError: `invalid_time_period` when `period` is not a grammar-valid
            `time_period` cell.
    """
    return parse_time_period(period).grain


def distinct_time_grains(periods: Iterable[str]) -> tuple[TimeGrain, ...]:
    """Every distinct grain `periods` carries, in `year`/`month`/`day`
    order. More than one entry means `periods` is not single-grain; empty
    only when `periods` itself is empty.
    """
    found = {time_grain(period) for period in periods}
    return tuple(grain for grain in _GRAIN_ORDER if grain in found)


def is_interval(period: str) -> bool:
    """Whether `period` is an interval (`"2018/2022"`) rather than one point (`"2018"`).

    Raises:
        FrameError: `invalid_time_period` when `period` is not a grammar-valid
            `time_period` cell.
    """
    cell = parse_time_period(period)
    return cell.start != cell.end


def span_label(periods: Iterable[str]) -> str:
    """The `time_period` cell covering every one of `periods`, from the earliest
    start to the latest end, written as the point when they are equal.

    `["2018", "2020", "2022"]` gives `"2018/2022"`, `["2018/2022"]` gives
    `"2018/2022"` and `["2022"]` gives `"2022"`.

    Raises:
        ValueError: `periods` is empty or mixes grains.
    """
    cells = [parse_time_period(period) for period in periods]
    if not cells:
        raise ValueError("span_label needs at least one period")
    if len(grains := {cell.grain for cell in cells}) > 1:
        raise ValueError(f"span_label periods mix grains: {sorted(grains)}")
    start = min(cell.start for cell in cells)
    end = max(cell.end for cell in cells)
    return start if start == end else f"{start}/{end}"
