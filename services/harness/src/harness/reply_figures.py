"""What a drawn card holds, and the values of it a reply may state.

`make_charts` (`harness.orchestrator_tools`) returns two things for each
card it draws, both built here from the drawn payload of the platform's
write reply, which holds every series the card draws with every point:
`holds`, what the card covers with no value in it, and `reply_figures`, the
values a reply may state, each with its series, unit and period.

The platform's fact summary lists at most twelve of a card's series, so no
count, extreme or value here is read from it. It supplies only identity,
where the payload carries none: a stat card's series id, and the coordinates
of the series it lists, which a series is named by.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from analysis_mcp.rolling_tool import coordinate_text
from dfo_frame_grammar.grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    FrameError,
    parse_time_period,
)

from harness.chart_binding import TopMembers

# The dimensions of a table row that name no entity, which its label leaves out.
_UNLABELLED_DIMENSIONS = frozenset({STAT_VAR_COLUMN, UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})
# The frame columns that are not coordinates of a series.
_NOT_COORDINATES = frozenset({TIME_PERIOD_COLUMN, VALUE_COLUMN, STATUS_COLUMN})

type Point = tuple[str | None, float]


@dataclass(frozen=True)
class DrawnSeries:
    """One series a card draws. `points` are its observed points, each a
    period and a value, in drawn order, which need not be period order
    (`series_ends`); a point with no value is left out. `coordinates` are
    None when nothing states them."""

    id: str | None
    label: str
    unit: Any
    points: tuple[Point, ...]
    coordinates: Mapping[str, str] | None = None


def drawn_series(component: Mapping[str, Any], summary: Mapping[str, Any]) -> list[DrawnSeries]:
    """Every series `component`, the drawn component of a write reply's
    payload, holds, in drawn order. A line or bar chart draws its `series`. A
    stat card is one series of its headline point. A table's rows are grouped
    as the platform groups them, by `series_id` when every row carries one
    and by `dimensions` otherwise. Any other card draws none.

    `summary`, the card's wire fact summary, gives a series its coordinates,
    and a series the payload leaves unnamed its id. It matches by id where the
    id is unique in both, and by position where the payload names no series
    and the summary lists as many as the card draws.
    """
    match component.get("type"):
        case "stat_card":
            value = component.get("value")
            series = [
                DrawnSeries(
                    id=None,
                    label=str(component.get("label") or ""),
                    unit=component.get("unit"),
                    points=() if value is None else ((_period(component), value),),
                )
            ]
        case "table":
            series = _table_series(component.get("rows") or [])
        case _:
            series = [
                DrawnSeries(
                    id=drawn.get("id") or None,
                    label=str(drawn.get("label") or drawn.get("id") or ""),
                    unit=drawn.get("unit"),
                    points=_observed(drawn.get("points") or []),
                )
                for drawn in component.get("series") or []
            ]
    return _identified(series, [e for e in summary.get("series") or [] if isinstance(e, Mapping)])


def _period(point: Mapping[str, Any]) -> str | None:
    period = point.get("period")
    return None if period is None else str(period)


def _observed(points: Sequence[Mapping[str, Any]]) -> tuple[Point, ...]:
    """The points holding a value. One the platform marks missing holds no
    `value`, and a null `value` is none."""
    return tuple(
        (_period(point), point["value"]) for point in points if point.get("value") is not None
    )


def _period_key(period: str | None) -> tuple[str, str] | None:
    """Where `period` sits in time, as the day it ends at and then the day it
    begins at, or None when it is not a period the frame grammar writes
    (`parse_time_period`): a year, a month, a day, or a span of two of one
    grain. A year or a month ends at its last day, so periods of every grain
    order together, and a span orders by its end and then its start."""
    if period is None:
        return None
    try:
        parsed = parse_time_period(period)
    except FrameError:
        return None
    # Zero-padded days order by code point as by date, and no month runs past day 31.
    return parsed.end + "-12-31"[len(parsed.end) - 4 :], parsed.start + "-01-01"[
        len(parsed.start) - 4 :
    ]


def _first_and_last(periods: Sequence[str | None]) -> tuple[int, int]:
    """The positions of the earliest and the latest of `periods`, at least
    one, by `_period_key`. When any of them has no key they keep the order
    they are given in, so the result is the first position and the last."""
    keys = [_period_key(period) for period in periods]
    if None in keys:
        return 0, len(periods) - 1
    order = sorted(range(len(periods)), key=lambda index: keys[index] or ("", ""))
    return order[0], order[-1]


def series_ends(series: DrawnSeries) -> tuple[Point, Point] | None:
    """The first and the last point of `series`, or None when it holds no
    point. They are the earliest and the latest by period (`_period_key`),
    whatever order the series draws its points in. A series holding a period
    with no key begins at its first drawn point and ends at its last."""
    points = series.points
    if not points:
        return None
    first, last = _first_and_last([period for period, _ in points])
    return points[first], points[last]


def _table_series(rows: Sequence[Mapping[str, Any]]) -> list[DrawnSeries]:
    """A table's rows as series, in first-seen order. A series is labelled as
    the platform labels it: the values of its entity dimensions, which the
    rows hold as labels, joined in the order of their names, or its
    `stat_var` when it has none."""
    by_id = bool(rows) and all(
        isinstance(row.get("series_id"), str) and row["series_id"] for row in rows
    )
    names = list(rows[0].get("dimensions") or {}) if rows else []
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        dimensions = row.get("dimensions") or {}
        key = row["series_id"] if by_id else tuple(dimensions.get(name) for name in names)
        groups.setdefault(key, []).append(row)
    series = []
    for key, members in groups.items():
        dimensions = members[0].get("dimensions") or {}
        entities = [
            str(dimensions[name])
            for name in sorted(names)
            if name not in _UNLABELLED_DIMENSIONS and dimensions.get(name) is not None
        ]
        series.append(
            DrawnSeries(
                id=key if by_id else None,
                label=" / ".join(entities) or str(dimensions.get(STAT_VAR_COLUMN) or ""),
                unit=members[0].get("unit"),
                points=_observed(members),
            )
        )
    return series


def _identified(
    series: list[DrawnSeries], listed: Sequence[Mapping[str, Any]]
) -> list[DrawnSeries]:
    """`series` with the ids and coordinates the summary's `listed` series
    state for them. A repeated id names no one series, so it gives nothing."""
    if all(one.id is None for one in series):
        if len(listed) != len(series):
            return series
        return [
            replace(one, id=entry.get("id") or None, coordinates=_coordinates(entry))
            for one, entry in zip(series, listed, strict=True)
        ]
    drawn_ids = Counter(one.id for one in series)
    listed_ids = Counter(entry.get("id") for entry in listed)
    by_id = {entry.get("id"): entry for entry in listed}
    return [
        replace(one, coordinates=_coordinates(by_id[one.id]))
        if one.id is not None and drawn_ids[one.id] == 1 and listed_ids[one.id] == 1
        else one
        for one in series
    ]


def _coordinates(entry: Mapping[str, Any]) -> dict[str, str] | None:
    held = {
        str(name): value
        for name, value in (entry.get("coordinates") or {}).items()
        if isinstance(value, str)
    }
    return held or None


def unplaced_names(series: Sequence[DrawnSeries], names: Sequence[str]) -> list[str]:
    """The `names` matching no series of a card some of whose series state no
    coordinates, since a name among them may be one of those series."""
    if all(one.coordinates is not None for one in series):
        return []
    return [name for name in names if not _named(series, name)]


def with_coordinates(
    series: Sequence[DrawnSeries], rows: Sequence[Mapping[str, Any]]
) -> list[DrawnSeries]:
    """`series` with the coordinates of each one that states none, found
    among `rows`, the rows of the frame the card draws: the one series of the
    frame holding every point the drawn series holds, at the same value. A
    drawn series that several of the frame's series could be, or that shares
    its frame series with another, stays without coordinates, as does one
    with no dated point."""
    held: dict[tuple[tuple[str, str], ...], dict[str, float]] = {}
    for row in rows:
        if row.get(VALUE_COLUMN) is None:
            continue
        cells = ((name, coordinate_text(cell)) for name, cell in row.items())
        key = tuple(
            sorted(
                (name, text)
                for name, text in cells
                if name not in _NOT_COORDINATES and text is not None
            )
        )
        held.setdefault(key, {})[str(row[TIME_PERIOD_COLUMN])] = row[VALUE_COLUMN]
    found: dict[int, tuple[tuple[str, str], ...]] = {}
    for index, one in enumerate(series):
        if one.coordinates is not None or not one.points:
            continue
        if any(period is None for period, _ in one.points):
            continue
        matches = [
            key
            for key, values in held.items()
            if all(values.get(period) == value for period, value in one.points)
        ]
        if len(matches) == 1:
            found[index] = matches[0]
    claimed = Counter(found.values())
    return [
        replace(one, coordinates=dict(found[index]))
        if index in found and claimed[found[index]] == 1
        else one
        for index, one in enumerate(series)
    ]


def card_result(
    series: Sequence[DrawnSeries],
    *,
    state_values_for: Sequence[str] = (),
    state_span: bool = False,
    state_period: str | None = None,
    on_canvas: Collection[str] = (),
    cut: TopMembers | None = None,
) -> dict[str, Any]:
    """What `make_charts` returns of a drawn card beside its title.

    `holds` is `card_holds`. `reply_figures` are the values a reply may
    state, each a series label, a value, its unit where the series has one
    and its period where the point has one:

    - A series named in `state_values_for` gives its last value; its first
      as well when `state_span`; and when `state_period` names a period, its
      value there in place of the last. First and last are by period
      (`series_ends`).
    - A card of one series gives the same with no name, and with
      `state_period` its last value beside the one at that period.
    - A card of several series that share one period and one unit also gives
      its largest and smallest series, marked, with the members the mark
      was chosen among (`among`: the series drawn, and on a cut card the
      members with a value they were cut from). Every series tying for the
      largest or for the smallest is given with that mark.
    - A redraw gives only the series named. A card every series of which was
      on the canvas as the turn began (`on_canvas`, series ids) has nothing
      new to state, and a card drawing any other series, or one with no id,
      is a new card.

    A name matches a series on its label or on the value of a coordinate
    that tells the card's series apart, whatever its case. A name matching
    none is listed under `not_in_card`, or under `not_matched` while a series
    of the card states no coordinates, since it may be that series. A stated
    series with no value at `state_period`, or with no value to give at all,
    is listed under `no_value`, with the period asked of it.
    """
    result: dict[str, Any] = {"holds": card_holds(series, cut=cut)}
    chosen: list[DrawnSeries] = []
    unmatched: list[str] = []
    for name in state_values_for:
        named = _named(series, name)
        if not named:
            unmatched.append(name)
        chosen += [one for one in named if one not in chosen]
    redrawn = bool(series) and all(one.id in on_canvas for one in series)
    if len(series) == 1 and not chosen and not redrawn:
        chosen = list(series)
    figures: list[dict[str, Any]] = []
    no_value: list[dict[str, str]] = []
    for one in chosen:
        stated = _stated_points(
            one, state_span=state_span, state_period=state_period, sole=len(series) == 1
        )
        figures += [point_figure(one, point) for point in stated]
        if not stated or state_period not in {None, *(period for period, _ in stated)}:
            no_value.append(
                {"series": one.label} | ({"period": state_period} if state_period else {})
            )
    extremes = {} if redrawn else _extremes(series)
    among = _among(len(_compared(series)), cut=cut)
    for mark, tied in extremes.items():
        for one in tied:
            figure = point_figure(one, one.points[0])
            if figure in figures:
                figure = figures[figures.index(figure)]
            else:
                figures.append(figure)
            figure.update(mark=mark, among=among)
    result["reply_figures"] = figures
    if unmatched:
        placed = all(one.coordinates is not None for one in series)
        result["not_in_card" if placed else "not_matched"] = unmatched
    if no_value:
        result["no_value"] = no_value
    return result


def card_holds(series: Sequence[DrawnSeries], *, cut: TopMembers | None = None) -> dict[str, Any]:
    """What a card covers, with no value in it: its series count, the first
    and last period any of its series holds a value at (ordered as
    `series_ends` orders a series' points), the unit its series
    share, and its series labels in drawn order. Every count and relation is
    taken over every drawn series:

    - `cut`, on a card cut to the largest members of a dimension: how many it
      draws, of how many members with a value at which period, and how many
      it leaves out as group totals, unallocated amounts or members of another
      level (`other_kinds`) and for having no value at that period (`unvalued`).
    - `direction`, on a card of several series: how many of those holding
      more than one period end higher than, lower than or level with their
      own first value by period (`series_ends`), and among how many.
    - `series_order`, on a card of several series sharing one period and one
      unit: `highest_first` or `lowest_first` when the labels are listed in
      that order of their values, series of one value in either order.
    """
    holds: dict[str, Any] = {"series_count": len(series)}
    periods = [period for one in series for period, _ in one.points if period is not None]
    if periods:
        first, last = _first_and_last(periods)
        holds["first_period"], holds["last_period"] = periods[first], periods[last]
    if (unit := _shared_unit(series)) is not None:
        holds["unit"] = unit
    holds["series"] = [one.label for one in series]
    if cut is not None:
        holds["cut"] = {
            "drawn": len(cut.members),
            "with_a_value": cut.holding,
            "period": cut.period,
            "other_kinds": cut.other_kinds,
            "unvalued": len(cut.unvalued),
        }
    spans = [
        ends
        for one in series
        if len({period for period, _ in one.points}) > 1 and (ends := series_ends(one))
    ]
    if len(series) > 1 and spans:
        ends = [(first[1], last[1]) for first, last in spans]
        holds["direction"] = {
            "higher_at_last": sum(last > first for first, last in ends),
            "lower_at_last": sum(last < first for first, last in ends),
            "unchanged": sum(last == first for first, last in ends),
            "among": _among(len(spans), cut=cut),
        }
    if order := _series_order(series):
        holds["series_order"] = order
    return holds


def _named(series: Sequence[DrawnSeries], name: str) -> list[DrawnSeries]:
    """The series `name` names: those labelled it, or holding it as the value
    of a coordinate that not every series of the card holds at that value."""
    wanted = name.strip().casefold()
    placed = [one.coordinates for one in series if one.coordinates is not None]

    def tells_apart(key: str, value: str) -> bool:
        return len(series) == 1 or any(held.get(key) != value for held in placed)

    return [
        one
        for one in series
        if wanted == one.label.casefold()
        or any(
            wanted == value.casefold() and tells_apart(key, value)
            for key, value in (one.coordinates or {}).items()
        )
    ]


def _stated_points(
    series: DrawnSeries, *, state_span: bool, state_period: str | None, sole: bool
) -> list[Point]:
    """The points of `series` a reply states, in period order: its last, or
    the one at `state_period` in its place; before it its first when
    `state_span`; and after the one at `state_period` its last when
    `state_span` or when the card draws no other series (`sole`). First and
    last are `series_ends`, and a period two of its points share gives no
    point."""
    ends = series_ends(series)
    at = [point for point in series.points if point[0] == state_period]
    stated = [ends[0]] if ends and state_span else []
    if state_period is not None and len(at) == 1:
        stated += at
    if ends and (state_period is None or state_span or sole):
        stated.append(ends[1])
    return list(dict.fromkeys(stated))


def point_figure(series: DrawnSeries, point: Point) -> dict[str, Any]:
    """`point` of `series` as an entry of `reply_figures`: the series label,
    the value, the unit where the series has one and the period where the
    point has one."""
    period, value = point
    figure: dict[str, Any] = {"series": series.label, "value": value}
    if series.unit is not None:
        figure["unit"] = series.unit
    if period is not None:
        figure["period"] = period
    return figure


def _shared_unit(series: Sequence[DrawnSeries]) -> Any:
    """The unit every series holds, or None when they differ or hold none."""
    units = [one.unit for one in series]
    return units[0] if units and all(unit == units[0] for unit in units) else None


def _compared(series: Sequence[DrawnSeries]) -> list[DrawnSeries]:
    """The series holding a value, when a card of several series draws each
    at one point, all at one period and in one unit, and their values differ;
    else none. Only such values rank against each other, and values with no
    unit are not known to be of one kind."""
    valued = [one for one in series if one.points]
    periods = {period for one in valued for period, _ in one.points}
    units = [one.unit for one in valued]
    if (
        len(valued) < 2
        or any(len(one.points) != 1 for one in valued)
        or len(periods) != 1
        or None in periods
        or any(unit is None or unit != units[0] for unit in units)
        or len({one.points[0][1] for one in valued}) < 2
    ):
        return []
    return valued


def _extremes(series: Sequence[DrawnSeries]) -> dict[str, list[DrawnSeries]]:
    """The largest and the smallest of the series that rank against each
    other (`_compared`): every series tying for each, in drawn order."""
    compared = _compared(series)
    if not compared:
        return {}
    values = [one.points[0][1] for one in compared]
    return {
        mark: [one for one in compared if one.points[0][1] == extreme]
        for mark, extreme in (("largest", max(values)), ("smallest", min(values)))
    }


def _among(drawn: int, *, cut: TopMembers | None) -> dict[str, int]:
    """The members a count or an extreme describes. They are `drawn` series of
    the card, and on a cut card the members with a value they were cut from."""
    among = {"drawn": drawn}
    return among if cut is None else among | {"with_a_value": cut.holding}


def _series_order(series: Sequence[DrawnSeries]) -> str | None:
    compared = _compared(series)
    if len(compared) != len(series):
        return None
    values = [one.points[0][1] for one in compared]
    if values == sorted(values, reverse=True):
        return "highest_first"
    if values == sorted(values):
        return "lowest_first"
    return None
