"""A line or bar chart as a PNG, for a host that cannot draw the MCP App view.

`chart_png` reads the envelope's `app_view.ChartModel`, the one the view
draws, lays it out as an SVG (`chart_svg`), and rasterises that with resvg.
It draws the view's card face: the title, the caption, the legend of
checkered swatches, the plot, and the footer's source line beside the
"Built with" mark. The plot follows the view's Prefab chart: series take
`theme.SERIES_COLOURS` in order, lines are 2 px wide without dots, bars have
4 px corners, horizontal grid lines sit at each value tick, and there are no
axis lines. Value ticks read in `Intl.NumberFormat`'s en-US compact form
("150", "1.5K"). Where the component authors no bound on its value axis they
follow Recharts' tick rule, from zero to a rounded step past the largest
value. An authored bound (`ChartModel.axis_min`, `axis_max`) is the axis's end
and a tick. An open side keeps zero or the data. The ticks between sit at
multiples of a round step, coarse enough that no two tick labels read the
same (`_authored_ticks`). Prefab's charts take no bounds, so there the image
and the view differ: the view's axis spans zero and the data. A bar starts at
zero, or at the end of the axis nearest zero where the bounds exclude it. A
point with only a status draws nothing, leaving a gap in its line or no bar.
A point with no value on either side draws as a dot, which the view leaves
out, since the image is all such a host sees. An entity-axis bar chart lays
out horizontally under the view's rule, with its categories down the side and
no value ticks.

There is no image for a log value axis, which the view draws as a table, or
for a member that is not a line or bar chart.

The layout is 500 px wide in the view's type sizes, rendered at twice that,
so a tick label is 16 px high in the PNG. The plot is 200 px high, shorter
than the view's 300, so a chart with a one-line title and caption is
1000 x 620, or 1000 x 678 with a legend.

The render is CPU-bound, and resvg holds the GIL while it rasterises, so
`chart_worker` runs `chart_png` in a worker process, off the server's event
loop. Its cost grows with the pixel count and the glyphs and marks it draws,
and the limits bound the work per render: a chart with more than 8 series, a
horizontal bar chart with more than 24 bars, a vertical one with more than
200, a line chart with more than 2400 points counted across its series, or
a layout taller than 600 px (1200 in the PNG) has no image, and the host
keeps the text projection. So does a chart whose value ticks leave the float
range, whose authored bounds span nothing, whose two ends carry the same
tick label, or whose tick labels would take more than 40% of the width. Timed
through the worker on an Apple M5 Pro, with a two-line title, 24 entity bars
with 40-character labels take 33 ms, 200 period bars 39 ms, 2400 points on
one line 39 ms, and 300 periods of 8 lines 66 ms. The slowest measured is
those 8 lines under a caption that brings the layout to its 600 px limit,
at 105 ms.

resvg loads only the vendored Outfit under `fonts/`, with system fonts
skipped, so the PNG is the same on any machine, including the runtime image,
which has no fonts. Label widths come from the same TTFs' advance widths, so
a wrapped title or truncated label fits as drawn. A character the fonts lack
draws as an empty box.
"""

from __future__ import annotations

import bisect
import functools
import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import resvg_py
from fontTools.ttLib import TTFont

from presentation_mcp.app_view import (
    ChartModel,
    PlottedSeries,
    chart_model,
    lone_points,
    plots_as_chart,
)
from presentation_mcp.theme import DFO_LOGO

FONTS_DIR = Path(__file__).with_name("fonts")
_OUTFIT = {300: "Outfit-Light.ttf", 400: "Outfit-Regular.ttf", 700: "Outfit-Bold.ttf"}

WIDTH = 500
ZOOM = 2
_PAD = 16
_GAP = 16
_PLOT_HEIGHT = 200
# Recharts' plot margin above and to the right of the axes.
_PLOT_MARGIN = 5
_TICK_MARGIN = 8
_TICK_COUNT = 5
# Recharts' `minTickGap`: the least space between two category labels.
_MIN_TICK_GAP = 5
_BAR_RADIUS = 4
_BAR_GAP = 4
# The radius of the dot a line draws for a point with no neighbour.
_DOT_RADIUS = 2.5
_BAR_CATEGORY_GAP = 0.1
# The limits past which a chart has no image. `_MAX_HEIGHT` is in layout px.
_MAX_SERIES = 8
# A horizontal chart's 195 px of bands give each of 24 bars at least the
# 8 px line of its label.
_MAX_ACROSS = 24
_MAX_BARS = 200
_MAX_POINTS = 2400
_MAX_HEIGHT = 600
# The share of the width a horizontal chart's category labels may take.
_MAX_GUTTER = 0.4
_SWATCH = 10
_SWATCH_TILE = 6
_LOGO_WIDTH = 26

_FOREGROUND = "#484842"
_MUTED = "#5e5e52"
_CAPTION = "#abb4b7"
_GRID = "#ececdb"
_BACKGROUND = "#ffffff"

# The compact form's letters, by the power of ten each stands for.
_COMPACT_STEPS = ((12, "T"), (9, "B"), (6, "M"), (3, "K"))
# Enough digits to hold exactly the difference of any two floats' shortest
# forms, which span 17 significant digits between 1e-324 and 1e309.
_PRECISION = 700
_NOT_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


@dataclass(frozen=True, slots=True)
class _Type:
    """A text style: Outfit weight, size, line height, and letter spacing in em."""

    weight: int
    size: float
    line_height: float = 1.3
    spacing: float = -0.01


_TITLE = _Type(300, 12, line_height=1.1)
_CAPTION_TYPE = _Type(400, 10)
_LEGEND = _Type(400, 10)
_TICK = _Type(700, 8, line_height=1.0, spacing=0.0)
_FOOTER = _Type(300, 12)
_BUILT_WITH = _Type(300, 8, line_height=1.1)


def chart_png(envelope: Mapping[str, Any]) -> bytes | None:
    """The envelope's chart as PNG bytes, 1000 px wide, or None for a log
    value axis, a member that is not a line or bar chart, or a chart past
    the size limits."""
    svg = chart_svg(envelope)
    if svg is None:
        return None
    return _rasterise(svg)


def warm() -> None:
    """Load what the first render would: Outfit's metrics in each weight,
    and resvg with the fonts."""
    for weight in _OUTFIT:
        _metrics(weight)
    _rasterise(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" font-family="Outfit">'
        '<text y="8" font-size="8">0</text></svg>'
    )


def _rasterise(svg: str) -> bytes:
    """`svg` as PNG bytes at `ZOOM`, with the vendored Outfit as the only font."""
    return resvg_py.svg_to_bytes(
        svg_string=svg,
        zoom=ZOOM,
        skip_system_fonts=True,
        font_dirs=[str(FONTS_DIR)],
        font_family="Outfit",
    )


def chart_svg(envelope: Mapping[str, Any]) -> str | None:
    """The envelope's chart as an SVG document 500 px wide, or None where
    `chart_png` gives none."""
    chart = chart_model(envelope) if plots_as_chart(envelope) else None
    if chart is None or not _within_limits(chart):
        return None
    parts: list[str] = []
    inner = WIDTH - 2 * _PAD
    y = float(_PAD)

    for line in _wrap(chart.title, _TITLE, inner):
        parts.append(_text(line, _PAD, y, _TITLE, _MUTED))
        y += _line(_TITLE)
    if chart.caption is not None:
        y += 4
        for line in _wrap(chart.caption, _CAPTION_TYPE, inner):
            parts.append(_text(line, _PAD, y, _CAPTION_TYPE, _MUTED))
            y += _line(_CAPTION_TYPE)
    y += _GAP

    if chart.legend:
        y = _legend(parts, chart.series, y, inner) + _GAP

    if chart.horizontal:
        plot_bottom = _horizontal_bars(parts, chart, y, inner)
    else:
        plot_bottom = _vertical_chart(parts, chart, y, inner)
    if plot_bottom is None:
        return None
    y = _footer(parts, chart.source_line, plot_bottom + _GAP, inner) + _PAD

    height = math.ceil(y)
    if height > _MAX_HEIGHT:
        return None
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="Outfit">'
        f'<rect width="{WIDTH}" height="{height}" fill="{_BACKGROUND}"/>'
        f"{''.join(parts)}</svg>"
    )


def _within_limits(chart: ChartModel) -> bool:
    """Whether the chart's series and marks are few enough to draw."""
    marks = len(chart.rows) * len(chart.series)
    if chart.horizontal:
        limit = _MAX_ACROSS
    elif chart.kind == "bar_chart":
        limit = _MAX_BARS
    else:
        limit = _MAX_POINTS
    return len(chart.series) <= _MAX_SERIES and marks <= limit


# --- parts --------------------------------------------------------------------


def _legend(parts: list[str], series: Sequence[PlottedSeries], top: float, inner: float) -> float:
    """The legend's rows of checkered swatches and labels from `top`, wrapped
    to `inner`. Returns the bottom of the last row."""
    row = _line(_LEGEND)
    x, y = float(_PAD), top
    for i, one in enumerate(series):
        width = _SWATCH + 4 + _width(one.label, _LEGEND)
        if x > _PAD and x + width > _PAD + inner:
            x, y = float(_PAD), y + row
        tile = _SWATCH_TILE / 2
        parts.append(
            f'<defs><pattern id="swatch{i}" width="{_SWATCH_TILE}" height="{_SWATCH_TILE}" '
            f'patternUnits="userSpaceOnUse" x="{_n(x)}" y="{_n(y)}">'
            f'<rect width="{_SWATCH_TILE}" height="{_SWATCH_TILE}" fill="{one.colour}"/>'
            f'<rect x="{_n(tile)}" width="{_n(tile)}" height="{_n(tile)}" fill="{one.light_colour}"/>'
            f'<rect y="{_n(tile)}" width="{_n(tile)}" height="{_n(tile)}" fill="{one.light_colour}"/>'
            "</pattern></defs>"
        )
        swatch_y = y + (row - _SWATCH) / 2
        parts.append(
            f'<rect x="{_n(x)}" y="{_n(swatch_y)}" width="{_SWATCH}" height="{_SWATCH}" '
            f'rx="2" fill="url(#swatch{i})"/>'
        )
        parts.append(_text(one.label, x + _SWATCH + 4, y, _LEGEND, _FOREGROUND))
        x += width + 10
    return y + row


def _vertical_chart(parts: list[str], chart: ChartModel, top: float, inner: float) -> float | None:
    """A line chart or vertical bar chart with value ticks down the left and
    category labels along the bottom. Returns the bottom of the labels, or
    None where the ticks leave the float range or span nothing, two of
    their labels read the same, or the labels are wider than `_MAX_GUTTER`
    of `inner`."""
    ticks = _value_ticks(_values(chart), lower=chart.axis_min, upper=chart.axis_max)
    if not _drawable(ticks):
        return None
    labels = [_tick_text(tick) for tick in ticks]
    if len(set(labels)) < len(labels):
        return None
    gutter = max(_width(label, _TICK) for label in labels)
    if gutter > inner * _MAX_GUTTER:
        return None
    left = _PAD + gutter + _TICK_MARGIN
    right = _PAD + inner - _PLOT_MARGIN
    plot_top = top + _PLOT_MARGIN
    plot_bottom = top + _PLOT_HEIGHT - _TICK_MARGIN - _line(_TICK)
    scale = _scale(ticks, plot_bottom, plot_top)

    for tick, label in zip(ticks, labels, strict=True):
        at = scale(tick)
        parts.append(
            f'<line x1="{_n(left)}" x2="{_n(right)}" y1="{_n(at)}" y2="{_n(at)}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )
        parts.append(
            _text(label, left - _TICK_MARGIN, at - _line(_TICK) / 2, _TICK, _FOREGROUND, "end")
        )

    count = len(chart.rows)
    if chart.kind == "line_chart":
        step = (right - left) / (count - 1) if count > 1 else 0.0
        centres = [left + i * step if count > 1 else (left + right) / 2 for i in range(count)]
        for one in chart.series:
            parts.extend(_line_path(chart.rows, one, centres, scale))
    else:
        band = (right - left) / max(count, 1)
        centres = [left + (i + 0.5) * band for i in range(count)]
        parts.extend(_vertical_bars(chart, centres, band, scale, _baseline(ticks, scale)))

    names = [str(row[chart.x_key]) for row in chart.rows]
    label_top = plot_bottom + _TICK_MARGIN
    parts.extend(
        _text(names[i], centres[i], label_top, _TICK, _FOREGROUND, "middle")
        for i in _spaced(names, centres)
    )
    return top + _PLOT_HEIGHT


def _line_path(
    rows: Sequence[Mapping[str, Any]],
    series: PlottedSeries,
    xs: Sequence[float],
    scale: _Scale,
) -> list[str]:
    """The series' line, broken where a row has no value for it, with a dot
    for a point that has no value on either side, which a line cannot draw."""
    runs: list[list[tuple[float, float]]] = [[]]
    for row, x in zip(rows, xs, strict=True):
        value = row.get(series.key)
        if value is None:
            if runs[-1]:
                runs.append([])
        else:
            runs[-1].append((x, scale(value)))
    commands = [
        f"{'L' if i else 'M'}{_n(x)} {_n(y)}"
        for run in runs
        if len(run) > 1
        for i, (x, y) in enumerate(run)
    ]
    marks = [
        f'<circle cx="{_n(x)}" cy="{_n(scale(row[series.key]))}" r="{_DOT_RADIUS}" '
        f'fill="{series.colour}"/>'
        for row, x, lone in zip(rows, xs, lone_points(rows, series.key), strict=True)
        if lone
    ]
    if commands:
        marks.insert(
            0,
            f'<path d="{" ".join(commands)}" fill="none" stroke="{series.colour}" '
            'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>',
        )
    return marks


def _vertical_bars(
    chart: ChartModel, centres: Sequence[float], band: float, scale: _Scale, base: float
) -> list[str]:
    """Each row's bars side by side in its band, rising from `base`, the
    baseline's y. Where the gaps between the bars would take the whole band,
    the bars touch, as Recharts' `getBarSizeList` lays them out."""
    count = len(chart.series)
    offset = band * _BAR_CATEGORY_GAP
    room = band - 2 * offset
    gap = _BAR_GAP if room > _BAR_GAP * (count - 1) else 0.0
    size = (room - gap * (count - 1)) / count
    bars = []
    for row, centre in zip(chart.rows, centres, strict=True):
        x = centre - band / 2 + offset
        for one in chart.series:
            value = row.get(one.key)
            if value is not None:
                at = scale(value)
                bars.append(_bar(x, min(at, base), size, abs(base - at), one.colour))
            x += size + gap
    return bars


def _horizontal_bars(parts: list[str], chart: ChartModel, top: float, inner: float) -> float | None:
    """An entity-axis bar chart laid out across: category labels down the
    left, truncated to fit, bars from the baseline, and no value ticks,
    since the view's horizontal chart hides its value axis. Returns its
    bottom, or None where the value span leaves the float range or is empty."""
    ticks = _value_ticks(_values(chart), lower=chart.axis_min, upper=chart.axis_max)
    if not _drawable(ticks):
        return None
    names = [str(row[chart.x_key]) for row in chart.rows]
    gutter = min(max((_width(name, _TICK) for name in names), default=0), inner * _MAX_GUTTER)
    left = _PAD + gutter + _TICK_MARGIN
    right = _PAD + inner - _PLOT_MARGIN
    plot_top = top + _PLOT_MARGIN
    band = (_PLOT_HEIGHT - _PLOT_MARGIN) / max(len(names), 1)
    scale = _scale(ticks, left, right)
    base = _baseline(ticks, scale)
    (one,) = chart.series
    offset = band * _BAR_CATEGORY_GAP
    for i, (row, name) in enumerate(zip(chart.rows, names, strict=True)):
        band_top = plot_top + i * band
        label = _truncate(name, _TICK, gutter)
        label_top = band_top + (band - _line(_TICK)) / 2
        parts.append(_text(label, left - _TICK_MARGIN, label_top, _TICK, _FOREGROUND, "end"))
        value = row.get(one.key)
        if value is not None:
            at = scale(value)
            parts.append(
                _bar(
                    min(at, base), band_top + offset, abs(at - base), band - 2 * offset, one.colour
                )
            )
    return top + _PLOT_HEIGHT


def _footer(parts: list[str], source_line: str, top: float, inner: float) -> float:
    """The card's footer: the source line, clear of the "Built with" mark at
    the right. Returns its bottom."""
    logo_height = _LOGO_WIDTH * 22 / 58
    mark = _width("Built with", _BUILT_WITH) + 4 + _LOGO_WIDTH
    source = _truncate(source_line, _FOOTER, inner - mark - _GAP)
    parts.append(_text(source, _PAD, top, _FOOTER, _CAPTION))
    row = _line(_FOOTER)
    logo_x = _PAD + inner - _LOGO_WIDTH
    logo_y = top + (row - logo_height) / 2
    parts.append(
        _text(
            "Built with",
            logo_x - 4,
            logo_y + logo_height - _line(_BUILT_WITH),
            _BUILT_WITH,
            _CAPTION,
            "end",
        )
    )
    parts.append(
        DFO_LOGO.replace(
            "<svg ",
            f'<svg x="{_n(logo_x)}" y="{_n(logo_y)}" width="{_LOGO_WIDTH}" '
            f'height="{_n(logo_height)}" color="{_CAPTION}" ',
            1,
        )
    )
    return top + row


def _bar(x: float, y: float, width: float, height: float, colour: str) -> str:
    """A bar with the view's rounded corners, which SVG clamps to half its
    width and height."""
    return (
        f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(width)}" height="{_n(height)}" '
        f'rx="{_BAR_RADIUS}" fill="{colour}"/>'
    )


# --- scales and ticks ---------------------------------------------------------


type _Scale = Callable[[float], float]


def _values(chart: ChartModel) -> list[float]:
    return [
        row[one.key] for row in chart.rows for one in chart.series if row.get(one.key) is not None
    ]


def _scale(ticks: Sequence[float], start: float, end: float) -> _Scale:
    """A linear map from the ticks' span onto `start`..`end`. Each term is
    halved, so a span wider than the largest float stays finite."""
    low, high = ticks[0] / 2, ticks[-1] / 2

    def at(value: float) -> float:
        return start + (value / 2 - low) / (high - low) * (end - start)

    return at


def _drawable(ticks: Sequence[float]) -> bool:
    """Whether the ticks span a finite range of floats that `_scale` can
    divide by: its ends differ once each is halved."""
    return math.isfinite(ticks[0]) and math.isfinite(ticks[-1]) and ticks[0] / 2 < ticks[-1] / 2


def _baseline(ticks: Sequence[float], scale: _Scale) -> float:
    """Where a bar starts: zero, or the end of the axis nearest zero where
    the ticks exclude it."""
    return scale(min(max(0.0, ticks[0]), ticks[-1]))


def _value_ticks(
    values: Sequence[float], *, lower: float | None = None, upper: float | None = None
) -> list[float]:
    """The value axis's ticks, in order, for a vertical or a horizontal
    chart. `lower` and `upper` are the authored bounds, None for a side left
    to the renderer. The axis spans the first tick to the last."""
    if lower is None and upper is None:
        return _recharts_ticks(values)
    return _authored_ticks(values, lower, upper)


def _authored_ticks(
    values: Sequence[float], lower: float | None, upper: float | None
) -> list[float]:
    """The ticks of an axis with an authored bound. An authored bound is the
    axis's end exactly. An open side is zero or the furthest value past it,
    moved outward to a multiple of the step, or one unit past the authored
    bound where neither lies beyond it. That step is `_format_step` of a
    quarter of the span. Between the ends, each multiple of the step at
    least half a step from both is a tick, so labels stay clear of each
    other, and the axis has two to seven ticks.

    No two ticks carry the same label. Where `_tick_text` would print two
    alike, the step becomes the next of 1, 2, and 5 times a power of ten
    (`_coarser`) and an open side moves outward to a multiple of it, until
    every label differs. An open side is therefore a multiple of the step
    its axis is drawn with. Two authored ends stay where they are, so where
    they share a label they come back alone, and `_vertical_chart` refuses
    them.

    Bounds that are not finite, or that span nothing, come back as they
    are, and `_drawable` refuses them, as it does an open side past the
    largest float. The arithmetic is `_recharts_ticks`'s: decimal at
    `_PRECISION`, quotients rounded up, so an open side covers every value."""
    low = min([0.0, *values]) if lower is None else float(lower)
    high = max([0.0, *values]) if upper is None else float(upper)
    authored = lower is not None and upper is not None
    if not (math.isfinite(low) and math.isfinite(high)) or (authored and low >= high):
        return [low, high]
    with localcontext(prec=_PRECISION, rounding=ROUND_CEILING):
        low_at, high_at = Decimal(repr(low)), Decimal(repr(high))
        if low_at >= high_at:
            if upper is None:
                high_at = low_at + 1
            else:
                low_at = high_at - 1
        step = _format_step((high_at - low_at) / (_TICK_COUNT - 1), 0)
        while True:
            first = low_at if lower is not None else -math.ceil(-low_at / step) * step
            last = high_at if upper is not None else math.ceil(high_at / step) * step
            multiples = (
                i * step for i in range(math.ceil(first / step), math.floor(last / step) + 1)
            )
            between = [at for at in multiples if at - first >= step / 2 and last - at >= step / 2]
            ticks = [float(at) for at in (first, *between, last)]
            if not _drawable(ticks) or (authored and not between):
                return ticks
            if len({_tick_text(tick) for tick in ticks}) == len(ticks):
                return ticks
            step = _coarser(step)


def _coarser(step: Decimal) -> Decimal:
    """The least of 1, 2, and 5 times a power of ten that is above `step`."""
    power = Decimal(1).scaleb(step.adjusted())
    return next(power * times for times in (2, 5, 10) if power * times > step)


def _recharts_ticks(values: Sequence[float]) -> list[float]:
    """The value axis's ticks as Recharts places them for its default
    `[0, "auto"]` domain and five ticks. The span holds zero and every
    value, at a step of a multiple of 0.05 or 0.1 of a power of ten, with
    any ticks to spare added above zero, or below it where no value is
    positive.

    The step is worked out in decimal at `_PRECISION` digits, which holds
    any two floats' difference exactly, and quotients round up, so the
    ticks cover every value. A tick past the largest float comes back
    infinite, and `_drawable` refuses it."""
    low = min([0.0, *values])
    high = max([0.0, *values])
    if low == high:
        high = 1.0
    with localcontext(prec=_PRECISION, rounding=ROUND_CEILING):
        low_at, high_at = Decimal(repr(low)), Decimal(repr(high))
        rough = (high_at - low_at) / (_TICK_COUNT - 1)
        correction = 0
        while True:
            step = _format_step(rough, correction)
            below = math.ceil(-low_at / step)
            above = math.ceil(high_at / step)
            spare = _TICK_COUNT - (below + above + 1)
            if spare >= 0:
                break
            correction += 1
        if high > 0:
            above += spare
        else:
            below += spare
        return [float((i - below) * step) for i in range(below + above + 1)]


def _format_step(rough: Decimal, correction: int) -> Decimal:
    """Recharts' `getFormatStep`: `rough` rounded up to a multiple of 0.05
    of its power of ten, or of 0.1 below ten, plus `correction` of those."""
    digits = rough.adjusted() + 1
    power = Decimal(1).scaleb(digits)
    ratio_step = Decimal("0.1") if digits == 1 else Decimal("0.05")
    multiple = (rough / power / ratio_step).to_integral_value(rounding=ROUND_CEILING)
    return (multiple + correction) * ratio_step * power


def _tick_text(value: float) -> str:
    """`value` as `toLocaleString("en-US", {notation: "compact"})` writes it,
    the view's tick format. It scales to K, M, B, or T, then rounds to a whole
    number from ten up and to two significant digits below."""
    if value == 0:
        return "0"
    with localcontext(prec=_PRECISION):
        number = Decimal(repr(value))
        suffix = ""
        for power, letter in _COMPACT_STEPS:
            if abs(number).adjusted() >= power:
                number, suffix = number.scaleb(-power), letter
                break
        places = 0 if abs(number) >= 10 else 1 - number.adjusted()
        rounded = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    text = f"{rounded:f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def _spaced(labels: Sequence[str], centres: Sequence[float]) -> list[int]:
    """The indexes of the category labels to draw, as Recharts'
    `preserveEnd` picks them. It keeps the last, then each earlier one that
    clears the one after it by `_MIN_TICK_GAP`."""
    kept: list[int] = []
    edge = math.inf
    for i in reversed(range(len(labels))):
        half = _width(labels[i], _TICK) / 2
        if centres[i] + half + _MIN_TICK_GAP <= edge:
            kept.append(i)
            edge = centres[i] - half
    return kept[::-1]


# --- text ---------------------------------------------------------------------


def _text(
    content: str, x: float, top: float, style: _Type, fill: str, anchor: str = "start"
) -> str:
    """`content` on one line whose box starts at `top`, placed as CSS places
    a line of text, with the font's ascent and descent centred in the line
    height."""
    metrics = _metrics(style.weight)
    content_height = (metrics.ascent + metrics.descent) * style.size
    baseline = top + (_line(style) - content_height) / 2 + metrics.ascent * style.size
    spacing = f' letter-spacing="{_n(style.spacing * style.size)}"' if style.spacing else ""
    return (
        f'<text x="{_n(x)}" y="{_n(baseline)}" font-weight="{style.weight}" '
        f'font-size="{_n(style.size)}"{spacing} fill="{fill}" text-anchor="{anchor}">'
        f"{escape(_NOT_XML.sub('', content))}</text>"
    )


def _line(style: _Type) -> float:
    return style.size * style.line_height


def _wrap(text: str, style: _Type, width: float) -> list[str]:
    """`text` broken at spaces into lines no wider than `width`, a word
    wider than that alone on its line."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and _width(candidate, style) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return [*lines, current] if current else lines


def _truncate(text: str, style: _Type, width: float) -> str:
    """`text`, cut and ended with an ellipsis where it is wider than `width`."""
    ends = list(itertools.accumulate(_width(char, style) for char in text))
    if not ends or ends[-1] <= width:
        return text
    cut = bisect.bisect_right(ends, width - _width("…", style))
    return f"{text[:cut].rstrip()}…"


def _width(text: str, style: _Type) -> float:
    """`text`'s advance width in `style`, from the Outfit TTF's `hmtx`."""
    metrics = _metrics(style.weight)
    ems = sum(metrics.advances.get(ord(char), metrics.missing) for char in text)
    return (ems + style.spacing * len(text)) * style.size


@dataclass(frozen=True, slots=True)
class _Metrics:
    """An Outfit weight's metrics in ems: the advance per code point, the
    advance of its missing-glyph box, and its ascent and descent."""

    advances: dict[int, float]
    missing: float
    ascent: float
    descent: float


@functools.cache
def _metrics(weight: int) -> _Metrics:
    font = TTFont(FONTS_DIR / "outfit" / _OUTFIT[weight])
    # fontTools sets a table's fields as it decompiles it, so its classes
    # declare none for a type checker.
    units = font["head"].unitsPerEm  # ty: ignore[unresolved-attribute]
    widths = font["hmtx"]
    cmap = font.getBestCmap()
    assert cmap is not None, f"{_OUTFIT[weight]} has no Unicode character map"
    return _Metrics(
        advances={code: widths[name][0] / units for code, name in cmap.items()},
        missing=widths[".notdef"][0] / units,
        ascent=font["hhea"].ascent / units,
        descent=-font["hhea"].descent / units,
    )


def _n(value: float) -> str:
    """`value` to two decimals without trailing zeros, for SVG attributes."""
    return f"{value:.2f}".rstrip("0").rstrip(".")
