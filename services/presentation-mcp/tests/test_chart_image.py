"""Chart image tests: the SVG a line or bar chart lays out, and its PNG.

Inputs are dfo-spec-kit's catalogue fixtures (`stub_assembler.resolved_revision`).
Each test reads the SVG `chart_svg` returns as XML, or the PNG `chart_png`
returns by its signature and IHDR chunk.
"""

from __future__ import annotations

import itertools
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import pytest
import resvg_py
from presentation_mcp import chart_image
from presentation_mcp.app_view import chart_model
from presentation_mcp.chart_image import (
    _PAD,
    _PLOT_MARGIN,
    FONTS_DIR,
    WIDTH,
    ZOOM,
    _drawable,
    _rasterise,
    _tick_text,
    _value_ticks,
    _vertical_chart,
    chart_png,
    chart_svg,
)
from presentation_mcp.theme import SERIES_COLOURS, SERIES_COLOURS_LIGHT

from .stub_assembler import resolved_revision

_SVG = "{http://www.w3.org/2000/svg}"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _svg(envelope: dict[str, Any]) -> ET.Element:
    svg = chart_svg(envelope)
    assert svg is not None
    return ET.fromstring(svg)


def _texts(svg: ET.Element) -> list[str]:
    return [node.text or "" for node in svg.iter(f"{_SVG}text")]


def _lines(svg: ET.Element) -> list[ET.Element]:
    """The series' lines: the paths with a stroke, which the logo's lack."""
    return [node for node in svg.iter(f"{_SVG}path") if node.get("stroke")]


def _bars(svg: ET.Element) -> list[dict[str, float | str]]:
    """Each bar's box and fill: the rounded rects outside the legend."""
    return [
        {key: float(node.get(key, "0")) for key in ("x", "y", "width", "height")}
        | {"fill": node.get("fill", "")}
        for node in svg.iter(f"{_SVG}rect")
        if node.get("rx") == "4"
    ]


def _value_ticks_drawn(svg: ET.Element) -> list[tuple[str, float]]:
    """A vertical chart's value ticks from the bottom up: each label, which
    is the bold text anchored at its end, with its grid line's y."""
    labels = [
        node.text or ""
        for node in svg.iter(f"{_SVG}text")
        if node.get("text-anchor") == "end" and node.get("font-weight") == "700"
    ]
    grid = [float(node.get("y1", "0")) for node in svg.iter(f"{_SVG}line")]
    return list(zip(labels, grid, strict=True))


def _ranged(envelope: dict[str, Any], bounds: dict[str, Any] | None) -> dict[str, Any]:
    """`envelope` with `bounds` as its value axis's range, or without a
    range where `bounds` is None."""
    component = envelope["component"]
    axis = component["y_axis" if component["type"] == "line_chart" else "value_axis"]
    axis.pop("range", None)
    if bounds is not None:
        axis["range"] = bounds
    return envelope


def _period_bar_chart() -> dict[str, Any]:
    envelope = resolved_revision("bar_chart")
    envelope["component"]["category_axis"] = {
        "mode": "period",
        "window": {"start": "2024", "end": "2024"},
    }
    return envelope


def _entity_bar_chart(count: int) -> dict[str, Any]:
    """An entity-axis bar chart of `count` countries."""
    envelope = resolved_revision("bar_chart")
    (one,) = envelope["component"]["series"][:1]
    envelope["component"]["series"] = [
        {**one, "id": f"c{i}", "label": f"Country {i}", "points": [{"period": "2024", "value": i}]}
        for i in range(count)
    ]
    return envelope


def _period_chart(member: str, periods: int, series: int) -> dict[str, Any]:
    """A line or bar chart of `series` series over `periods` years."""
    envelope = resolved_revision(member)
    component = envelope["component"]
    (one,) = component["series"][:1]
    points = [{"period": str(1900 + i), "value": 100.0 + i % 7} for i in range(periods)]
    component["series"] = [
        {**one, "id": f"s{i}", "label": f"Series {i}", "points": points} for i in range(series)
    ]
    window = {"start": "1900", "end": str(1900 + periods - 1)}
    if member == "bar_chart":
        component["category_axis"] = {"mode": "period", "window": window}
    else:
        component["x_axis"] = {"window": window}
        component["y_axis"]["range"] = {"min": None, "max": None}
    return envelope


def _png_size(png: bytes) -> tuple[int, int]:
    assert png[:8] == _PNG_SIGNATURE
    length, kind = struct.unpack(">I4s", png[8:16])
    assert (length, kind) == (13, b"IHDR")
    return struct.unpack(">II", png[16:24])


def test_line_chart_draws_its_title_caption_legend_and_ticks() -> None:
    texts = _texts(_svg(resolved_revision("line_chart")))

    assert texts[:3] == [
        "Malaria grant disbursements to Ethiopia",
        "Disbursements, USD millions, constant 2023 prices",
        "Ethiopia",
    ]
    assert {"2015", "2024"} <= set(texts)
    assert "OECD DAC" in texts


def test_the_shipped_line_chart_spans_its_authored_range() -> None:
    svg = _svg(resolved_revision("line_chart"))

    ticks = _value_ticks_drawn(svg)
    assert [label for label, _ in ticks] == ["390", "400", "420", "440", "460"]
    bottom, top = ticks[0][1], ticks[-1][1]
    (line,) = _lines(svg)
    heights = [float(y) for y in re.findall(r"[ML]\S+ (\S+)", line.get("d", ""))]
    # 398.2 and 455.1, the least and largest values, on an axis from 390 to 460.
    assert max(heights) == pytest.approx(bottom + (top - bottom) * 8.2 / 70, abs=0.01)
    assert min(heights) == pytest.approx(bottom + (top - bottom) * 65.1 / 70, abs=0.01)


@pytest.mark.parametrize(
    ("bounds", "labels"),
    [
        ({"min": 390}, ["390", "400", "420", "440", "460"]),
        ({"min": 390, "max": None}, ["390", "400", "420", "440", "460"]),
        ({"max": 460}, ["0", "150", "300", "460"]),
        ({"min": None, "max": 1000}, ["0", "250", "500", "750", "1K"]),
        ({"min": None, "max": None}, ["0", "150", "300", "450", "600"]),
        ({}, ["0", "150", "300", "450", "600"]),
        (None, ["0", "150", "300", "450", "600"]),
    ],
    ids=["min", "min, null max", "max", "null min, max", "both null", "empty", "absent"],
)
def test_a_side_without_an_authored_bound_keeps_zero_or_the_data(
    bounds: dict[str, Any] | None, labels: list[str]
) -> None:
    svg = _svg(_ranged(resolved_revision("line_chart"), bounds))

    assert [label for label, _ in _value_ticks_drawn(svg)] == labels


@pytest.mark.parametrize("bounds", [{"min": None, "max": None}, {}])
@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
def test_a_range_without_a_bound_draws_as_no_range_does(
    member: str, bounds: dict[str, Any]
) -> None:
    absent = chart_svg(_ranged(resolved_revision(member), None))

    assert absent is not None
    assert chart_svg(_ranged(resolved_revision(member), bounds)) == absent


def test_a_horizontal_bar_chart_scales_its_bars_to_the_authored_range() -> None:
    envelope = _ranged(resolved_revision("bar_chart"), {"min": 0, "max": 883.8})
    envelope["component"]["series"][0]["label"] = "Democratic Republic of the Congo"

    first, ghana, _ = _bars(_svg(envelope))

    # 441.9 is half of the authored 883.8, so its bar ends halfway to the
    # plot's right edge.
    right = WIDTH - _PAD - _PLOT_MARGIN
    assert first["x"] == ghana["x"]
    assert first["x"] + first["width"] == pytest.approx((first["x"] + right) / 2, abs=0.02)
    assert ghana["width"] / first["width"] == pytest.approx(225.3 / 441.9, abs=0.001)


@pytest.mark.parametrize("horizontal", [False, True], ids=["vertical", "horizontal"])
@pytest.mark.parametrize(
    ("sign", "bounds"),
    [(1, {"min": 200, "max": 500}), (-1, {"min": -500, "max": -200})],
    ids=["above zero", "below zero"],
)
def test_bars_start_at_the_end_of_a_range_that_excludes_zero(
    horizontal: bool, sign: int, bounds: dict[str, Any]
) -> None:
    envelope = _ranged(resolved_revision("bar_chart"), bounds)
    for one in envelope["component"]["series"]:
        one["points"][0]["value"] *= sign
    if horizontal:
        envelope["component"]["series"][0]["label"] = "Democratic Republic of the Congo"

    svg = _svg(envelope)

    start, size = ("x", "width") if horizontal else ("y", "height")
    bars = _bars(svg)
    # A bar's length is its value's distance from the bound nearest zero:
    # 441.9, 225.3, and 305.6 from 200.
    lengths = [float(bar[size]) for bar in bars]
    assert [length / lengths[0] for length in lengths] == pytest.approx(
        [1, 25.3 / 241.9, 105.6 / 241.9], abs=0.002
    )
    # The bars share the edge at that bound: the left or bottom for a range
    # above zero, drawn towards larger values, and the right or top below it.
    grows_from_start = (sign == 1) == horizontal
    edges = [bar[start] if grows_from_start else bar[start] + bar[size] for bar in bars]
    assert edges == pytest.approx([edges[0]] * 3, abs=0.02)
    if not horizontal:
        ticks = _value_ticks_drawn(svg)
        assert [label for label, _ in ticks] == (
            ["200", "300", "375", "450", "500"]
            if sign == 1
            else ["-500", "-450", "-375", "-300", "-200"]
        )
        assert edges[0] == pytest.approx(ticks[0 if sign == 1 else -1][1], abs=0.02)


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
@pytest.mark.parametrize(
    "bounds", [{"min": 500, "max": 500}, {"min": 600, "max": 500}], ids=["equal", "crossed"]
)
def test_bounds_that_span_nothing_have_no_image(member: str, bounds: dict[str, Any]) -> None:
    envelope = _ranged(resolved_revision(member), bounds)

    assert chart_svg(envelope) is None
    assert chart_png(envelope) is None


def test_a_chart_without_values_draws_a_unit_past_its_one_authored_bound() -> None:
    envelope = _ranged(resolved_revision("line_chart"), {"min": 390})
    (one,) = envelope["component"]["series"]
    one["points"] = [{"period": point["period"], "status": "missing"} for point in one["points"]]

    ticks = _value_ticks_drawn(_svg(envelope))

    assert [label for label, _ in ticks] == ["390", "391"]
    assert _value_ticks([], lower=390) == [390, 391]
    assert _value_ticks([], upper=-10) == [-11, -10]


@pytest.mark.parametrize(
    ("bounds", "labels"),
    [
        ({"min": 12_000, "max": 15_000}, ["12K", "13K", "14K", "15K"]),
        ({"min": 1200, "max": 1300}, ["1.2K", "1.3K"]),
        ({"min": 98, "max": 100}, ["98", "99", "100"]),
        ({"min": 70, "max": 73}, ["70", "71", "72", "73"]),
        ({"min": 0.001, "max": 0.002}, ["0.001", "0.0013", "0.0015", "0.0018", "0.002"]),
    ],
    ids=["12K to 15K", "1.2K to 1.3K", "98 to 100", "70 to 73", "0.001 to 0.002"],
)
def test_a_narrow_authored_range_draws_no_tick_label_twice(
    bounds: dict[str, Any], labels: list[str]
) -> None:
    envelope = _ranged(resolved_revision("line_chart"), bounds)
    middle = (bounds["min"] + bounds["max"]) / 2
    for point in envelope["component"]["series"][0]["points"]:
        point["value"] = middle

    ticks = _value_ticks_drawn(_svg(envelope))

    assert [label for label, _ in ticks] == labels
    assert [at for _, at in ticks] == sorted({at for _, at in ticks}, reverse=True)


@pytest.mark.parametrize(
    ("bounds", "values", "ticks", "labels"),
    [
        ({"min": 12_000}, [12_050, 12_300], [12_000, 12_500], ["12K", "13K"]),
        ({"min": 98}, [99.6], [98, 99, 100], ["98", "99", "100"]),
    ],
    ids=["12K up", "98 up"],
)
def test_an_open_side_ends_on_the_step_its_axis_is_drawn_with(
    bounds: dict[str, Any], values: list[float], ticks: list[float], labels: list[str]
) -> None:
    envelope = _ranged(resolved_revision("line_chart"), bounds)
    points = envelope["component"]["series"][0]["points"]
    for point, value in zip(points, itertools.cycle(values), strict=False):
        point["value"] = value

    assert _value_ticks(values, lower=bounds["min"]) == ticks
    assert [label for label, _ in _value_ticks_drawn(_svg(envelope))] == labels


def test_a_range_whose_ends_read_the_same_has_no_image_unless_it_draws_no_ticks() -> None:
    # 12,000 and 12,400 both read "12K". A horizontal bar chart writes no
    # value ticks, so it draws.
    bounds = {"min": 12_000, "max": 12_400}
    vertical = _ranged(resolved_revision("line_chart"), bounds)
    for point in vertical["component"]["series"][0]["points"]:
        point["value"] = 12_200
    across = _ranged(resolved_revision("bar_chart"), bounds)
    across["component"]["series"][0]["label"] = "Democratic Republic of the Congo"
    for one in across["component"]["series"]:
        one["points"][0]["value"] = 12_200

    assert _value_ticks([12_200], lower=12_000, upper=12_400) == [12_000, 12_400]
    assert chart_svg(vertical) is None
    assert chart_png(vertical) is None
    assert chart_svg(across) is not None


def test_a_line_breaks_at_a_point_with_only_a_status() -> None:
    (line,) = _lines(_svg(resolved_revision("line_chart")))

    assert line.get("stroke") == SERIES_COLOURS[0]
    assert line.get("stroke-width") == "2"
    moves = [command[0] for command in line.get("d", "").split(" ") if command[0].isalpha()]
    assert moves.count("M") == 2
    assert len(moves) == 9


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
def test_series_take_the_palette_in_order_under_the_legend(member: str) -> None:
    envelope = resolved_revision(member) if member == "line_chart" else _period_bar_chart()
    component = envelope["component"]
    (one,) = component["series"][:1]
    component["series"] = [{**one, "label": f"Series {i}"} for i in range(7)]

    svg = _svg(envelope)

    order = [*range(6), 0]
    if member == "line_chart":
        drawn = [line.get("stroke") for line in _lines(svg)]
    else:
        drawn = [bar["fill"] for bar in _bars(svg)]
    assert drawn == [SERIES_COLOURS[i] for i in order]
    patterns = [[rect.get("fill") for rect in pattern] for pattern in svg.iter(f"{_SVG}pattern")]
    assert patterns == [
        [SERIES_COLOURS[i], SERIES_COLOURS_LIGHT[i], SERIES_COLOURS_LIGHT[i]] for i in order
    ]
    assert [f"Series {i}" for i in range(7)] == [t for t in _texts(svg) if t.startswith("Series")]


def test_entity_bar_chart_stands_its_bars_on_one_baseline_without_a_legend() -> None:
    svg = _svg(resolved_revision("bar_chart"))

    bars = _bars(svg)
    assert len(bars) == 3
    assert {bar["fill"] for bar in bars} == {SERIES_COLOURS[0]}
    assert len({round(bar["y"] + bar["height"], 6) for bar in bars}) == 1
    assert len({bar["x"] for bar in bars}) == 3
    assert not list(svg.iter(f"{_SVG}pattern"))
    assert {"Ethiopia", "Ghana", "Kenya"} <= set(_texts(svg))


def test_a_bar_chart_with_a_long_label_lays_its_bars_across() -> None:
    envelope = resolved_revision("bar_chart")
    envelope["component"]["series"][0]["label"] = "Democratic Republic of the Congo"

    svg = _svg(envelope)

    bars = _bars(svg)
    assert len({bar["x"] for bar in bars}) == 1
    assert len({bar["y"] for bar in bars}) == 3
    first, ghana, kenya = (bar["width"] for bar in bars)
    assert first > kenya > ghana
    assert "Democratic Republic of the Congo" in _texts(svg)
    assert "0" not in _texts(svg)


def test_a_label_wider_than_the_gutter_is_cut_with_an_ellipsis() -> None:
    envelope = resolved_revision("bar_chart")
    label = "Recipients in the region not attributed to any single country"
    envelope["component"]["series"][0]["label"] = label

    texts = _texts(_svg(envelope))

    (cut,) = [text for text in texts if text.endswith("…")]
    assert label.startswith(cut.removesuffix("…"))
    assert label not in texts


def test_a_legend_wider_than_the_card_wraps_onto_rows() -> None:
    envelope = resolved_revision("line_chart")
    (one,) = envelope["component"]["series"]
    envelope["component"]["series"] = [
        {**one, "label": f"Recipient country number {i}"} for i in range(4)
    ]

    svg = _svg(envelope)

    rows = {
        text.get("y")
        for text in svg.iter(f"{_SVG}text")
        if (text.text or "").startswith("Recipient")
    }
    assert len(rows) == 2


def test_a_series_with_only_statuses_draws_no_line() -> None:
    envelope = resolved_revision("line_chart")
    (one,) = envelope["component"]["series"]
    silent = {**one, "label": "Withheld", "points": [{"period": "2015", "status": "withheld"}]}
    envelope["component"]["series"].append(silent)

    svg = _svg(envelope)

    assert [line.get("stroke") for line in _lines(svg)] == [SERIES_COLOURS[0]]
    assert "Withheld" in _texts(svg)


def test_a_bar_chart_with_many_bars_lays_them_across() -> None:
    envelope = resolved_revision("bar_chart")
    (one,) = envelope["component"]["series"][:1]
    envelope["component"]["series"] = [{**one, "label": f"C{i}"} for i in range(9)]

    bars = _bars(_svg(envelope))

    assert len({bar["x"] for bar in bars}) == 1
    assert len({bar["y"] for bar in bars}) == 9


def test_a_bar_with_only_a_status_is_not_drawn() -> None:
    envelope = resolved_revision("bar_chart")
    envelope["component"]["series"][1]["points"] = [{"period": "2024", "status": "withheld"}]

    svg = _svg(envelope)

    assert len(_bars(svg)) == 2
    assert "Ghana" in _texts(svg)


def test_period_bar_chart_groups_each_periods_bars_and_writes_compact_ticks() -> None:
    envelope = _period_bar_chart()
    for one in envelope["component"]["series"]:
        one["points"][0]["value"] *= 1000

    svg = _svg(envelope)

    bars = _bars(svg)
    assert [bar["fill"] for bar in bars] == list(SERIES_COLOURS[:3])
    assert len({round(bar["y"] + bar["height"], 6) for bar in bars}) == 1
    assert {"150K", "300K", "450K", "600K"} <= set(_texts(svg))


def test_payload_text_is_drawn_as_written() -> None:
    envelope = resolved_revision("line_chart")
    envelope["component"]["title"]["text"] = "Aid <b>&</b> {{ loans }}\x01"

    texts = _texts(_svg(envelope))

    assert texts[0] == "Aid <b>&</b> {{ loans }}"


def test_a_long_title_wraps_within_the_card() -> None:
    envelope = resolved_revision("line_chart")
    # Under the 120 characters `projection.title_text` keeps, and wider than the card.
    words = ["disbursements"] * 8
    envelope["component"]["title"]["text"] = " ".join(words)

    texts = _texts(_svg(envelope))

    title_lines = texts[: texts.index("Disbursements, USD millions, constant 2023 prices")]
    assert len(title_lines) > 1
    assert " ".join(title_lines).split() == words


@pytest.mark.parametrize(
    ("member", "axis"), [("line_chart", "y_axis"), ("bar_chart", "value_axis")]
)
def test_a_log_axis_has_no_image(member: str, axis: str) -> None:
    envelope = resolved_revision(member)
    envelope["component"][axis]["scale"] = "log"

    assert chart_svg(envelope) is None
    assert chart_png(envelope) is None


def test_the_image_is_drawn_only_where_the_view_s_rule_plots_a_chart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = resolved_revision("line_chart")
    assert chart_svg(envelope) is not None

    monkeypatch.setattr(chart_image, "plots_as_chart", lambda _: False)

    assert chart_svg(envelope) is None


@pytest.mark.parametrize("member", ["table", "stat_card", "note_narration", "source_card"])
def test_a_member_that_is_not_a_chart_has_no_image(member: str) -> None:
    envelope = resolved_revision(member)

    assert chart_svg(envelope) is None
    assert chart_png(envelope) is None


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
def test_the_png_is_the_svg_at_twice_its_size(member: str) -> None:
    envelope = resolved_revision(member)
    svg = _svg(envelope)

    png = chart_png(envelope)

    assert png is not None
    assert _png_size(png) == (WIDTH * ZOOM, int(svg.get("height", "0")) * ZOOM)
    assert png.endswith(b"IEND\xaeB`\x82")


def test_the_png_draws_its_text_with_the_vendored_fonts_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = resvg_py.svg_to_bytes
    calls: list[dict[str, Any]] = []

    def recording(**kwargs: Any) -> bytes:
        calls.append(kwargs)
        return real(**kwargs)

    def glyph(char: str) -> bytes:
        return _rasterise(
            '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" font-family="Outfit">'
            f'<text x="8" y="30" font-size="24">{char}</text></svg>'
        )

    # Outfit has neither character, so each draws as its missing-glyph box.
    # A machine with a CJK system font, as macOS has, would draw two
    # different glyphs.
    monkeypatch.setattr(resvg_py, "svg_to_bytes", recording)
    assert glyph("漢") == glyph("字") != glyph("")

    assert calls
    for call in calls:
        assert call["skip_system_fonts"] is True
        assert call["font_dirs"] == [str(FONTS_DIR)]


def test_an_entity_bar_chart_of_200_countries_has_no_image() -> None:
    envelope = _entity_bar_chart(200)

    assert chart_svg(envelope) is None
    assert chart_png(envelope) is None


@pytest.mark.parametrize(
    ("envelope", "height"),
    [
        (_entity_bar_chart(24), 620),
        (_period_chart("bar_chart", 200, 1), 678),
        (_period_chart("bar_chart", 33, 6), 678),
        (_period_chart("line_chart", 2400, 1), 678),
        (_period_chart("line_chart", 300, 8), 678),
    ],
    ids=["24 entity bars", "200 periods", "33 periods of 6", "2400 points", "300 periods of 8"],
)
def test_a_chart_at_the_limits_draws_at_its_usual_size(
    envelope: dict[str, Any], height: int
) -> None:
    png = chart_png(envelope)

    assert png is not None
    assert _png_size(png) == (WIDTH * ZOOM, height)


@pytest.mark.parametrize(
    "envelope",
    [
        _entity_bar_chart(25),
        _period_chart("bar_chart", 201, 1),
        _period_chart("bar_chart", 34, 6),
        _period_chart("line_chart", 2401, 1),
        _period_chart("line_chart", 301, 8),
        _period_chart("line_chart", 2, 9),
    ],
    ids=[
        "25 entity bars",
        "201 periods",
        "34 periods of 6",
        "2401 points",
        "301 periods of 8",
        "9 series",
    ],
)
def test_a_chart_past_the_limits_has_no_image(envelope: dict[str, Any]) -> None:
    assert chart_svg(envelope) is None
    assert chart_png(envelope) is None


def test_a_top_20_chart_with_40_character_labels_draws() -> None:
    envelope = _entity_bar_chart(20)
    for i, one in enumerate(envelope["component"]["series"]):
        one["label"] = f"{i:02} {'Democratic Republic of the Congo'}"[:40].ljust(40, "x")

    png = chart_png(envelope)

    assert png is not None
    assert _png_size(png) == (WIDTH * ZOOM, 620)


def test_a_chart_taller_than_1200_px_has_no_image() -> None:
    envelope = _period_chart("line_chart", 2, 8)
    for i, one in enumerate(envelope["component"]["series"]):
        one["label"] = f"{i} {'recipient ' * 8}"
    envelope["component"]["y_axis"]["label"] = "official development assistance " * 150

    assert chart_svg(envelope) is None


def test_grouped_bars_that_cannot_keep_their_gaps_touch_inside_the_plot() -> None:
    chart = chart_model(_period_chart("bar_chart", 30, 6))
    assert chart is not None
    parts: list[str] = []

    _vertical_chart(parts, chart, 0.0, WIDTH - 2 * _PAD)

    plot = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>')
    grid = next(plot.iter(f"{_SVG}line"))
    left, right = float(grid.get("x1", "0")), float(grid.get("x2", "0"))
    bars = _bars(plot)
    assert len(bars) == 180
    assert all(left <= bar["x"] and bar["x"] + bar["width"] <= right for bar in bars)
    edges = [(bar["x"], bar["x"] + bar["width"]) for bar in bars]
    gaps = [start - end for (_, end), (start, _) in itertools.pairwise(edges)]
    assert min(gaps) >= -0.02
    within = [gap for i, gap in enumerate(gaps, 1) if i % 6]
    assert within == pytest.approx([0.0] * len(within), abs=0.02)


@pytest.mark.parametrize(
    ("statuses", "dots"),
    [
        ({"2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"}, 1),
        ({"2016", "2018", "2020", "2021", "2022"}, 3),
        ({"2018"}, 0),
    ],
    ids=["one period", "isolated points", "no isolated point"],
)
def test_a_point_with_no_neighbour_draws_as_a_dot(statuses: set[str], dots: int) -> None:
    envelope = resolved_revision("line_chart")
    (one,) = envelope["component"]["series"]
    one["points"] = [
        {"period": point["period"], "status": "missing"} if point["period"] in statuses else point
        for point in one["points"]
    ]
    if len(statuses) == 9:
        one["points"] = one["points"][:1]
        envelope["component"]["x_axis"]["window"]["end"] = "2015"

    svg = _svg(envelope)

    circles = list(svg.iter(f"{_SVG}circle"))
    assert len(circles) == dots
    assert all(circle.get("fill") == SERIES_COLOURS[0] for circle in circles)
    assert all(float(circle.get("r", "0")) > 1 for circle in circles)


def test_the_light_and_regular_weights_draw_different_glyphs() -> None:
    def title(weight: int) -> bytes:
        return _rasterise(
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40" font-family="Outfit">'
            f'<text x="8" y="26" font-size="24" font-weight="{weight}">Malaria</text></svg>'
        )

    assert title(300) != title(400)


def test_every_vendored_font_sits_beside_its_licence() -> None:
    fonts = sorted(FONTS_DIR.glob("*/*.ttf"))

    assert {font.parent.name for font in fonts} == {"outfit"}
    for font in fonts:
        assert "SIL Open Font License" in (font.parent / "OFL.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0, "0"),
        (150, "150"),
        (0.25, "0.25"),
        (0.125, "0.13"),
        (1500, "1.5K"),
        (12_345, "12K"),
        (2_500_000, "2.5M"),
        (3e9, "3B"),
        (1e12, "1T"),
        (-1500, "-1.5K"),
    ],
)
def test_ticks_read_as_intl_compact_writes_them(value: float, text: str) -> None:
    assert _tick_text(value) == text


@pytest.mark.parametrize(
    ("values", "ticks"),
    [
        ([441.9], [0, 150, 300, 450, 600]),
        ([1, 2, 3], [0, 0.75, 1.5, 2.25, 3]),
        ([-50, 100], [-50, 0, 50, 100, 150]),
        ([-441.9], [-600, -450, -300, -150, 0]),
        ([], [0, 0.25, 0.5, 0.75, 1]),
    ],
)
def test_value_ticks_follow_recharts(values: list[float], ticks: list[float]) -> None:
    assert _value_ticks(values) == ticks


_LARGEST = sys.float_info.max


@pytest.mark.parametrize(
    ("values", "ticks"),
    [
        ([1e40], [0, 2.5e39, 5e39, 7.5e39, 1e40]),
        ([-1e40], [-1e40, -7.5e39, -5e39, -2.5e39, 0]),
        ([1e-323], [0, 5e-324, 5e-324, 1e-323, 1e-323]),
        ([1e-300, 1e300], [0, 2.5e299, 5e299, 7.5e299, 1e300]),
        ([_LARGEST], [0, 4.5e307, 9e307, 1.35e308, math.inf]),
        ([-_LARGEST, _LARGEST], [-math.inf, -9e307, 0, 9e307, math.inf]),
        ([7, 7, 7], [0, 2, 4, 6, 8]),
        ([0, 0], [0, 0.25, 0.5, 0.75, 1]),
    ],
)
def test_value_ticks_cover_the_float_range(values: list[float], ticks: list[float]) -> None:
    assert _value_ticks(values) == ticks


_VALUES = [412.3, 455.1, 398.2]


@pytest.mark.parametrize(
    ("values", "lower", "upper", "ticks"),
    [
        (_VALUES, 390, 460, [390, 400, 420, 440, 460]),
        (_VALUES, 387.5, 461.2, [387.5, 400, 420, 440, 461.2]),
        (_VALUES, 399, 460, [399, 420, 440, 460]),
        (_VALUES, 390, None, [390, 400, 420, 440, 460]),
        (_VALUES, None, 460, [0, 150, 300, 460]),
        (_VALUES, None, 600, [0, 150, 300, 450, 600]),
        ([-3, 441.9], None, 460, [-150, 0, 150, 300, 460]),
        ([-50, 100], -50, 100, [-50, 0, 40, 80, 100]),
        ([-50, -20], None, -10, [-50, -40, -30, -20, -10]),
        ([0.2, 0.9], 0, 1, [0, 0.25, 0.5, 0.75, 1]),
        ([], 5, 5, [5, 5]),
        ([], 9, 5, [9, 5]),
    ],
)
def test_value_ticks_end_at_an_authored_bound_with_round_ticks_between(
    values: list[float], lower: float | None, upper: float | None, ticks: list[float]
) -> None:
    assert _value_ticks(values, lower=lower, upper=upper) == ticks


@pytest.mark.parametrize(
    ("values", "lower", "upper", "ticks"),
    [
        ([_LARGEST], -_LARGEST, _LARGEST, [-_LARGEST, -9e307, 0, 9e307, _LARGEST]),
        ([1e308], -1e308, 1e308, [-1e308, -5e307, 0, 5e307, 1e308]),
        ([_LARGEST], 0, None, [0, 4.5e307, 9e307, 1.35e308, math.inf]),
        ([-_LARGEST, _LARGEST], None, _LARGEST, [-math.inf, -9e307, 0, 9e307, _LARGEST]),
        ([1.0], -_LARGEST, None, [-_LARGEST, -1.35e308, -9e307, -4.5e307, 0, 4.5e307]),
        ([1e-300, 1e300], 1e-300, None, [1e-300, 2.5e299, 5e299, 7.5e299, 1e300]),
        ([1e-323], 0, 1e-323, [0, 5e-324, 1e-323]),
        ([], 1e300, None, [1e300] * 5),
        ([1.0], None, math.inf, [0, math.inf]),
    ],
)
def test_bounded_value_ticks_cover_the_float_range(
    values: list[float], lower: float | None, upper: float | None, ticks: list[float]
) -> None:
    assert _value_ticks(values, lower=lower, upper=upper) == ticks


_SWEEP_STARTS = [1, 1.2, 2.5, 3.9, 7, 9.8, 12.345, 70, 99.5, 390, 999]
_SWEEP_WIDTHS = [0.0004, 0.003, 0.01, 0.02, 0.031, 0.1, 0.25, 0.5, 1, 2, 7.3]
_SWEEP_POWERS = range(-6, 13)


def _swept_ranges() -> list[tuple[float, float]]:
    """Value-axis ranges at each magnitude from a millionth to a trillion:
    round and awkward starts, widths from 0.04% of the start to seven times
    it, each above zero, mirrored below it, and across it."""
    ranges = []
    for power, start, width in itertools.product(_SWEEP_POWERS, _SWEEP_STARTS, _SWEEP_WIDTHS):
        low = start * 10.0**power
        high = low + low * width
        ranges += [(low, high), (-high, -low), (-low, high)]
    return ranges


def _on_one_step(ticks: Sequence[float], *, open_end: int) -> bool:
    """Whether every tick but the authored end is a multiple of one step,
    which is the gap at the open end, and none is missing: the gap at the
    authored end is at least half a step and less than one and a half."""
    ats = [Decimal(repr(tick)) for tick in ticks]
    if open_end == 0:
        ats = [-at for at in reversed(ats)]
    *between, last = ats[1:]
    step = last - between[-1]
    return (
        all(at % step == 0 for at in (*between, last))
        and all(b - a == step for a, b in itertools.pairwise(between))
        and step / 2 <= between[0] - ats[0] < step * 3 / 2
    )


def test_no_authored_range_draws_a_tick_label_twice() -> None:
    """Over the sweep, with both bounds authored and with either one alone:
    the ticks ascend, an authored bound is an end exactly, and an open side
    covers the values. With one bound authored the labels all differ and
    the ticks past the bound are multiples of one step. With both authored
    the one axis with a repeated label is two ends that read the same,
    which has no image."""
    ranges = _swept_ranges()
    assert len(ranges) == 3 * len(_SWEEP_POWERS) * len(_SWEEP_STARTS) * len(_SWEEP_WIDTHS)
    alone = stepped = 0
    for low, high in ranges:
        values = [low + (high - low) * share for share in (0.1, 0.5, 0.9)]
        for lower, upper in [(low, high), (low, None), (None, high)]:
            ticks = _value_ticks(values, lower=lower, upper=upper)
            labels = [_tick_text(tick) for tick in ticks]
            case = f"{lower} to {upper}: {ticks} read {labels}"

            assert all(a < b for a, b in itertools.pairwise(ticks)), case
            assert 2 <= len(ticks) <= 7, case
            assert ticks[0] == low if lower is not None else ticks[0] <= min(0, *values), case
            assert ticks[-1] == high if upper is not None else ticks[-1] >= max(0, *values), case
            if lower is None or upper is None:
                assert len(set(labels)) == len(labels), case
                if len(ticks) > 2:
                    assert _on_one_step(ticks, open_end=0 if lower is None else -1), case
                    stepped += 1
            elif len(set(labels)) < len(labels):
                assert len(ticks) == 2, case
                alone += 1
    # The sweep reaches each outcome: most one-sided axes have ticks between
    # their ends, most two-sided axes draw, and some of those have two ends
    # too close to read apart.
    assert stepped > len(ranges)
    assert 0 < alone < len(ranges) / 2


@pytest.mark.parametrize(
    "ticks",
    [[0, 5e-324], [-5e-324, 0, 5e-324], [1.0, 1.0], [0, math.inf]],
    ids=["0 to 5e-324", "-5e-324 to 5e-324", "equal", "infinite"],
)
def test_ticks_whose_halved_span_is_empty_are_not_drawable(ticks: list[float]) -> None:
    assert not _drawable(ticks)


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
@pytest.mark.parametrize(
    "bounds",
    [{"min": 0, "max": 5e-324}, {"min": -5e-324, "max": 5e-324}],
    ids=["0 to 5e-324", "-5e-324 to 5e-324"],
)
def test_a_range_too_narrow_to_scale_has_no_image(member: str, bounds: dict[str, Any]) -> None:
    envelope = _ranged(_with_values(member, [0.0, 0.0]), bounds)

    assert chart_svg(envelope) is None
    assert chart_png(envelope) is None


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (1e40, "10000000000000000000000000000T"),
        (-2.5e39, "-2500000000000000000000000000T"),
        (5e-324, f"0.{'0' * 323}5"),
        (1e15, "1000T"),
    ],
)
def test_tick_text_is_exact_at_any_magnitude(value: float, text: str) -> None:
    assert _tick_text(value) == text


def _with_values(member: str, values: list[float]) -> dict[str, Any]:
    """A one-series `member` chart of `values`, by period for a line chart
    and by country for a bar chart, whose long labels lay it out across."""
    if member == "bar_chart":
        envelope = _entity_bar_chart(len(values))
        for one, value in zip(envelope["component"]["series"], values, strict=True):
            one["label"] = f"Recipient {one['label']}"
            one["points"][0]["value"] = value
        return envelope
    envelope = _period_chart("line_chart", len(values), 1)
    for point, value in zip(envelope["component"]["series"][0]["points"], values, strict=True):
        point["value"] = value
    return envelope


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
@pytest.mark.parametrize(
    ("values", "line_draws"),
    [
        ([1e40, 2e40], True),
        ([-1e40, -2e40], True),
        ([1e-323, 5e-324], False),
        ([_LARGEST, 1.0], False),
        ([-_LARGEST, _LARGEST], False),
        ([-1e308, 1e308], False),
        ([1e-300, 1e300], False),
        ([5e-324, 1e-10, 3.0], True),
        ([4.0, 4.0, 4.0], True),
        ([0.0, 0.0], True),
    ],
    ids=[
        "1e40",
        "-1e40",
        "1e-323",
        "the largest float",
        "both largest floats",
        "a span past the largest float",
        "tiny and huge",
        "tiny and small",
        "all equal",
        "all zero",
    ],
)
def test_a_chart_at_the_float_range_draws_or_has_no_image(
    member: str, values: list[float], line_draws: bool
) -> None:
    """A value axis whose ticks pass the largest float has no image. So does
    a line chart whose tick labels would take more than 40% of the width. A
    horizontal bar chart has no value ticks to write, so it draws inside
    the float range, even where its span is wider than the largest float,
    and every coordinate it draws is finite."""
    envelope = _with_values(member, values)
    png = chart_png(envelope)

    draws = line_draws if member == "line_chart" else not any(abs(v) == _LARGEST for v in values)
    assert (png is not None) is draws
    svg = chart_svg(envelope)
    assert svg is None or re.search(r"\b(nan|inf)\b", svg, re.IGNORECASE) is None


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
@pytest.mark.parametrize(
    ("values", "bounds", "bar_draws"),
    [
        ([_LARGEST, 1.0], {"min": -_LARGEST, "max": _LARGEST}, True),
        ([-1e308, 1e308], {"min": -1e308, "max": 1e308}, True),
        ([_LARGEST, 1.0], {"min": 0}, False),
        ([-_LARGEST, _LARGEST], {"max": _LARGEST}, False),
        ([1e308, 1.5e308], {"min": 1e308}, True),
    ],
    ids=["both largest floats", "a span past the largest float", "min", "max", "above zero"],
)
def test_a_chart_with_bounds_at_the_float_range_draws_or_has_no_image(
    member: str, values: list[float], bounds: dict[str, Any], bar_draws: bool
) -> None:
    """Authored bounds are finite ticks, so a horizontal bar chart draws
    between the two largest floats, with every coordinate finite. An open
    side whose tick passes the largest float has no image, and neither has a
    line chart, whose tick labels at this size are wider than 40% of the
    width."""
    envelope = _ranged(_with_values(member, values), bounds)
    png = chart_png(envelope)

    assert (png is not None) is (bar_draws and member == "bar_chart")
    svg = chart_svg(envelope)
    assert svg is None or re.search(r"\b(nan|inf)\b", svg, re.IGNORECASE) is None
