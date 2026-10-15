"""The MCP App view: a resolved revision as a Prefab shell and the data it binds.

write_component and describe_component return `app_shell` as a result's
`structuredContent` to a client `server._view_support` answers as an MCP Apps
host, which draws it with Prefab's renderer, the tools' `ui://` resource. Some
hosts pass `structuredContent` to the model, so the shell holds the
component's layout and words and no data value. The values the view binds
travel in the same result's `_meta`, which hosts keep from the model, as
`view_state`: `app_data` under `data` and `loaded` true. The renderer page
merges that into the shell's state (`theme.renderer_html`), where the chart,
table, or stat binds the points, rows, or headline value. The shell's own
state holds empty values and `loaded` false, and a conditional branch
(Prefab's `If`) draws `_NO_DATA` only while `loaded` is false, so a host that
drops `_meta` or draws with an older cached page shows that sentence in place
of a blank chart. The sentence is part of the shell's layout, and the state
holds no words: a host that passes `structuredContent` to the model would
otherwise hand it a sentence that reads as a live error even where the chart
drew.
The view makes no further call, so any host and any MCP session can draw it.
A note_narration or source_card holds no data value, so its shell draws
whole.

The card follows a website Studio card. Its face holds one title (the
model-authored title, or the derived description for a member without
one); for a line or bar chart, a caption under it naming the value axis's
metric and the series' shared unit; the body; and a footer row: the source
names, with a count of caveats where the card carries any, on
the left and a "Built with" DFO mark on the right. The source line opens an
accordion below it holding the rest: the description, a table's shared
unit, each source with its licence, retrieval date, and link, the freshness
line, every warning whole, and the citation. Each view forces Prefab's light mode,
since the website card is light only. Every view is a `theme.themed_app`, so
its payload names, by URL, the stylesheet that defines the `dfo-*` classes,
the info icon, and the DFO logo (a payload is model-visible in some hosts, so
the CSS itself stays out of it), and the series colours are hex values in the
chart and legend props.

Each catalogue member maps to one Prefab component: line_chart to
`LineChart` without dots, bar_chart to `BarChart`, table to `DataTable`,
stat_card to a headline amount with its unit in a lighter weight,
note_narration and source_card to text. Chart ticks use Prefab's `compact`
format. A period-axis chart draws its own legend above the plot, a swatch
and label per series, as the website does. An entity-axis bar chart has
one series standing for its categories, so it has none. A table's value
column uses Prefab's `number` format. Prefab has no logarithmic axis, so a
line or bar chart whose value axis is log draws as a `DataTable` of the same
rows, under a line saying so. Prefab draws a stat value as given, so
`app_data` formats it and its change with website's `compact()` rule.

A result with nothing to draw still opens the renderer, so it gets
`notice_view`, a line of text saying why.

A line or bar chart is read once into a `ChartModel` (`chart_model`): the
title, caption, series labels and colours, rows, the horizontal-bar rule, the
log flag, and the value axis's authored bounds. The view draws from it, and so
does `chart_image`, which turns it into a PNG for a host that cannot draw the
view, so the two agree. Prefab's charts take no axis bounds, so the view's
value axis spans zero and the data whatever the bounds, and the image alone
draws them.
`plots_as_chart` says whether the envelope draws as a chart at all, from the
member type and the value axis's scale, without building the rows.

The words come from `projection`'s helpers, so the view names labels,
sources, and warnings in the text projection's terms.

Prefab's renderer evaluates `{{ ... }}` in the strings it draws, so both
`app_shell` and `app_data` read the envelope with each `{{` in its strings
broken by a zero-width space, and payload text always draws as written.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Any, Literal

from prefab_ui.components import (
    H3,
    Accordion,
    AccordionItem,
    Column,
    Component,
    DataTable,
    DataTableColumn,
    Div,
    Link,
    Muted,
    Small,
    Span,
    Text,
)
from prefab_ui.components.charts import BarChart, ChartSeries, LineChart
from prefab_ui.components.control_flow import If

from presentation_mcp.projection import (
    MAX_SOURCES,
    CardFacts,
    Source,
    card_facts,
    change_text,
    citation_text,
    describe,
    freshness_text,
    shared_unit,
    source_url,
    sources_for,
    title_text,
    unit_text,
    warning_texts,
)
from presentation_mcp.theme import SERIES_COLOURS, SERIES_COLOURS_LIGHT, themed_app

# A bar chart with more bars than this, or any category label longer than
# `_LONG_LABEL`, lays out horizontally so its labels stay readable.
_MANY_BARS = 8
_LONG_LABEL = 12
_TABLE_PAGE_SIZE = 20
# website's `compact()` steps.
_COMPACT_STEPS = ((1e9, "B"), (1e6, "M"), (1e3, "K"))
# Wide enough to round any finite float's exact expansion to one decimal.
_EXACT = Context(prec=400)

_LOG_AXIS_NOTE = (
    "The value axis is logarithmic, which this view cannot draw, so the values show as a table."
)
# Drawn while the view's `loaded` flag is false, which stays so when the view
# never receives `view_state`: the host dropped the result's `_meta`, or it
# drew the view with a cached renderer page that lacks the merge script.
# Claude Desktop keeps a page until the app restarts, whatever
# URI the tools name. It sits in the layout and not in the state, because a
# host may hand `structuredContent` to the model, which would read it as a
# live error.
_NO_DATA = "This view could not read its data. Restarting the app loads the current view."

_TEMPLATE_OPEN = re.compile(r"\{(?=\{)")

type _Body = list[Component]
type _Data = dict[str, Any]


@dataclass(frozen=True, slots=True)
class PlottedSeries:
    """One series as a chart draws it: the row key holding its values, its
    label, and its line colour and the lighter half of its legend swatch."""

    key: str
    label: str
    colour: str
    light_colour: str


@dataclass(frozen=True, slots=True)
class ChartModel:
    """A line or bar chart as the view and the chart image draw it.

    Attributes:
        kind: The catalogue member, `line_chart` or `bar_chart`.
        title: The model-authored title, or the derived description.
        caption: The value axis's metric and the series' shared unit, or
            None where the axis names no metric.
        x_key: The row key of the category axis, `period`, or `category`
            for an entity-axis bar chart.
        x_label: The category axis's header where the chart draws as a
            table, "Period" or the entity axis's dimension.
        series: The plotted series in palette order. An entity-axis bar
            chart has one, keyed `value` and labelled with the value axis's
            metric, standing for all its bars.
        legend: Whether the chart draws a legend above the plot. An
            entity-axis bar chart has none.
        horizontal: Whether the bars lay out horizontally.
        log: Whether the value axis is logarithmic. The view then draws a
            table of the rows, and there is no chart image.
        axis_min: The value axis's authored lower bound, or None where the
            component leaves that side to the renderer.
        axis_max: The value axis's authored upper bound, or None likewise.
        rows: One row per category, the `x_key` then each series' value by
            its key. A point with only a status has no key in its row.
        source_line: The footer's source names and caveat count.
    """

    kind: Literal["line_chart", "bar_chart"]
    title: str
    caption: str | None
    x_key: Literal["period", "category"]
    x_label: str
    series: tuple[PlottedSeries, ...]
    legend: bool
    horizontal: bool
    log: bool
    axis_min: float | None
    axis_max: float | None
    rows: tuple[dict[str, Any], ...]
    source_line: str


def chart_model(envelope: Mapping[str, Any]) -> ChartModel | None:
    """The envelope's chart, or None for a member that is not a line or bar
    chart. The text is the envelope's as written, and `app_shell` and `app_data`
    pass an envelope already made inert."""
    component = envelope["component"]
    kind = component["type"]
    axis = _value_axis(component)
    if axis is None:
        return None
    series = component["series"]
    entity = kind == "bar_chart" and component["category_axis"]["mode"] == "entity"
    bounds = axis.get("range") or {}
    if entity:
        x_key, x_label = "category", component["category_axis"]["category_dimension"]
        plotted = (_plotted("value", axis["label"], 0),)
        rows = tuple(
            _datum({"category": one["label"]}, "value", one["points"][0]) for one in series
        )
    else:
        x_key, x_label = "period", "Period"
        plotted = tuple(
            _plotted(f"s{i}", label, i) for i, label in enumerate(_series_labels(series))
        )
        rows = _period_rows(series)
    return ChartModel(
        kind=kind,
        title=title_text(component) or describe(component),
        caption=_caption(component, axis),
        x_key=x_key,
        x_label=x_label,
        series=plotted,
        legend=not entity,
        horizontal=entity
        and (len(series) > _MANY_BARS or any(len(one["label"]) > _LONG_LABEL for one in series)),
        log=axis["scale"] == "log",
        axis_min=bounds.get("min"),
        axis_max=bounds.get("max"),
        rows=rows,
        source_line=_source_label(card_facts(envelope)),
    )


def plots_as_chart(envelope: Mapping[str, Any]) -> bool:
    """Whether the envelope draws as a chart: a line or bar chart whose value
    axis is not logarithmic. It is true exactly where `chart_model` gives a
    model whose `log` is false. It reads two fields, so a caller can ask before
    paying for the model's rows."""
    axis = _value_axis(envelope["component"])
    return axis is not None and axis["scale"] != "log"


def app_shell(envelope: Mapping[str, Any], *, base_url: str) -> dict[str, Any]:
    """The envelope's view without its data, as Prefab's wire format.

    Args:
        envelope: The resolved revision, `{schema_version, catalog_version,
            component_id, revision, provenance, component}`, with a
            `description` on a described card.
        base_url: The public URL the view's theme stylesheet is served under.

    Returns:
        The `PrefabApp` JSON, ready for `structuredContent`.
    """
    envelope = _inert(envelope)
    component = envelope["component"]
    facts = card_facts(envelope)
    title = title_text(component)
    head: list[Component] = [H3(title or describe(component), css_class="dfo-title")]
    chart = chart_model(envelope)
    if chart is not None and chart.caption is not None:
        head.append(Small(chart.caption, css_class="dfo-caption"))
    children: list[Component] = [Div(css_class="dfo-head", children=head)]
    if chart is not None:
        children.extend(_chart_body(chart))
    else:
        body = _BODIES.get(component["type"])
        if body is not None:
            children.extend(body(component, facts))
    data = _data(component, chart)
    if data is not None:
        children.append(If("!loaded", children=[Muted(_NO_DATA)]))
    children.append(_footer(component, facts, described=title is None))
    if data is None:
        card = Column(children=children, css_class="dfo-card")
        return themed_app(card, base_url=base_url).to_json()

    # The renderer draws a binding to a missing key as its literal template,
    # so the state starts with every key the data will fill, each empty.
    empty = {key: [] if isinstance(value, list) else "" for key, value in data.items()}
    return themed_app(
        Column(children=children, css_class="dfo-card"),
        base_url=base_url,
        state={"data": empty, "loaded": False},
    ).to_json()


def view_state(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """The state `app_shell`'s view takes from the result's `_meta`: its
    `app_data` under `data`, and `loaded` true, which hides the view's
    no-data sentence."""
    return {"data": app_data(envelope), "loaded": True}


def app_data(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """The values `app_shell`'s view binds.

    Returns:
        `rows` for a line chart, bar chart, or table; `value` and `unit`,
        and `change` where it has a change, for a stat card; nothing for a
        member without data values.
    """
    envelope = _inert(envelope)
    return _data(envelope["component"], chart_model(envelope)) or {}


def notice_view(text: str, *, base_url: str) -> dict[str, Any]:
    """A view stating `text` alone, for a result with no revision to draw:
    a refusal, or a reply that carries no envelope or could not be drawn."""
    return themed_app(Column(children=[Text(_inert(text))]), base_url=base_url).to_json()


# --- bodies -------------------------------------------------------------------


def _chart_body(chart: ChartModel) -> _Body:
    """A line or bar chart with compact ticks and no legend of Prefab's,
    under the website's legend where it has one. On a log axis it is a table
    of the same rows under a line saying why."""
    if chart.log:
        headers = {chart.x_key: chart.x_label, **{one.key: one.label for one in chart.series}}
        numeric = {one.key for one in chart.series}
        return [Small(_LOG_AXIS_NOTE), _data_table(headers, len(chart.rows), numeric=numeric)]
    series = [
        ChartSeries(dataKey=one.key, label=one.label, color=one.colour) for one in chart.series
    ]
    shared: dict[str, Any] = {
        "data": _bound("rows"),
        "series": series,
        "x_axis": chart.x_key,
        "show_legend": False,
        "value_format": "compact",
    }
    plot: Component
    if chart.kind == "line_chart":
        plot = LineChart(**shared, show_dots=False)
    else:
        plot = BarChart(**shared, horizontal=chart.horizontal)
    return [_legend(chart.series), plot] if chart.legend else [plot]


def _table_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    del facts
    rows = component["rows"]
    columns = component["columns"]
    headers = [column["label"] for column in columns]
    if shared_unit([row["unit"] for row in rows]) is None:
        headers.append("Unit")
    numeric = {f"c{i}" for i, column in enumerate(columns) if column["key"] == "value"}
    table = _data_table(
        {f"c{i}": header for i, header in enumerate(headers)}, len(rows), numeric=numeric
    )
    return [table]


def _stat_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    del facts
    amount = Text(
        Span(_bound("value")),
        Span(_bound("unit"), css_class="dfo-stat-unit"),
        css_class="dfo-stat-value",
    )
    change = [Muted(_bound("change"))] if component.get("change") is not None else []
    return [amount, Muted(component["period"]), *change]


def _note_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    del facts
    return [Text(component["text"])]


def _source_body(component: Mapping[str, Any], facts: CardFacts) -> _Body:
    return _source_lines(sources_for(facts, component["source_refs"]))


_BODIES: dict[str, Callable[[Mapping[str, Any], CardFacts], _Body]] = {
    "table": _table_body,
    "stat_card": _stat_body,
    "note_narration": _note_body,
    "source_card": _source_body,
}


# --- data ---------------------------------------------------------------------


def _data(component: Mapping[str, Any], chart: ChartModel | None) -> _Data | None:
    """The values the view binds, or None for a member without data values."""
    if chart is not None:
        return {"rows": list(chart.rows)}
    data = _DATA.get(component["type"])
    return None if data is None else data(component)


def _table_data(component: Mapping[str, Any]) -> _Data:
    columns = component["columns"]
    rows = component["rows"]
    unit = shared_unit([row["unit"] for row in rows])
    table_rows = []
    for row in rows:
        cells = [_table_cell(row, column["key"]) for column in columns]
        if unit is None:
            cells.append(unit_text(row["unit"]))
        table_rows.append({f"c{i}": cell for i, cell in enumerate(cells)})
    return {"rows": table_rows}


def _stat_data(component: Mapping[str, Any]) -> _Data:
    """The compact amount, and its magnitude and unit, split as the website's
    stat card splits them; and the change where the card has one."""
    amount, _, magnitude = _compact(component["value"]).partition(" ")
    unit = " ".join(part for part in (magnitude, unit_text(component["unit"])) if part)
    data: _Data = {"value": amount, "unit": unit}
    change = component.get("change")
    if change is not None:
        data["change"] = f"Change: {change_text(change, number=_compact)}."
    return data


_DATA: dict[str, Callable[[Mapping[str, Any]], _Data]] = {
    "table": _table_data,
    "stat_card": _stat_data,
}


# --- helpers ------------------------------------------------------------------


def _bound(key: str) -> str:
    """A binding to `key` of the view's `data` state."""
    return f"{{{{ data.{key} }}}}"


def _value_axis(component: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The value axis of a line or bar chart, or None for any other member."""
    kind = component["type"]
    if kind == "line_chart":
        return component["y_axis"]
    if kind == "bar_chart":
        return component["value_axis"]
    return None


def _plotted(key: str, label: str, index: int) -> PlottedSeries:
    """The series at `index` in the palette's colours. Series take the
    palette in order and wrap."""
    return PlottedSeries(
        key=key,
        label=label,
        colour=SERIES_COLOURS[index % len(SERIES_COLOURS)],
        light_colour=SERIES_COLOURS_LIGHT[index % len(SERIES_COLOURS_LIGHT)],
    )


def _caption(component: Mapping[str, Any], axis: Mapping[str, Any]) -> str | None:
    """A chart's metric and the series' shared unit, the line under its
    title, or None where the value axis names no metric."""
    if not axis["label"]:
        return None
    unit = shared_unit([one["unit"] for one in component["series"]])
    return ", ".join(part for part in (axis["label"], unit) if part)


def _period_rows(series: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """One row per period, one key per series, a missing or withheld point
    left out."""
    by_period: dict[str, dict[str, Any]] = {}
    for i, one in enumerate(series):
        for point in one["points"]:
            row = by_period.setdefault(point["period"], {"period": point["period"]})
            _datum(row, f"s{i}", point)
    return tuple(by_period[period] for period in sorted(by_period))


def _legend(series: Sequence[PlottedSeries]) -> Div:
    """The website's legend: a row of checkered swatches in each series'
    colour pair, each beside its label."""
    return Div(
        css_class="dfo-legend",
        children=[
            Div(
                css_class="dfo-legend-item",
                children=[
                    Span(
                        "",
                        css_class="dfo-legend-swatch",
                        style={
                            "--series-dark": one.colour,
                            "--series-light": one.light_colour,
                        },
                    ),
                    Span(one.label),
                ],
            )
            for one in series
        ],
    )


def _series_labels(series: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each series' label, with its unit where the series' units differ."""
    unit = shared_unit([one["unit"] for one in series])
    return [
        one["label"]
        if unit is not None
        else f"{one['label']} ({unit_text(one['unit']) or 'no unit'})"
        for one in series
    ]


def _data_table(
    headers: Mapping[str, str], row_count: int, *, numeric: Collection[str]
) -> DataTable:
    """A sortable table of the bound rows, one column per `headers` key,
    the `numeric` keys' values grouped as en-US numbers."""
    return DataTable(
        columns=[
            DataTableColumn(
                key=key, header=header, sortable=True, format="number" if key in numeric else None
            )
            for key, header in headers.items()
        ],
        rows=_bound("rows"),
        paginated=row_count > _TABLE_PAGE_SIZE,
        page_size=_TABLE_PAGE_SIZE,
    )


def _datum(row: dict[str, Any], key: str, point: Mapping[str, Any]) -> dict[str, Any]:
    """`row` with `point`'s value under `key`, where the point has one."""
    if "value" in point:
        row[key] = point["value"]
    return row


def _table_cell(row: Mapping[str, Any], key: str) -> Any:
    if key == "period":
        return row["period"]
    if key == "value":
        return row.get("value", row.get("status"))
    return row["dimensions"].get(key, "")


def _compact(value: float) -> str:
    """`value` as website's `compact()` writes it: billions, millions, or
    thousands to one decimal with a spaced letter ("1.2 B"), anything smaller
    to at most two decimals with en-US grouping.

    The website rounds the first with `toFixed`, which rounds the float's
    exact binary value, and the second with `Intl.NumberFormat`, which
    rounds its shortest decimal form, so 1.005 gives "1.01". Both round
    halves away from zero.
    """
    for step, letter in _COMPACT_STEPS:
        if abs(value) >= step:
            return f"{_half_up(Decimal(value / step), 1)} {letter}"
    return f"{_half_up(Decimal(repr(value)), 2):,f}".rstrip("0").rstrip(".")


def _half_up(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP, context=_EXACT)


def _source_label(facts: CardFacts) -> str:
    """The footer's source names, or "More info" for a card without sources,
    then the count of caveats where the card carries any.

    The website adds the count of indicators behind the card. The envelope
    names none, so the label counts caveats in its place.
    """
    names = [source.name for source in sources_for(facts)]
    parts = names[:MAX_SOURCES] or ["More info"]
    if len(names) > MAX_SOURCES:
        parts.append(f"{len(names) - MAX_SOURCES} more sources")
    caveats = len(facts.caveats)
    if caveats:
        parts.append(f"{caveats} caveat{'' if caveats == 1 else 's'}")
    return " · ".join(parts)


def _source_lines(sources: Sequence[Source]) -> list[Component]:
    """Each source's name, licence, and retrieval date, each but the name
    where the card states it, linked where https."""
    lines: list[Component] = []
    for source in sources[:MAX_SOURCES]:
        parts = [source.name]
        if source.licence:
            parts.append(f"licence {source.licence}")
        if source.retrieved:
            parts.append(f"retrieved {source.retrieved}")
        lines.append(Small(", ".join(parts)))
        url = source_url(source)
        if url is not None:
            lines.append(Link(url, href=url))
    if len(sources) > MAX_SOURCES:
        lines.append(Muted(f"And {len(sources) - MAX_SOURCES} more sources."))
    return lines


def _footer(component: Mapping[str, Any], facts: CardFacts, *, described: bool) -> Div:
    """The footer row: the source line, which opens the card's details, and
    the "Built with" mark. The details are the description (unless the title
    already is it), a table's shared unit (a chart's is in its caption),
    sources (a source card lists its own on the face), freshness, every
    warning, and the citation. A card with no details, a narration note that
    carries lineage alone, gets the mark without the source line."""
    details: list[Component] = [] if described else [Muted(f"{describe(component)}.")]
    rows = component.get("rows") if component["type"] == "table" else None
    unit = shared_unit([row["unit"] for row in rows]) if rows else None
    if unit:
        details.append(Muted(f"Unit: {unit}."))
    if component["type"] != "source_card":
        details.extend(_source_lines(sources_for(facts)))
    freshness = freshness_text(facts)
    if freshness is not None:
        details.append(Muted(freshness))
    details.extend(Muted(text) for text in warning_texts(facts, limit=None, count=None))
    citation = citation_text(facts)
    if citation is not None:
        details.append(Muted(citation))
    mark = Span("Built with", css_class="dfo-built-with")
    if not details:
        return Div(css_class="dfo-chrome", children=[mark])
    item = AccordionItem(
        _source_label(facts), children=[Column(children=details, css_class="dfo-details")]
    )
    return Div(
        css_class="dfo-chrome",
        children=[Accordion(children=[item], css_class="dfo-more"), mark],
    )


def _inert(tree: Any) -> Any:
    """`tree` with every `{{` in a string broken by a zero-width space."""
    if isinstance(tree, str):
        return _TEMPLATE_OPEN.sub("{\u200b", tree)
    if isinstance(tree, list):
        return [_inert(item) for item in tree]
    if isinstance(tree, dict):
        return {key: _inert(value) for key, value in tree.items()}
    return tree
