"""Prefab view tests: the shell per catalogue member, the data it binds, and inert text.

Inputs are dfo-spec-kit's catalogue fixtures (`stub_assembler.resolved_revision`).
Each test reads the Prefab wire JSON `app_shell` returns, the tree a host's
renderer receives as `structuredContent`, or the payload `app_data` returns,
the values the result's `_meta` carries to the view.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from presentation_mcp.app_view import (
    _NO_DATA,
    _compact,
    app_data,
    app_shell,
    chart_model,
    lone_points,
    notice_view,
    plots_as_chart,
    view_state,
)
from presentation_mcp.projection import describe, number_text, signed_text
from presentation_mcp.theme import FONTS_URL, theme_url

from .stub_assembler import resolved_revision

MEMBERS = ("line_chart", "bar_chart", "table", "stat_card", "note_narration", "source_card")
BASE_URL = "https://example.test/mcp/presentation"
SERIES_PALETTE = ["#6467de", "#f6935a", "#f46f58", "#bb6fdc", "#662d6e", "#463138"]
SERIES_PALETTE_LIGHT = ["#8285e7", "#f9b170", "#f78f76", "#cf94e8", "#7f4487", "#5e474f"]


def _shell(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return app_shell(envelope, base_url=BASE_URL)


def _nodes(tree: Any) -> Iterator[dict[str, Any]]:
    """Every component node in `tree`, depth first."""
    if isinstance(tree, dict):
        if "type" in tree:
            yield tree
        for value in tree.values():
            yield from _nodes(value)
    elif isinstance(tree, list):
        for item in tree:
            yield from _nodes(item)


def _only(view: dict[str, Any], kind: str) -> dict[str, Any]:
    (node,) = (node for node in _nodes(view["view"]) if node["type"] == kind)
    return node


def _texts(view: dict[str, Any]) -> list[str]:
    return [node["content"] for node in _nodes(view["view"]) if "content" in node]


def _more_info(view: dict[str, Any]) -> dict[str, Any]:
    return _only(view, "Accordion")


def _source_line(view: dict[str, Any]) -> str:
    """The footer's source line, the More info accordion's trigger."""
    (item,) = _more_info(view)["children"]
    return item["title"]


def _legend(view: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    """Each legend entry's label and swatch style, or nothing without a legend."""
    legends = [node for node in _nodes(view["view"]) if node.get("cssClass") == "dfo-legend"]
    if not legends:
        return []
    (legend,) = legends
    entries = []
    for item in legend["children"]:
        swatch, label = item["children"]
        assert (item["cssClass"], swatch["cssClass"]) == ("dfo-legend-item", "dfo-legend-swatch")
        entries.append((label["content"], swatch["style"]))
    return entries


def _details(view: dict[str, Any]) -> list[str]:
    """The texts inside the More info accordion."""
    return [node["content"] for node in _nodes(_more_info(view)) if "content" in node]


def _face_texts(view: dict[str, Any]) -> list[str]:
    """The texts drawn outside the More info accordion."""
    hidden = {id(node) for node in _nodes(_more_info(view))}
    return [
        node["content"]
        for node in _nodes(view["view"])
        if "content" in node and id(node) not in hidden
    ]


def _leaves(tree: Any) -> Iterator[Any]:
    """Every scalar in `tree`."""
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _leaves(value)
    elif isinstance(tree, list):
        for item in tree:
            yield from _leaves(item)
    else:
        yield tree


def _without_conditions(tree: Any) -> Any:
    """`tree` with every `Condition` node (Prefab's `If` branches) removed."""
    if isinstance(tree, dict):
        return {
            key: _without_conditions(value)
            for key, value in tree.items()
            if not (isinstance(value, dict) and value.get("type") == "Condition")
        }
    if isinstance(tree, list):
        return [
            _without_conditions(item)
            for item in tree
            if not (isinstance(item, dict) and item.get("type") == "Condition")
        ]
    return tree


def _data_values(component: Mapping[str, Any]) -> list[float]:
    """Every numeric value a component's points, rows, and headline carry."""
    values = [
        point["value"]
        for one in component.get("series", ())
        for point in one["points"]
        if "value" in point
    ]
    values.extend(row["value"] for row in component.get("rows", ()) if "value" in row)
    if "value" in component:
        values.append(component["value"])
    if component.get("change") is not None:
        values.append(component["change"]["value"])
    return values


@pytest.mark.parametrize("member", MEMBERS)
def test_each_member_is_a_light_card_with_its_citation_in_more_info(member: str) -> None:
    envelope = resolved_revision(member)

    view = _shell(envelope)

    assert {"$prefab", "view"} <= set(view)
    assert view["mode"] == "light"
    assert view["view"]["children"][0]["cssClass"] == "dfo-card"
    assert _more_info(view)["cssClass"] == "dfo-more"
    assert envelope["provenance"]["citation"] in _details(view)


@pytest.mark.parametrize("member", MEMBERS)
def test_the_footer_is_the_source_line_beside_the_built_with_mark(member: str) -> None:
    envelope = resolved_revision(member)

    view = _shell(envelope)

    card = view["view"]["children"][0]
    footer = card["children"][-1]
    assert footer["cssClass"] == "dfo-chrome"
    accordion, mark = footer["children"]
    assert accordion["type"] == "Accordion"
    assert (mark["type"], mark["cssClass"], mark["content"]) == (
        "Span",
        "dfo-built-with",
        "Built with",
    )
    names = [source["name"] for source in envelope["provenance"]["sources"]]
    assert _source_line(view) == (" · ".join(names) or "More info")


@pytest.mark.parametrize("member", MEMBERS)
def test_the_face_carries_one_title_and_the_description_goes_to_more_info(member: str) -> None:
    envelope = resolved_revision(member)
    description = describe(envelope["component"])

    view = _shell(envelope)

    (title,) = (node for node in _nodes(view["view"]) if node["type"] == "H3")
    assert title["cssClass"] == "dfo-title"
    expected = envelope["component"].get("title")
    if expected is None:
        assert title["content"] == description
        assert f"{description}." not in _details(view)
    else:
        assert title["content"] == expected["text"]
        assert f"{description}." in _details(view)
        assert f"{description}." not in _face_texts(view)


@pytest.mark.parametrize("member", MEMBERS)
def test_no_shell_carries_a_data_value(member: str) -> None:
    envelope = resolved_revision(member)
    values = _data_values(envelope["component"])

    view = _shell(envelope)

    leaves = list(_leaves(view))
    strings = [leaf for leaf in leaves if isinstance(leaf, str)]
    for value in values:
        assert value not in leaves
        for text in (number_text(value), signed_text(value), _compact(value)):
            assert not any(text in string for string in strings), (value, text)


def test_the_data_value_check_finds_a_value_in_a_view() -> None:
    envelope = resolved_revision("stat_card")
    envelope["component"]["label"] = "Disbursements of 441.9"

    strings = [leaf for leaf in _leaves(_shell(envelope)) if isinstance(leaf, str)]

    assert any(number_text(envelope["component"]["value"]) in string for string in strings)


@pytest.mark.parametrize("member", ["line_chart", "bar_chart", "table", "stat_card"])
def test_a_shell_with_data_starts_empty_and_not_loaded(member: str) -> None:
    envelope = resolved_revision(member)

    view = _shell(envelope)

    assert "onMount" not in view["view"]["children"][0]
    assert set(view["state"]) == {"data", "loaded"}
    assert view["state"]["loaded"] is False
    assert set(view["state"]["data"]) == set(app_data(envelope))
    assert all(value in ([], "") for value in view["state"]["data"].values())


@pytest.mark.parametrize("member", ["line_chart", "bar_chart", "table", "stat_card"])
def test_the_no_data_sentence_draws_only_while_the_view_is_not_loaded(member: str) -> None:
    view = _shell(resolved_revision(member))

    (condition,) = (node for node in _nodes(view["view"]) if node["type"] == "Condition")

    (case,) = condition["cases"]
    assert case["when"] == "{{ !loaded }}"
    assert [node["content"] for node in _nodes(case["children"])] == [_NO_DATA]


@pytest.mark.parametrize("member", ["line_chart", "bar_chart", "table", "stat_card"])
def test_the_shell_names_a_failure_to_read_data_only_inside_the_not_loaded_branch(
    member: str,
) -> None:
    view = _shell(resolved_revision(member))

    outside = json.dumps({"view": _without_conditions(view["view"]), "state": view["state"]})

    assert "could not read" not in outside
    assert not any(isinstance(leaf, str) and "read" in leaf for leaf in _leaves(view["state"]))
    assert _NO_DATA in json.dumps(view["view"])


@pytest.mark.parametrize("member", ["line_chart", "bar_chart", "table", "stat_card"])
def test_the_view_state_replaces_the_shells_state_key_for_key(member: str) -> None:
    envelope = resolved_revision(member)

    view = _shell(envelope)
    state = view_state(envelope)

    assert state == {"data": app_data(envelope), "loaded": True}
    assert set(state) == set(view["state"])
    assert set(state["data"]) == set(view["state"]["data"])


@pytest.mark.parametrize("member", ["line_chart", "bar_chart", "table", "stat_card"])
def test_no_value_of_app_data_is_in_the_shell(member: str) -> None:
    envelope = resolved_revision(member)
    shell = json.dumps(_shell(envelope))
    values = list(_leaves(app_data(envelope)))

    assert values
    for value in values:
        assert json.dumps(value) not in shell, value


def test_the_app_data_check_finds_a_value_in_a_shell() -> None:
    envelope = resolved_revision("table")
    row = app_data(envelope)["rows"][0]
    envelope["component"]["title"] = {"text": next(iter(row.values()))}

    assert any(json.dumps(value) in json.dumps(_shell(envelope)) for value in row.values())


@pytest.mark.parametrize("member", ["note_narration", "source_card"])
def test_a_shell_without_data_draws_whole(member: str) -> None:
    envelope = resolved_revision(member)

    view = _shell(envelope)

    assert "state" not in view
    assert app_data(envelope) == {}
    assert not any("{{" in text for text in _texts(view))


def test_line_chart_binds_one_row_per_period_with_a_gap_for_missing() -> None:
    envelope = resolved_revision("line_chart")
    view = _shell(envelope)

    chart = _only(view, "LineChart")
    assert chart["data"] == "{{ data.rows }}"
    assert chart["xAxis"] == "period"
    assert chart["series"] == [{"dataKey": "s0", "label": "Ethiopia", "color": "#6467de"}]
    rows = app_data(envelope)["rows"]
    assert [row["period"] for row in rows] == [str(year) for year in range(2015, 2025)]
    assert rows[3] == {"period": "2018"}
    assert rows[0] == {"period": "2015", "s0": 412.3}
    assert "Malaria grant disbursements to Ethiopia" in _face_texts(view)


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
def test_a_period_chart_draws_the_websites_legend_above_the_plot_even_for_one_series(
    member: str,
) -> None:
    envelope = resolved_revision(member)
    if member == "bar_chart":
        envelope["component"]["category_axis"] = {
            "mode": "period",
            "window": {"start": "2024", "end": "2024"},
        }
    kind = "LineChart" if member == "line_chart" else "BarChart"
    labels = [one["label"] for one in envelope["component"]["series"]]

    view = _shell(envelope)

    card = view["view"]["children"][0]["children"]
    kinds = [node["type"] if node.get("cssClass") != "dfo-legend" else "legend" for node in card]
    assert kinds.index("legend") + 1 == kinds.index(kind)
    assert [label for label, _ in _legend(view)] == labels
    assert _only(view, kind)["showLegend"] is False


def test_line_chart_draws_website_lines_with_compact_ticks() -> None:
    chart = _only(_shell(resolved_revision("line_chart")), "LineChart")

    assert chart["showDots"] is False
    assert chart["valueFormat"] == "compact"
    assert chart["showLegend"] is False


@pytest.mark.parametrize(
    ("missing", "dots"),
    [
        ({"2018"}, False),
        ({"2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"}, True),
        ({"2016", "2018", "2020", "2021", "2022"}, True),
        ({"2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"}, True),
    ],
    ids=["a gap between runs", "first point alone", "isolated points", "last point alone"],
)
def test_line_chart_draws_dots_when_a_point_has_no_neighbour(missing: set[str], dots: bool) -> None:
    envelope = resolved_revision("line_chart")
    (one,) = envelope["component"]["series"]
    one["points"] = [
        {"period": point["period"], "status": "missing"} if point["period"] in missing else point
        for point in one["points"]
    ]

    assert _only(_shell(envelope), "LineChart")["showDots"] is dots


@pytest.mark.parametrize(
    ("values", "lone"),
    [
        ([], []),
        ([None], [False]),
        ([1.0], [True]),
        ([1.0, 2.0, 3.0], [False, False, False]),
        ([1.0, None, 3.0], [True, False, True]),
        ([1.0, 2.0, None, 4.0, None], [False, False, False, True, False]),
        ([0.0, None, None, 0.0], [True, False, False, True]),
    ],
    ids=["no rows", "no value", "single", "all joined", "all lone", "mixed", "zeros"],
)
def test_lone_points_marks_a_value_with_no_value_on_either_side(
    values: list[float | None], lone: list[bool]
) -> None:
    rows = [{"period": str(i)} | ({} if v is None else {"s0": v}) for i, v in enumerate(values)]

    assert lone_points(rows, "s0") == lone


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
def test_several_series_take_the_palette_in_order_under_a_legend(member: str) -> None:
    envelope = resolved_revision(member)
    component = envelope["component"]
    if member == "bar_chart":
        component["category_axis"] = {"mode": "period", "window": {"start": "2024", "end": "2024"}}
    (one,) = component["series"][:1]
    component["series"] = [{**one, "label": f"Series {i}"} for i in range(7)]

    view = _shell(envelope)

    chart = _only(view, "LineChart" if member == "line_chart" else "BarChart")
    assert chart["valueFormat"] == "compact"
    order = [*range(6), 0]
    assert [series["color"] for series in chart["series"]] == [SERIES_PALETTE[i] for i in order]
    assert [style for _, style in _legend(view)] == [
        {"--series-dark": SERIES_PALETTE[i], "--series-light": SERIES_PALETTE_LIGHT[i]}
        for i in order
    ]


def test_entity_bar_chart_binds_one_bar_per_entity() -> None:
    envelope = resolved_revision("bar_chart")

    chart = _only(_shell(envelope), "BarChart")

    assert chart["data"] == "{{ data.rows }}"
    assert app_data(envelope)["rows"] == [
        {"category": "Ethiopia", "value": 441.9},
        {"category": "Ghana", "value": 225.3},
        {"category": "Kenya", "value": 305.6},
    ]
    assert chart["series"] == [{"dataKey": "value", "label": "Disbursements", "color": "#6467de"}]
    assert chart["horizontal"] is False
    assert chart["showLegend"] is False
    assert _legend(_shell(envelope)) == []
    assert chart["valueFormat"] == "compact"


def test_bar_chart_with_long_labels_lays_out_horizontally() -> None:
    envelope = resolved_revision("bar_chart")
    envelope["component"]["series"][0]["label"] = "Democratic Republic of the Congo"

    chart = _only(_shell(envelope), "BarChart")

    assert chart["horizontal"] is True


def test_period_bar_chart_bars_each_period() -> None:
    envelope = resolved_revision("bar_chart")
    envelope["component"]["category_axis"] = {
        "mode": "period",
        "window": {"start": "2024", "end": "2024"},
    }

    chart = _only(_shell(envelope), "BarChart")

    assert chart["xAxis"] == "period"
    assert app_data(envelope)["rows"] == [{"period": "2024", "s0": 441.9, "s1": 225.3, "s2": 305.6}]


@pytest.mark.parametrize(
    ("member", "columns"),
    [
        ("line_chart", ["period", "s0"]),
        ("bar_chart", ["category", "value"]),
    ],
)
def test_a_log_scale_chart_draws_as_a_table_of_its_rows_and_says_why(
    member: str, columns: list[str]
) -> None:
    envelope = resolved_revision(member)
    axis = "y_axis" if member == "line_chart" else "value_axis"
    envelope["component"][axis]["scale"] = "log"

    view = _shell(envelope)

    kinds = {node["type"] for node in _nodes(view["view"])}
    assert not kinds & {"LineChart", "BarChart"}
    table = _only(view, "DataTable")
    assert [column["key"] for column in table["columns"]] == columns
    assert [column.get("format") for column in table["columns"]] == [None, "number"]
    assert table["rows"] == "{{ data.rows }}"
    assert any("logarithmic" in text for text in _texts(view))
    assert _legend(view) == []
    assert app_data(envelope) == app_data(resolved_revision(member))


@pytest.mark.parametrize("member", MEMBERS)
@pytest.mark.parametrize("scale", ["linear", "log"])
def test_an_envelope_plots_as_a_chart_where_its_model_is_not_on_a_log_axis(
    member: str, scale: str
) -> None:
    envelope = resolved_revision(member)
    axis = {"line_chart": "y_axis", "bar_chart": "value_axis"}.get(member)
    if axis is not None:
        envelope["component"][axis]["scale"] = scale

    chart = chart_model(envelope)

    assert plots_as_chart(envelope) is (chart is not None and not chart.log)
    assert plots_as_chart(envelope) is (axis is not None and scale == "linear")


@pytest.mark.parametrize(
    ("bounds", "authored"),
    [
        ({"min": 390, "max": 460}, (390, 460)),
        ({"min": 390}, (390, None)),
        ({"min": None, "max": 460}, (None, 460)),
        ({"min": None, "max": None}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
    ],
    ids=["both", "min", "max", "both null", "empty", "null"],
)
@pytest.mark.parametrize(
    ("member", "axis"), [("line_chart", "y_axis"), ("bar_chart", "value_axis")]
)
def test_a_chart_model_carries_the_value_axis_s_authored_bounds(
    member: str, axis: str, bounds: dict[str, Any] | None, authored: tuple[Any, Any]
) -> None:
    envelope = resolved_revision(member)
    envelope["component"][axis]["range"] = bounds

    chart = chart_model(envelope)

    assert chart is not None
    assert (chart.axis_min, chart.axis_max) == authored


def test_a_chart_model_reads_the_shipped_range_and_none_where_the_axis_has_no_range() -> None:
    envelope = resolved_revision("line_chart")
    shipped = chart_model(envelope)
    del envelope["component"]["y_axis"]["range"]
    without = chart_model(envelope)

    assert shipped is not None
    assert without is not None
    assert (shipped.axis_min, shipped.axis_max) == (390, 460)
    assert (without.axis_min, without.axis_max) == (None, None)


@pytest.mark.parametrize(
    ("member", "axis"), [("line_chart", "y_axis"), ("bar_chart", "value_axis")]
)
def test_the_view_is_the_same_whatever_the_authored_range(member: str, axis: str) -> None:
    envelope = resolved_revision(member)
    del envelope["component"][axis]["range"]
    shell, data = _shell(envelope), app_data(envelope)

    for bounds in ({"min": -1000, "max": 1000}, {"min": 0}, {"max": 5000}, {"min": None}):
        envelope["component"][axis]["range"] = bounds
        assert _shell(envelope) == shell
        assert app_data(envelope) == data


def test_whether_an_envelope_plots_as_a_chart_is_read_without_its_rows() -> None:
    envelope = resolved_revision("line_chart")
    del envelope["component"]["series"]
    del envelope["provenance"]

    assert plots_as_chart(envelope)


def test_table_binds_its_own_columns_and_rows() -> None:
    envelope = resolved_revision("table")
    view = _shell(envelope)

    table = _only(view, "DataTable")
    assert [column["header"] for column in table["columns"]] == ["recipient", "Period", "Value"]
    assert [column.get("format") for column in table["columns"]] == [None, None, "number"]
    assert table["rows"] == "{{ data.rows }}"
    rows = app_data(envelope)["rows"]
    assert rows[0] == {"c0": "Ethiopia", "c1": "2023", "c2": 433.7}
    assert len(rows) == 4
    assert "Unit: USD millions, constant 2023 prices." in _details(view)


def test_table_rows_with_differing_units_carry_a_unit_column() -> None:
    envelope = resolved_revision("table")
    envelope["component"]["rows"][0]["unit"] = "percent"

    table = _only(_shell(envelope), "DataTable")

    assert table["columns"][-1]["header"] == "Unit"
    assert app_data(envelope)["rows"][0]["c3"] == "percent"


def test_stat_card_binds_its_amount_unit_and_change() -> None:
    envelope = resolved_revision("stat_card")
    view = _shell(envelope)

    stat = next(node for node in _nodes(view["view"]) if node.get("cssClass") == "dfo-stat-value")
    amount, unit = stat["children"]
    assert amount["content"] == "{{ data.value }}"
    assert unit["content"] == "{{ data.unit }}"
    assert unit["cssClass"] == "dfo-stat-unit"
    face = _face_texts(view)
    assert face.index("2024") < face.index("{{ data.change }}")
    assert app_data(envelope) == {
        "value": "441.9",
        "unit": "USD millions, constant 2023 prices",
        "change": "Change: +1.89% (percent change, 2023 to 2024).",
    }


def test_a_large_stat_carries_its_magnitude_with_its_unit() -> None:
    envelope = resolved_revision("stat_card")
    component = envelope["component"]
    component["value"] = 1_234_567_890
    component["change"] = {**component["change"], "derivation": "difference", "value": -45_678}

    data = app_data(envelope)

    assert data["value"] == "1.2"
    assert data["unit"] == "B USD millions, constant 2023 prices"
    assert data["change"] == "Change: -45.7 K (difference, 2023 to 2024)."


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0, "0"),
        (441.9, "441.9"),
        (2.0, "2"),
        (0.125, "0.13"),
        (999.999, "1,000"),
        (1250, "1.3 K"),
        (-1250, "-1.3 K"),
        (1.005, "1.01"),
        (2.675, "2.68"),
        (1_000_000, "1.0 M"),
        (1_450_000, "1.4 M"),
        (1_234_567_890, "1.2 B"),
    ],
)
def test_compact_follows_the_website_rule(value: float, text: str) -> None:
    assert _compact(value) == text


def test_compact_writes_the_largest_float_without_a_decimal_error() -> None:
    assert _compact(1.7e308).endswith(".0 B")


def test_narration_is_plain_text_labelled_as_the_models_voice() -> None:
    envelope = resolved_revision("note_narration")

    view = _shell(envelope)

    assert _only(view, "Text")["content"] == envelope["component"]["text"]
    assert _only(view, "H3")["content"] == "Narration in the model's voice"
    assert not any(node["type"] == "Markdown" for node in _nodes(view["view"]))


def test_source_card_lists_and_links_its_sources_once() -> None:
    view = _shell(resolved_revision("source_card"))

    link = _only(view, "Link")
    assert link["href"] == "https://stats.oecd.org/"
    assert _texts(view).count("OECD DAC, licence CC BY 4.0, retrieved 2026-06-12") == 1


def test_a_chart_names_its_sources_on_the_face_and_details_them_in_more_info() -> None:
    view = _shell(resolved_revision("line_chart"))

    assert _source_line(view) == "OECD DAC"
    assert not any("retrieved" in text or "licence" in text for text in _face_texts(view))
    assert "OECD DAC, licence CC BY 4.0, retrieved 2026-06-12" in _details(view)
    (link,) = (node for node in _nodes(_more_info(view)) if node["type"] == "Link")
    assert link["href"] == "https://stats.oecd.org/"
    assert (
        "Data through 2024; last refreshed 2026-06-12; next update expected 2026-10."
        in _details(view)
    )
    assert not any(text.startswith("Unit:") for text in _details(view))


@pytest.mark.parametrize(
    ("member", "caption"),
    [
        ("line_chart", "Disbursements, USD millions, constant 2023 prices"),
        ("bar_chart", "Disbursements, USD millions, constant 2023 prices"),
    ],
)
@pytest.mark.parametrize("scale", ["linear", "log"])
def test_a_chart_names_its_metric_and_unit_under_its_title(
    member: str, caption: str, scale: str
) -> None:
    envelope = resolved_revision(member)
    envelope["component"]["y_axis" if member == "line_chart" else "value_axis"]["scale"] = scale

    view = _shell(envelope)

    (head,) = (node for node in _nodes(view["view"]) if node.get("cssClass") == "dfo-head")
    title, line = head["children"]
    assert title["type"] == "H3"
    assert (line["cssClass"], line["content"]) == ("dfo-caption", caption)


def test_a_chart_whose_series_units_differ_names_its_metric_alone() -> None:
    envelope = resolved_revision("line_chart")
    (one,) = envelope["component"]["series"]
    envelope["component"]["series"] = [
        one,
        {**one, "id": "pct", "label": "Kenya", "unit": "percent"},
    ]

    view = _shell(envelope)

    (line,) = (node for node in _nodes(view["view"]) if node.get("cssClass") == "dfo-caption")
    assert line["content"] == "Disbursements"


@pytest.mark.parametrize("member", ["table", "stat_card", "note_narration", "source_card"])
def test_only_a_chart_carries_a_caption(member: str) -> None:
    view = _shell(resolved_revision(member))

    assert not any(node.get("cssClass") == "dfo-caption" for node in _nodes(view["view"]))


def test_a_chart_whose_axis_names_no_metric_has_no_caption() -> None:
    envelope = resolved_revision("line_chart")
    envelope["component"]["y_axis"]["label"] = ""

    view = _shell(envelope)

    assert not any(node.get("cssClass") == "dfo-caption" for node in _nodes(view["view"]))


def test_the_face_source_line_joins_names_and_counts_the_rest() -> None:
    envelope = resolved_revision("line_chart")
    (one,) = envelope["provenance"]["sources"]
    envelope["provenance"]["sources"] = [
        {**one, "id": f"src_{i}", "name": f"Source {i}"} for i in range(7)
    ]

    view = _shell(envelope)

    assert (
        _source_line(view)
        == "Source 0 · Source 1 · Source 2 · Source 3 · Source 4 · 2 more sources"
    )


def test_more_info_shows_each_warning_whole() -> None:
    envelope = resolved_revision("line_chart")
    detail = "Figures for 2018 were withheld by the reporter. " * 5
    envelope["provenance"]["warnings"] = [{"kind": "partial_coverage", "detail": detail}]

    view = _shell(envelope)

    assert f"partial coverage: {detail}" in _details(view)
    assert not any("withheld" in text for text in _face_texts(view))
    assert _source_line(view) == "OECD DAC · 1 caveat"


def test_more_info_shows_every_warning_and_the_source_line_counts_them() -> None:
    envelope = resolved_revision("line_chart")
    envelope["provenance"]["warnings"] = [
        {"kind": "provisional_data", "detail": f"Figures for {year} are provisional."}
        for year in range(2020, 2025)
    ]

    view = _shell(envelope)

    details = _details(view)
    assert [text for text in details if text.startswith("provisional data: ")] == [
        f"provisional data: Figures for {year} are provisional." for year in range(2020, 2025)
    ]
    assert not any("more warnings" in text for text in details)
    assert _source_line(view) == "OECD DAC · 5 caveats"


def test_a_card_without_sources_counts_its_caveats_after_more_info() -> None:
    envelope = resolved_revision("note_narration")
    envelope["provenance"]["warnings"] = [{"kind": "no_data", "detail": "One gap."}] * 2

    assert _source_line(_shell(envelope)) == "More info · 2 caveats"


_CRS = "OECD Creditor Reporting System (CRS)"
_CRS_SOURCE = f"{_CRS}, licence OECD Terms and Conditions, retrieved 2026-09-28"
_CRS_CITATION = f"{_CRS} via ONE DFO, retrieved 2026-09-28"


@pytest.mark.parametrize(
    ("fixture", "source_line", "last_details"),
    [
        (
            "line_chart_description",
            f"{_CRS} · 1 caveat",
            [
                _CRS_SOURCE,
                "Data through 2024.",
                "provisional: Values of infectious disease control in 2024 are provisional.",
                _CRS_CITATION,
            ],
        ),
        (
            "source_card_description",
            _CRS,
            [
                "Last refreshed 2026-09-28; a newer source download on 2026-10-08 is not in this card; next update expected 2026-12.",
                _CRS_CITATION,
            ],
        ),
    ],
)
def test_a_card_with_lineage_alone_in_its_provenance_reads_its_description(
    fixture: str, source_line: str, last_details: list[str]
) -> None:
    envelope = resolved_revision(fixture)

    view = _shell(envelope)

    details = _details(view)
    assert _source_line(view) == source_line
    assert details[len(details) - len(last_details) :] == last_details
    if envelope["component"]["type"] == "source_card":
        assert _CRS_SOURCE in _face_texts(view)
    model = chart_model(envelope)
    assert model is None or model.source_line == source_line


def test_a_narration_note_with_lineage_alone_draws_the_mark_without_more_info() -> None:
    view = _shell(resolved_revision("note_narration_lineage"))

    footer = view["view"]["children"][0]["children"][-1]
    assert footer["cssClass"] == "dfo-chrome"
    assert [node["type"] for node in _nodes(footer)] == ["Div", "Span"]
    assert not any(node["type"] == "Accordion" for node in _nodes(view["view"]))


def test_a_non_https_source_is_named_but_not_linked() -> None:
    envelope = resolved_revision("line_chart")
    envelope["provenance"]["sources"][0]["url"] = "javascript:alert(1)"

    view = _shell(envelope)

    assert not any(node["type"] == "Link" for node in _nodes(view["view"]))
    assert "OECD DAC, licence CC BY 4.0, retrieved 2026-06-12" in _texts(view)


def test_payload_text_never_reaches_the_renderer_as_a_template() -> None:
    envelope = resolved_revision("table")
    envelope["component"]["title"]["text"] = "{{ state }} and {{{ more }}}"
    envelope["component"]["rows"][0]["dimensions"]["recipient"] = "{{ rows }}"

    view = _shell(envelope)
    data = app_data(envelope)

    bindings = {"{{ !loaded }}", "{{ data.rows }}"}
    assert all(leaf in bindings for leaf in _leaves(view) if isinstance(leaf, str) and "{{" in leaf)
    assert not any(isinstance(leaf, str) and "{{" in leaf for leaf in _leaves(data))
    assert _only(view, "H3")["content"].replace("\u200b", "") == "{{ state }} and {{{ more }}}"
    assert data["rows"][0]["c0"].replace("\u200b", "") == "{{ rows }}"


def test_an_unknown_member_draws_its_description_and_back_of_card_only() -> None:
    envelope = resolved_revision("stat_card")
    envelope["component"] = {"type": "scatter_chart", "title": {"text": "Not drawn"}}

    view = _shell(envelope)

    texts = _texts(view)
    assert (
        _only(view, "H3")["content"]
        == "A scatter_chart component, a type this server does not draw"
    )
    assert "Not drawn" not in texts
    assert "state" not in view
    assert app_data(envelope) == {}


def test_a_notice_view_is_light_too() -> None:
    view = notice_view("Nothing to draw.", base_url=BASE_URL)

    assert view["mode"] == "light"
    assert _only(view, "Text")["content"] == "Nothing to draw."


def _assert_carries_the_theme(view: dict[str, Any]) -> None:
    assert view["stylesheets"] == [FONTS_URL, theme_url(BASE_URL)]
    assert view["stylesheets"][1].startswith(f"{BASE_URL}/theme-")
    assert "css" not in view
    assert view["mode"] == "light"


@pytest.mark.parametrize("member", MEMBERS)
def test_every_shell_names_the_fonts_and_theme_by_url_in_light_mode(member: str) -> None:
    """Members with data (the charts, table, and stat card) and without
    (`note_narration`, `source_card`) build their `PrefabApp` alike."""
    _assert_carries_the_theme(_shell(resolved_revision(member)))


def test_the_notice_view_names_the_fonts_and_theme_by_url_in_light_mode() -> None:
    _assert_carries_the_theme(notice_view("Nothing to draw.", base_url=BASE_URL))


def test_the_notice_view_draws_its_text_as_written() -> None:
    view = notice_view("Refused: {{ state }}", base_url=BASE_URL)

    assert _only(view, "Text")["content"].replace("\u200b", "") == "Refused: {{ state }}"
    assert not any(isinstance(leaf, str) and "{{" in leaf for leaf in _leaves(view["view"]))


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
def test_series_colours_are_hex_literals_a_renderer_needs_no_stylesheet_to_resolve(
    member: str,
) -> None:
    view = _shell(resolved_revision(member))

    colours = [
        series["color"]
        for series in _only(view, "LineChart" if member == "line_chart" else "BarChart")["series"]
    ]
    swatches = [value for _, style in _legend(view) for value in style.values()]
    assert colours
    assert all(re.fullmatch(r"#[0-9a-f]{6}", value) for value in [*colours, *swatches])
    assert "var(" not in json.dumps(view["view"])
