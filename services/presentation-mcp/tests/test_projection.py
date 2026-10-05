"""Text projection tests, one per catalogue member, the bounds, and escaping.

Inputs are dfo-spec-kit's catalogue fixtures (`stub_assembler.resolved_revision`).
Each member's projection is pinned whole in `fixtures/projection/<member>.txt`,
a `.txt` so the Markdown formatter leaves it byte-exact. The bound tests
build the largest payloads the catalogue admits, with four-byte characters
and Markdown-active characters in every string, and hold the projection to
`MAX_PROJECTION_BYTES`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from presentation_mcp.projection import (
    MAX_PROJECTION_BYTES,
    MAX_ROWS,
    MAX_TEXT_CHARS,
    MAX_VALUE_COLUMNS,
    PROJECTION_CUT_NOTICE,
    ROWLESS_NOTICE,
    delta_projection,
    describe,
    text_projection,
    warning_texts,
)

from .stub_assembler import resolved_revision

MEMBERS = ("line_chart", "bar_chart", "table", "stat_card", "note_narration", "source_card")
_GOLDEN = Path(__file__).parent / "fixtures" / "projection"

# Four bytes in UTF-8, so a clipped string is as large in bytes as it gets.
_WIDE = "\N{GLOBE WITH MERIDIANS}"


def _wide(length: int) -> str:
    return _WIDE * length


@pytest.mark.parametrize("member", MEMBERS)
def test_each_member_projects_to_its_pinned_markdown(member: str) -> None:
    envelope = resolved_revision(member)

    projection = text_projection(envelope)

    assert projection == (_GOLDEN / f"{member}.txt").read_text(encoding="utf-8").rstrip("\n")


@pytest.mark.parametrize("member", MEMBERS)
def test_each_projection_is_stamped_and_cited(member: str) -> None:
    envelope = resolved_revision(member)

    first, *_, last = text_projection(envelope).splitlines()

    assert f"`{envelope['component_id']}`, revision {envelope['revision']}" in first
    assert first.startswith(describe(envelope["component"]))
    assert last == f"Source: {envelope['provenance']['citation']}"


@pytest.mark.parametrize("member", ["line_chart", "bar_chart", "table"])
def test_a_rowless_projection_puts_the_notice_in_place_of_the_data_table(member: str) -> None:
    envelope = resolved_revision(member)
    blocks = text_projection(envelope).split("\n\n")

    rowless = text_projection(envelope, rowless_notice=ROWLESS_NOTICE)

    assert rowless == f"{blocks[0]}\n\n{ROWLESS_NOTICE}\n\n{blocks[-1]}"
    assert "| --- |" not in rowless


@pytest.mark.parametrize("member", ["stat_card", "note_narration", "source_card"])
def test_a_rowless_projection_keeps_every_other_body_whole(member: str) -> None:
    envelope = resolved_revision(member)

    assert text_projection(envelope, rowless_notice=ROWLESS_NOTICE) == text_projection(envelope)


def test_a_rowless_projection_of_a_cut_payload_drops_the_cut_line() -> None:
    envelope = _worst_series("line_chart", series=24, points=500)

    rowless = text_projection(envelope, rowless_notice=ROWLESS_NOTICE)

    assert ROWLESS_NOTICE in rowless
    assert "Showing" not in rowless
    assert len(rowless.encode()) <= MAX_PROJECTION_BYTES


def test_warning_texts_clip_each_detail_unless_the_limit_is_none() -> None:
    detail = "word " * 60
    provenance = {"warnings": [{"kind": "facet_axis", "detail": detail}]}

    (clipped,) = warning_texts(provenance)
    (whole,) = warning_texts(provenance, limit=None)
    (short,) = warning_texts(provenance, limit=10)

    assert clipped == f"facet axis: {detail.strip()[: MAX_TEXT_CHARS - 1]}…"
    assert whole == f"facet axis: {detail}"
    assert short == "facet axis: word word…"


def test_warning_texts_count_the_rest_past_the_cap_unless_the_count_is_none() -> None:
    provenance = {"warnings": [{"kind": "no_data", "detail": f"Gap {i}."} for i in range(5)]}

    capped = warning_texts(provenance)
    every = warning_texts(provenance, count=None)

    assert capped == ["no data: Gap 0.", "no data: Gap 1.", "no data: Gap 2.", "2 more warnings"]
    assert every == [f"no data: Gap {i}." for i in range(5)]


def test_every_line_chart_point_appears_in_its_period_row() -> None:
    envelope = resolved_revision("line_chart")
    projection = text_projection(envelope)

    for point in envelope["component"]["series"][0]["points"]:
        cell = point.get("status") or f"{point['value']:g}"
        assert f"| {point['period']} | {cell} |" in projection


def test_the_title_is_announced_as_the_models() -> None:
    projection = text_projection(resolved_revision("line_chart"))

    assert "Title, written by the model: Malaria grant disbursements to Ethiopia" in projection


def test_narration_is_labelled_as_the_models_voice() -> None:
    envelope = resolved_revision("note_narration")

    projection = text_projection(envelope)

    assert f"Model narration: {envelope['component']['text']}" in projection


def test_an_unknown_member_projects_its_description_and_footer_only() -> None:
    envelope = resolved_revision("stat_card")
    envelope["component"] = {"type": "scatter_chart", "title": {"text": "Not drawn"}}

    projection = text_projection(envelope)

    assert projection.splitlines()[0].startswith(
        "A scatter\\_chart component, a type this server does not draw"
    )
    assert "Not drawn" not in projection
    assert projection.endswith("Source: OECD DAC via ONE DFO, retrieved 12 June 2026")


def test_markdown_active_characters_in_payload_text_are_escaped() -> None:
    envelope = resolved_revision("bar_chart")
    envelope["component"]["title"]["text"] = "**Aid** | [link](x)"
    envelope["component"]["series"][0]["label"] = "A|B"

    projection = text_projection(envelope)

    assert "Title, written by the model: \\*\\*Aid\\*\\* \\| \\[link\\](x)" in projection
    assert "| A\\|B | 441.9 |" in projection


def test_units_that_differ_are_named_per_series() -> None:
    envelope = resolved_revision("bar_chart")
    envelope["component"]["series"][1]["unit"] = "percent"

    projection = text_projection(envelope)

    assert (
        "Units: Ethiopia: USD millions, constant 2023 prices; Ghana: percent; "
        "Kenya: USD millions, constant 2023 prices."
    ) in projection


def test_warnings_and_a_partial_period_reach_the_footer() -> None:
    envelope = resolved_revision("line_chart")
    envelope["provenance"]["freshness"]["partial_period"] = True
    envelope["provenance"]["warnings"] = [
        {"kind": "coverage_gap", "detail": "2018 is not reported."}
    ]

    projection = text_projection(envelope)

    assert "Data through 2024, the last period partial; last refreshed 2026-06-12" in projection
    assert "Warning, coverage gap: 2018 is not reported." in projection


def test_a_non_https_source_url_is_left_out() -> None:
    envelope = resolved_revision("source_card")
    envelope["provenance"]["sources"][0]["url"] = "http://stats.oecd.org/"

    projection = text_projection(envelope)

    assert "- OECD DAC, licence CC BY 4.0, retrieved 2026-06-12\n" in projection
    assert "http://" not in projection


def test_a_delta_names_each_changed_field_and_is_stamped() -> None:
    envelope = resolved_revision("line_chart")
    changed = [
        {"field": "scale", "prior": "linear", "new": "log"},
        {"field": "title"},
        {"field": "subtitle", "prior": None, "new": "*Aid*"},
        {"field": "caption", "prior": "old", "new": None},
    ]

    delta = delta_projection(envelope, changed)

    assert delta == (
        "scale linear to log; title changed; subtitle set to \\*Aid\\*; caption cleared, "
        f"values unchanged (`{envelope['component_id']}`, revision {envelope['revision']})."
    )


def test_a_delta_for_a_revise_that_changed_nothing_says_so() -> None:
    envelope = resolved_revision("line_chart")

    assert delta_projection(envelope, []) == (
        f"Nothing changed (`{envelope['component_id']}`, revision {envelope['revision']})."
    )


# --- bounds -------------------------------------------------------------------


def _worst_series(member: str, *, series: int, points: int) -> dict[str, Any]:
    envelope = resolved_revision(member)
    template = envelope["component"]["series"][0]
    envelope["component"]["title"]["text"] = _wide(200)
    envelope["component"]["series"] = [
        {
            **copy.deepcopy(template),
            "id": f"s{i}",
            "label": f"{i}{_wide(199)}",
            "unit": _wide(200),
            "points": [
                {"period": f"{1500 + p}", "value": -123456789.123456} for p in range(points)
            ],
        }
        for i in range(series)
    ]
    return _worst_footer(envelope)


def _worst_footer(envelope: dict[str, Any]) -> dict[str, Any]:
    provenance = envelope["provenance"]
    provenance["citation"] = "*" * 2000
    provenance["warnings"] = [{"kind": "provisional", "detail": _wide(2000)}] * 50
    return envelope


def test_the_measured_worst_line_chart_is_bounded_and_says_what_it_cut() -> None:
    envelope = _worst_series("line_chart", series=24, points=500)

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    assert f"and {MAX_VALUE_COLUMNS} of 24 series" in projection
    assert "the full data is in frame `spec_9f2e41a7c03b58d2a4c1f6b2e8d35790`" in projection
    # The latest periods are the ones kept.
    assert "| 1999 |" in projection
    assert "| 1500 |" not in projection


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
def test_the_largest_admissible_series_payload_is_bounded(member: str) -> None:
    envelope = _worst_series(member, series=50, points=500)
    if member == "bar_chart":
        envelope["component"]["category_axis"] = {
            "mode": "period",
            "window": {"start": "1500", "end": "1999"},
        }

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    assert f"and {MAX_VALUE_COLUMNS} of 50 series" in projection


def test_the_largest_admissible_entity_bar_chart_is_bounded() -> None:
    envelope = _worst_series("bar_chart", series=50, points=1)

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    assert "of 50 bars" in projection


@pytest.mark.parametrize("series", [24, 50])
def test_an_entity_bar_chart_whose_bars_differ_in_unit_is_bounded(series: int) -> None:
    envelope = _worst_series("bar_chart", series=series, points=1)
    component = envelope["component"]
    component["value_axis"]["label"] = _wide(200)
    component["category_axis"]["category_dimension"] = _wide(200)
    for i, one in enumerate(component["series"]):
        one["unit"] = f"{i}{_wide(199)}"

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    units = next(line for line in projection.splitlines() if line.startswith("Units: "))
    assert units.count("; ") == MAX_ROWS - 1
    assert f"of {series} bars" in projection
    assert projection.splitlines()[-1].startswith("Source: ")


def test_a_projection_past_the_bound_is_cut_at_a_whole_line_and_says_so() -> None:
    envelope = resolved_revision("line_chart")
    envelope["provenance"]["freshness"]["data_through"] = _wide(5000)

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    assert projection.endswith(f"\n\n{PROJECTION_CUT_NOTICE}")
    assert projection.startswith(text_projection(resolved_revision("line_chart")).split("\n\n")[0])


def test_the_largest_admissible_table_is_bounded_and_says_what_it_cut() -> None:
    envelope = _worst_footer(resolved_revision("table"))
    component = envelope["component"]
    dimensions = [f"d{i}" for i in range(30)]
    component["title"]["text"] = _wide(200)
    component["columns"] = [
        *({"key": key, "label": _wide(200)} for key in dimensions),
        {"key": "period", "label": "Period"},
        {"key": "value", "label": "Value"},
    ]
    component["rows"] = [
        {
            "series_id": "s",
            "dimensions": dict.fromkeys(dimensions, _wide(200)),
            "period": "2024",
            "source_ref": "src_oecd_dac",
            "unit": f"{i % 2}{_wide(199)}",
            "value": -123456789.123456,
        }
        for i in range(2000)
    ]

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    assert "of 2,000 rows" in projection
    assert f"and {MAX_VALUE_COLUMNS + 2} of 32 columns" in projection


def test_a_short_table_keeps_every_row_up_to_the_row_cap() -> None:
    envelope = resolved_revision("table")
    rows = envelope["component"]["rows"]
    envelope["component"]["rows"] = [
        {**rows[0], "period": f"{2000 + i}"} for i in range(MAX_ROWS + 5)
    ]

    projection = text_projection(envelope)

    assert f"Showing the first {MAX_ROWS} of {MAX_ROWS + 5} rows" in projection


def test_the_longest_narration_is_kept_whole_and_bounded() -> None:
    envelope = _worst_footer(resolved_revision("note_narration"))
    envelope["component"]["text"] = _wide(2000)

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    assert _wide(2000) in projection


def test_the_largest_admissible_source_card_is_bounded() -> None:
    envelope = _worst_footer(resolved_revision("source_card"))
    sources = [
        {
            "id": f"src_{i}",
            "name": _wide(200),
            "url": "https://example.org/" + "a" * 2000,
            "licence": _wide(200),
            "retrieved": "2026-06-12",
        }
        for i in range(20)
    ]
    envelope["provenance"]["sources"] = sources
    envelope["component"]["source_refs"] = [source["id"] for source in sources]

    projection = text_projection(envelope)

    assert len(projection.encode()) <= MAX_PROJECTION_BYTES
    assert "Showing 5 of 20 sources" in projection
