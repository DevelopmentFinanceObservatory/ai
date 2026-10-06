"""Tests for `data_mcp.get_data_digest`: rendering a stripped mint manifest."""

from __future__ import annotations

from typing import Any

import pytest
from common.materialisation import FrameRef
from data_mcp.frame_materialisation import DegradedReason, MaterialisationOutcome
from data_mcp.get_data_digest import UnheldRegionSummary, render_get_data_digest
from data_mcp.manifest import Manifest, SeriesStatisticsModel, strip_manifest


def _manifest_payload(
    *,
    spec_id: str = "spec_2dffbc390c1c97d9992ee162ce690b19",
    shape_id: str = "MalariaGrantDisbursements",
    entities: dict[str, list[str]] | None = None,
    rows: int = 5,
    dimensions: list[dict[str, Any]] | None = None,
    window: dict[str, Any] | None = None,
    content_version: str = "cv_1",
    citation_text: str = "OECD DAC2a",
    citation_sources: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    requested_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "content_version": content_version,
        "coverage": {
            "rows": rows,
            "dimensions": dimensions
            if dimensions is not None
            else [{"count": 1, "label": "recipient"}],
            "window": window if window is not None else {"start_year": 2015, "end_year": 2024},
        },
        "citation": {
            "text": citation_text,
            "sources": (
                citation_sources
                if citation_sources is not None
                else [{"ref": "OECD_DAC2A", "label": "OECD DAC2a"}]
            ),
        },
        "trace": {},
        "warnings": warnings if warnings is not None else [],
        "artifacts": [],
    }
    payload["definition"] = {
        "schema_version": 1,
        "spec_id": spec_id,
        "identity": {
            "identity_version": "spec1",
            "shape": {"id": shape_id, "kind": "stat_var"},
            "entities": entities if entities is not None else {"recipient": ["country/KEN"]},
            "facets": {},
            "requested_window": requested_window
            if requested_window is not None
            else {"kind": "all"},
            "slots": [],
        },
    }
    return payload


def _frame_ref(manifest: Manifest) -> FrameRef:
    """The manifest's own pair, the default rendering's frame reference. Built
    unvalidated, since these payloads use short `content_version`s."""
    spec_id = manifest.definition.spec_id if manifest.definition is not None else ""
    return FrameRef.model_construct(spec_id=spec_id, content_version=manifest.content_version)


def _digest(
    *,
    statistics: list[dict[str, Any]] | None = None,
    statistics_fetch_failed: bool = False,
    **overrides: Any,
) -> str:
    """Render a digest for a manifest payload built from `overrides`.

    `statistics` stands in for what `get_data_tool.py` would have already
    fetched off the manifest's own statistics artifact before calling
    `render_get_data_digest`: `None` means `manifest.series_count` is `None`
    too (no statistics section at all, matching an old record or a mint that
    never resolved to any series); a list sets `series_count` to its length
    and, unless `statistics_fetch_failed`, feeds the parsed rows straight
    through as `series_stats`.
    """
    payload = _manifest_payload(**overrides)
    payload["series_count"] = None if statistics is None else len(statistics)
    manifest = strip_manifest(payload)
    series_stats = (
        None
        if statistics is None
        else [SeriesStatisticsModel.model_validate(s) for s in statistics]
    )
    return render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
        series_stats=series_stats,
        statistics_fetch_failed=statistics_fetch_failed,
    )


def _series(
    *,
    stat_var: str = "dcid:MalariaGrantDisbursements",
    entities: list[dict[str, str]] | None = None,
    facet: dict[str, Any] | None = None,
    observations: int = 2,
    first: tuple[float, str] = (12_400_000.0, "2015"),
    latest: tuple[float, str] = (9_800_000.0, "2024"),
    minimum: tuple[float, str] = (7_100_000.0, "2020"),
    maximum: tuple[float, str] = (14_200_000.0, "2017"),
    change: tuple[float, float | None] | None = (-2_600_000.0, -21.0),
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "stat_var": stat_var,
        "entities": (
            entities if entities is not None else [{"role": "recipient", "ref": "country/TGO"}]
        ),
        "facet": facet if facet is not None else {"unit": "dcid:USDollar"},
        "observations": observations,
        "first": {"value": first[0], "period": first[1]},
        "latest": {"value": latest[0], "period": latest[1]},
        "minimum": {"value": minimum[0], "period": minimum[1]},
        "maximum": {"value": maximum[0], "period": maximum[1]},
    }
    if change is not None:
        entry["change"] = {"absolute": change[0]}
        if change[1] is not None:
            entry["change"]["percent"] = change[1]
    return entry


# Acknowledgement line


def test_acknowledgement_names_the_shape() -> None:
    digest = _digest(shape_id="MalariaGrantDisbursements")

    assert "MalariaGrantDisbursements" in digest


def test_acknowledgement_substitutes_a_resolved_label_for_a_dcid() -> None:
    manifest = strip_manifest(_manifest_payload(entities={"recipient": ["country/KEN"]}))

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={"country/KEN": "Kenya"},
        dropped_roles=[],
        unheld_regions=[],
    )

    assert "recipient Kenya" in digest
    assert "country/KEN" not in digest


def test_acknowledgement_falls_back_to_the_raw_dcid_without_a_label() -> None:
    manifest = strip_manifest(_manifest_payload(entities={"recipient": ["country/KEN"]}))

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
    )

    assert "recipient country/KEN" in digest


def test_acknowledgement_joins_multiple_dcids_under_one_role() -> None:
    manifest = strip_manifest(
        _manifest_payload(entities={"recipient": ["country/KEN", "country/UGA"]})
    )

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={"country/KEN": "Kenya", "country/UGA": "Uganda"},
        dropped_roles=[],
        unheld_regions=[],
    )

    assert "recipient Kenya, Uganda" in digest


@pytest.mark.parametrize(
    ("dimensions", "noun"),
    [
        # No dimension sits on the open axis, so the axis noun stands.
        ([{"count": 1, "label": "recipient", "axis": "destination"}], "provider"),
        # A CRS-shaped response declares the financingProvider column on the open axis.
        (
            [
                {"count": 4, "label": "financingProvider", "axis": "origin"},
                {"count": 1, "label": "recipient", "axis": "destination"},
            ],
            "financingProvider",
        ),
        # An IDS-shaped response declares a creditor column there.
        (
            [
                {"count": 4, "label": "creditor", "axis": "origin"},
                {"count": 1, "label": "debtor", "axis": "destination"},
            ],
            "creditor",
        ),
    ],
    ids=["no-axis-column", "crs", "ids"],
)
def test_acknowledgement_names_an_open_role_beside_the_bound_entities(
    dimensions: list[dict[str, Any]], noun: str
) -> None:
    payload = _manifest_payload(entities={"destination": ["country/NGA"]}, dimensions=dimensions)
    payload["definition"]["identity"]["open_roles"] = ["origin"]
    manifest = strip_manifest(payload)

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={"country/NGA": "Nigeria"},
        dropped_roles=[],
        unheld_regions=[],
    )

    assert digest.splitlines()[0] == (
        f"Retrieved MalariaGrantDisbursements for destination Nigeria · every {noun}."
    )


def test_acknowledgement_omits_a_role_with_no_entities() -> None:
    manifest = strip_manifest(_manifest_payload(entities={"recipient": []}))

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
    )

    assert "recipient" not in digest.splitlines()[0]


# Coverage line


def test_coverage_line_renders_rows_dimensions_and_window() -> None:
    digest = _digest(
        rows=42,
        dimensions=[{"count": 2, "label": "recipient"}, {"count": 10, "label": "years"}],
        window={"start_year": 2015, "end_year": 2024},
    )

    assert "42 rows" in digest
    assert "2 recipient" in digest
    assert "10 years" in digest
    assert "2015-2024" in digest


def test_coverage_line_singular_row() -> None:
    digest = _digest(rows=1)

    assert "1 row" in digest
    assert "1 rows" not in digest


def test_coverage_line_handles_a_missing_window() -> None:
    digest = _digest(window=None)

    assert "Coverage:" in digest


def test_coverage_line_states_the_requested_window_when_it_falls_short() -> None:
    digest = _digest(
        window={"start_year": 2015, "end_year": 2022},
        requested_window={"kind": "range", "start_year": 2015, "end_year": 2023},
    )

    assert "requested 2015-2023, served 2015-2022" in digest


def test_coverage_line_states_only_the_served_window_when_it_matches_the_request() -> None:
    digest = _digest(
        window={"start_year": 2015, "end_year": 2024},
        requested_window={"kind": "range", "start_year": 2015, "end_year": 2024},
    )

    assert "2015-2024" in digest
    assert "requested" not in digest


def test_coverage_line_ignores_a_non_range_requested_window() -> None:
    digest = _digest(
        window={"start_year": 2015, "end_year": 2022}, requested_window={"kind": "latest"}
    )

    assert "2015-2022" in digest
    assert "requested" not in digest


# Frame reference


def test_frame_reference_line_carries_spec_id_and_content_version_from_the_manifest() -> None:
    digest = _digest(
        spec_id="spec_75bacda207b42d5632b1502efc37fa2b", content_version="cv_re_minted"
    )

    assert "spec_75bacda207b42d5632b1502efc37fa2b" in digest
    assert "cv_re_minted" in digest


# Citation line


def test_citation_line_renders_text_and_source_labels() -> None:
    digest = _digest(
        citation_text="OECD DAC2a",
        citation_sources=[{"ref": "OECD_DAC2A", "label": "OECD DAC table 2a"}],
    )

    assert "OECD DAC2a" in digest
    assert "OECD DAC table 2a" in digest


def test_citation_line_with_no_sources_omits_the_parens() -> None:
    digest = _digest(citation_text="OECD DAC2a", citation_sources=[])

    assert "Source: OECD DAC2a." in digest


# Dropped-role notes


def test_dropped_role_note_names_the_role() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=["provider"],
        unheld_regions=[],
    )

    assert "provider" in digest
    assert "no longer bound" in digest


def test_no_dropped_role_notes_when_the_list_is_empty() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
    )

    assert "no longer bound" not in digest


def test_multiple_dropped_roles_each_get_their_own_note() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=["provider", "donor"],
        unheld_regions=[],
    )

    assert digest.count("no longer bound") == 2


# Unheld-region notes


def test_unheld_region_note_shows_all_names_under_the_cap() -> None:
    manifest = strip_manifest(_manifest_payload())
    region = UnheldRegionSummary(
        region_name="Sub-Saharan Africa", unheld_names=["Chad", "Mali"], unheld_count=2
    )

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[region],
    )

    assert "Sub-Saharan Africa" in digest
    assert "Chad" in digest
    assert "Mali" in digest
    assert "2 member" in digest
    assert "more" not in digest


def test_unheld_region_note_caps_names_and_reports_the_remainder() -> None:
    manifest = strip_manifest(_manifest_payload())
    names = [f"Country{i}" for i in range(45)]
    region = UnheldRegionSummary(region_name="A Continent", unheld_names=names, unheld_count=45)

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[region],
    )

    assert "45 member" in digest
    assert "Country0" in digest
    assert "Country1" in digest
    assert "Country2" in digest
    assert "Country3" not in digest
    assert "42 more" in digest


# Per-series statistics


def test_statistics_block_renders_stat_var_entities_unit_and_figures() -> None:
    digest = _digest(
        statistics=[
            _series(
                stat_var="dcid:MalariaGrantDisbursements",
                entities=[
                    {"role": "provider", "ref": "country/FRA"},
                    {"role": "recipient", "ref": "country/TGO"},
                ],
            )
        ]
    )

    assert "dcid:MalariaGrantDisbursements" in digest
    assert "provider country/FRA" in digest
    assert "recipient country/TGO" in digest
    assert "unit dcid:USDollar" in digest
    assert "first 12.4m (2015)" in digest
    assert "latest 9.8m (2024)" in digest
    assert "min 7.1m (2020)" in digest
    assert "max 14.2m (2017)" in digest
    assert "change over window -2.6m (-21%)" in digest


def test_statistics_single_observation_series_reports_first_as_latest_and_omits_change() -> None:
    digest = _digest(
        statistics=[
            _series(
                observations=1,
                first=(133713.37, "2015"),
                latest=(133713.37, "2015"),
                minimum=(133713.37, "2015"),
                maximum=(133713.37, "2015"),
                change=None,
            )
        ]
    )

    assert "first 133.7k (2015)" in digest
    assert "latest 133.7k (2015)" in digest
    assert "change over window" not in digest


def test_statistics_absent_manifest_omits_the_block_but_renders_everything_else() -> None:
    digest = _digest(statistics=None)

    assert "first" not in digest
    assert "Statistics" not in digest
    assert "Coverage:" in digest
    assert "Source:" in digest


def test_statistics_zero_series_omits_the_block_the_same_as_absent() -> None:
    digest = _digest(statistics=[])

    assert "first" not in digest
    assert "Statistics" not in digest


def test_statistics_fetch_failure_degrades_to_a_count_only_line() -> None:
    digest = _digest(statistics=[_series()], statistics_fetch_failed=True)

    assert "Statistics: 1 series retrieved" in digest
    assert "could not fetch or decode" in digest
    assert "first " not in digest


def test_statistics_render_up_to_the_eight_series_cap() -> None:
    series_list = [
        _series(stat_var=f"dcid:Var{i}", entities=[{"role": "recipient", "ref": f"country/C{i}"}])
        for i in range(8)
    ]

    digest = _digest(statistics=series_list)

    for i in range(8):
        assert f"dcid:Var{i}" in digest
        assert f"country/C{i}" in digest


def test_statistics_above_the_cap_renders_series_count_only() -> None:
    series_list = [
        _series(stat_var=f"dcid:Var{i}", entities=[{"role": "recipient", "ref": f"country/C{i}"}])
        for i in range(9)
    ]

    digest = _digest(statistics=series_list)

    assert "9 series" in digest
    for i in range(9):
        assert f"dcid:Var{i}" not in digest
    assert "12.4m" not in digest


def test_statistics_reads_the_manifests_own_entries_not_frame_dimensions() -> None:
    """Coverage claims 3 recipients worth of rows; only 2 statistics entries exist.

    A renderer that (incorrectly) derived series count from `coverage.dimensions`
    instead of reading `manifest.statistics` directly would render 3 blocks here.
    """
    digest = _digest(
        dimensions=[{"count": 3, "label": "recipient"}],
        statistics=[
            _series(stat_var="dcid:VarA", entities=[{"role": "recipient", "ref": "country/AAA"}]),
            _series(stat_var="dcid:VarB", entities=[{"role": "recipient", "ref": "country/BBB"}]),
        ],
    )

    assert digest.count("first ") == 2
    assert "VarA" in digest
    assert "VarB" in digest


# Unit readings


def test_sdg_current_price_unit_reads_currency_and_magnitude() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "SDG_CU_USD_M"})])

    assert "unit SDG_CU_USD_M (current prices, US dollars, millions)" in digest


def test_sdg_current_price_unit_without_a_magnitude_suffix() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "SDG_CU_USD"})])

    assert "unit SDG_CU_USD (current prices, US dollars)" in digest


def test_sdg_constant_price_unit_reads_the_base_year() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "SDG_CON_2015_EUR_K"})])

    assert "unit SDG_CON_2015_EUR_K (constant 2015 prices, euros, thousands)" in digest


def test_sdg_constant_price_unit_without_a_base_year() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "SDG_CON_GBP_B"})])

    assert "unit SDG_CON_GBP_B (constant prices, pounds sterling, billions)" in digest


def test_deployments_own_current_price_ref_reads_unchanged_code_plus_reading() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "USD"})])

    assert "unit USD (current prices, US dollars)" in digest


def test_deployments_own_constant_price_ref_reads_the_base_year() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "ConstantEUR_2023"})])

    assert "unit ConstantEUR_2023 (constant 2023 prices, euros)" in digest


def test_an_unrecognised_unit_code_renders_unchanged_with_no_reading() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "SDG_PER_1000_POP"})])

    assert "unit SDG_PER_1000_POP" in digest
    assert "SDG_PER_1000_POP (" not in digest


def test_a_family_lookalike_that_fails_the_strict_regex_renders_unchanged() -> None:
    """`SDG_KG_PER_CON_USD` contains the `CON` token but doesn't match the anchored
    price-basis form, so it must not be misread as a price-basis unit.
    """
    digest = _digest(statistics=[_series(facet={"unit": "SDG_KG_PER_CON_USD"})])

    assert "unit SDG_KG_PER_CON_USD" in digest
    assert "SDG_KG_PER_CON_USD (" not in digest


def test_dcid_qualified_unit_renders_unchanged_with_no_reading() -> None:
    digest = _digest(statistics=[_series(facet={"unit": "dcid:USDollar"})])

    assert "unit dcid:USDollar" in digest
    assert "dcid:USDollar (" not in digest


# Statistics block discipline: what varies with the frame, what doesn't


def test_statistics_block_is_unaffected_by_the_series_observation_count() -> None:
    """A block's size tracks the five figures it names, not how many rows fed them.

    `observations` swings from 1 to 10,000 -- four orders of magnitude -- while
    the five statistic figures and their periods stay fixed. The rendered
    block must come out byte-for-byte identical either way: `first`, `latest`,
    `minimum`, `maximum`, and `change` are the citable figures this block
    exists to carry, and nothing about how many rows produced them belongs in
    it.
    """
    few_rows = _series(observations=1)
    many_rows = _series(observations=10_000)

    assert _digest(statistics=[few_rows]) == _digest(statistics=[many_rows])


def test_statistics_block_never_renders_a_field_outside_the_named_five() -> None:
    """An entry carrying an extra numeric field must not leak it into the digest.

    `SeriesStatisticsModel` already drops an unrecognised field at validation
    (`extra="ignore"`), so this exercises the full path from a raw mint
    payload through to rendered text -- and would also catch a renderer that
    started iterating a series entry's fields generically instead of naming
    the five it means to print.
    """
    entry = _series()
    entry["raw_frame_total"] = 999999.99

    digest = _digest(statistics=[entry])

    assert "999999.99" not in digest
    assert "raw_frame_total" not in digest
    # The entry still rendered -- this isn't passing because nothing printed.
    assert "first 12.4m (2015)" in digest


def test_statistics_block_never_renders_a_facet_value_besides_unit() -> None:
    """`provenance`/`measurementMethod` ride along on the entry but aren't figures.

    A series' `facet` carries real data the manifest delivered, not an
    injection -- but only `unit` is part of the block's contract. A raw facet
    value that isn't `unit` is exactly the "value that is not one of the
    named statistics" this module must never print.
    """
    entry = _series(facet={"unit": "dcid:USDollar", "provenance": "OECD_DAC2A"})

    digest = _digest(statistics=[entry])

    assert "unit dcid:USDollar" in digest
    assert "OECD_DAC2A" not in digest


# Manifest warnings


def test_a_warning_message_reaches_the_digest() -> None:
    digest = _digest(
        warnings=[
            {
                "code": "facet_fallback",
                "severity": "info",
                "message": "Fell back to the default provenance.",
            }
        ]
    )

    assert "Fell back to the default provenance." in digest
    assert "facet_fallback" in digest


def test_no_warnings_renders_no_warning_text() -> None:
    digest = _digest(warnings=[])

    assert "Warning" not in digest


def test_warnings_are_capped_with_a_remainder_count() -> None:
    warnings = [{"code": f"code{i}", "severity": "info", "message": f"msg{i}"} for i in range(8)]

    digest = _digest(warnings=warnings)

    for i in range(5):
        assert f"msg{i}" in digest
    for i in range(5, 8):
        assert f"msg{i}" not in digest
    assert "3 more" in digest


def test_a_coverage_scope_warning_is_placed_after_coverage_and_before_the_frame_reference() -> None:
    digest = _digest(
        window={"start_year": 2015, "end_year": 2022},
        statistics=[_series()],
        warnings=[
            {
                "code": "coverage_below_scope",
                "severity": "warn",
                "message": "The served coverage falls short of the requested scope.",
            }
        ],
    )

    coverage_at = digest.index("Coverage:")
    warning_at = digest.index("coverage_below_scope")
    frame_at = digest.index("Frame: spec_id=")
    statistics_at = digest.index("first ")

    assert coverage_at < warning_at < frame_at < statistics_at


def test_a_non_coverage_warning_stays_at_the_end_after_statistics() -> None:
    digest = _digest(
        statistics=[_series()],
        warnings=[
            {"code": "facet_fallback", "severity": "info", "message": "Fell back to default."}
        ],
    )

    statistics_at = digest.index("first ")
    warning_at = digest.index("facet_fallback")

    assert statistics_at < warning_at


def test_coverage_and_non_coverage_warnings_each_land_in_their_own_place() -> None:
    digest = _digest(
        statistics=[_series()],
        warnings=[
            {
                "code": "coverage_below_scope",
                "severity": "warn",
                "message": "The served coverage falls short of the requested scope.",
            },
            {"code": "facet_fallback", "severity": "info", "message": "Fell back to default."},
        ],
    )

    coverage_at = digest.index("Coverage:")
    coverage_warning_at = digest.index("coverage_below_scope")
    frame_at = digest.index("Frame: spec_id=")
    statistics_at = digest.index("first ")
    other_warning_at = digest.index("facet_fallback")

    assert coverage_at < coverage_warning_at < frame_at < statistics_at < other_warning_at


def test_coverage_and_other_warnings_share_one_budget_of_five() -> None:
    coverage_warnings = [
        {"code": f"coverage_reason{i}", "severity": "warn", "message": f"covmsg{i}"}
        for i in range(5)
    ]
    other_warnings = [
        {"code": f"other_reason{i}", "severity": "info", "message": f"othermsg{i}"}
        for i in range(5)
    ]

    digest = _digest(warnings=coverage_warnings + other_warnings)

    for i in range(5):
        assert f"covmsg{i}" in digest
    for i in range(5):
        assert f"othermsg{i}" not in digest
    assert "5 more warning(s) not shown." in digest

    coverage_at = digest.index("Coverage:")
    warning_at = digest.index("covmsg0")
    frame_at = digest.index("Frame: spec_id=")
    assert coverage_at < warning_at < frame_at


def test_the_shared_budget_splits_across_both_groups_when_neither_fills_it() -> None:
    coverage_warnings = [
        {"code": f"coverage_reason{i}", "severity": "warn", "message": f"covmsg{i}"}
        for i in range(2)
    ]
    other_warnings = [
        {"code": f"other_reason{i}", "severity": "info", "message": f"othermsg{i}"}
        for i in range(4)
    ]

    digest = _digest(warnings=coverage_warnings + other_warnings)

    for i in range(2):
        assert f"covmsg{i}" in digest
    for i in range(3):
        assert f"othermsg{i}" in digest
    assert "othermsg3" not in digest
    assert "1 more warning(s) not shown." in digest

    coverage_at = digest.index("Coverage:")
    coverage_warning_at = digest.index("covmsg0")
    frame_at = digest.index("Frame: spec_id=")
    other_warning_at = digest.index("othermsg0")
    remainder_at = digest.rindex("1 more warning(s) not shown.")

    assert coverage_at < coverage_warning_at < frame_at < other_warning_at < remainder_at


# Summed StatVar family

_FAMILY_SPEC = "spec_" + "a" * 32

# `datasets_api.render`'s wire form of the warning.
_STAT_VARS_SUMMED_WARNING: dict[str, Any] = {
    "code": "stat_vars_summed",
    "severity": "info",
    "message": (
        "The series sums observations from more than one StatVar in the requested family. "
        f"3 StatVars were summed into dcid:{_FAMILY_SPEC}."
    ),
    "contributing_member_count": 3,
}


def _family_digest(
    *,
    warnings: list[dict[str, Any]],
    entities: dict[str, list[str]] | None = None,
    dimensions: list[dict[str, Any]] | None = None,
    series_entities: list[list[dict[str, str]]] | None = None,
) -> str:
    """A described family's digest, each series labelled `dcid:<spec_id>`."""
    payload = _manifest_payload(
        spec_id=_FAMILY_SPEC, warnings=warnings, entities=entities, dimensions=dimensions
    )
    payload["definition"]["identity"]["shape"] = {"kind": "statvar_schema"}
    per_series = series_entities or [[{"role": "recipient", "ref": "country/KEN"}]]
    payload["series_count"] = len(per_series)
    return render_get_data_digest(
        manifest=(manifest := strip_manifest(payload)),
        frame_ref=_frame_ref(manifest),
        entity_labels={"country/KEN": "Kenya"},
        dropped_roles=[],
        unheld_regions=[],
        series_stats=[
            SeriesStatisticsModel.model_validate(
                _series(stat_var=f"dcid:{_FAMILY_SPEC}", entities=series)
            )
            for series in per_series
        ],
    )


def test_a_summed_family_is_acknowledged_and_headed_as_its_total() -> None:
    digest = _family_digest(
        warnings=[_STAT_VARS_SUMMED_WARNING],
        dimensions=[{"label": "recipient", "count": 1, "axis": "destination"}],
    )

    assert digest.splitlines()[0] == (
        "Retrieved the requested StatVar family for recipient Kenya, "
        "with its StatVars summed into one total series per recipient."
    )
    assert (
        f"Summed total of the family [dcid:{_FAMILY_SPEC}] · recipient country/KEN · "
        "unit dcid:USDollar\n"
    ) in digest
    # The warning still names how many StatVars were summed.
    assert f"Warning [stat_vars_summed] (info): {_STAT_VARS_SUMMED_WARNING['message']}" in digest


@pytest.mark.parametrize("absent", [None, ""], ids=["null", "empty"])
def test_a_summed_family_with_no_usable_label_is_named_by_its_ref(absent: str | None) -> None:
    digest = _family_digest(warnings=[{**_STAT_VARS_SUMMED_WARNING, "stat_var_label": absent}])

    assert digest.splitlines()[0].startswith("Retrieved the requested StatVar family for ")
    assert f"Summed total of the family [dcid:{_FAMILY_SPEC}] · " in digest


def test_a_summed_family_is_named_by_the_label_its_warning_carries() -> None:
    digest = _family_digest(
        warnings=[{**_STAT_VARS_SUMMED_WARNING, "stat_var_label": "French education grants"}],
        dimensions=[{"label": "recipient", "count": 1, "axis": "destination"}],
    )

    assert digest.splitlines()[0] == (
        "Retrieved French education grants for recipient Kenya, "
        "with its StatVars summed into one total series per recipient."
    )
    assert (
        "\n\nSummed total of the family French education grants · recipient country/KEN · "
        "unit dcid:USDollar\n"
    ) in digest
    # The frame line still carries the spec_id the family's ref is built from.
    assert f"Frame: spec_id={_FAMILY_SPEC}" in digest


def test_a_summed_family_names_a_role_the_question_left_open() -> None:
    # "Top donors for education aid to Kenya": search binds only the
    # recipient, keyed by its flow axis, and leaves the provider role out of
    # the identity. dfo-apis keeps every declared role column, so the frame
    # holds one summed series per provider.
    digest = _family_digest(
        warnings=[_STAT_VARS_SUMMED_WARNING],
        entities={"destination": ["country/KEN"]},
        dimensions=[
            {"label": "financingProvider", "count": 2, "axis": "origin"},
            {"label": "financingRecipient", "count": 1, "axis": "destination"},
            {"label": "years", "count": 10},
        ],
        series_entities=[
            [
                {"role": "financingProvider", "ref": donor},
                {"role": "financingRecipient", "ref": "country/KEN"},
            ]
            for donor in ("country/FRA", "country/DEU")
        ],
    )

    assert digest.splitlines()[0] == (
        "Retrieved the requested StatVar family for destination Kenya, "
        "with its StatVars summed into one total series per provider and recipient."
    )
    assert (
        digest.count(f"Summed total of the family [dcid:{_FAMILY_SPEC}] · financingProvider ") == 2
    )


def test_a_summed_subject_family_reads_per_entity() -> None:
    digest = _family_digest(
        warnings=[_STAT_VARS_SUMMED_WARNING],
        entities={"observationAbout": ["country/KEN"]},
        dimensions=[{"label": "observationAbout", "count": 1}, {"label": "years", "count": 10}],
        series_entities=[[{"role": "observationAbout", "ref": "country/KEN"}]],
    )

    assert digest.splitlines()[0].endswith("summed into one total series per entity.")


def test_a_summed_family_with_no_role_column_reads_per_entity() -> None:
    digest = _family_digest(
        warnings=[_STAT_VARS_SUMMED_WARNING],
        entities={},
        dimensions=[{"label": "years", "count": 10}],
    )

    assert digest.splitlines()[0] == (
        "Retrieved the requested StatVar family, "
        "with its StatVars summed into one total series per entity."
    )


def test_the_summed_warning_shows_however_many_warnings_fill_the_budget() -> None:
    # Only the warning says how many StatVars were summed, so it keeps its
    # place whatever coverage and other warnings fill the budget.
    crowding = [
        {"code": f"coverage_reason{i}", "severity": "warn", "message": f"covmsg{i}"}
        for i in range(5)
    ] + [{"code": "other_reason", "severity": "info", "message": "othermsg"}]

    digest = _family_digest(warnings=[*crowding, _STAT_VARS_SUMMED_WARNING])

    assert f"Warning [stat_vars_summed] (info): {_STAT_VARS_SUMMED_WARNING['message']}" in digest
    assert digest.count("3 StatVars were summed") == 1
    assert "othermsg" not in digest
    assert "1 more warning(s) not shown." in digest


def test_the_summed_warning_takes_no_place_from_the_five_other_warnings() -> None:
    others = [
        {"code": f"other_reason{i}", "severity": "info", "message": f"othermsg{i}"}
        for i in range(5)
    ]

    digest = _family_digest(warnings=[_STAT_VARS_SUMMED_WARNING, *others])

    for i in range(5):
        assert f"othermsg{i}" in digest
    assert "3 StatVars were summed" in digest
    assert "more warning(s) not shown" not in digest


def test_a_family_series_without_the_summed_warning_keeps_the_plain_digest() -> None:
    digest = _family_digest(warnings=[])

    assert digest.splitlines()[0] == "Retrieved the requested data for recipient Kenya."
    assert f"\n\ndcid:{_FAMILY_SPEC} · recipient country/KEN · unit dcid:USDollar\n" in digest
    assert "summed" not in digest.lower()


# Degraded-frame note


def _degraded(*, attempts: int, reason: DegradedReason) -> MaterialisationOutcome:
    return MaterialisationOutcome(status="degraded", attempts=attempts, degraded_reason=reason)


def test_a_degraded_fetch_adds_a_note() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
        degraded_frame=_degraded(attempts=3, reason="fetch"),
    )

    assert "could not be retrieved after 3 attempt(s)" in digest
    assert "frame_not_held" in digest
    assert "Call get_data again" in digest
    assert "otherwise fetches and writes it again" in digest


def test_an_unconfirmed_store_write_says_the_artifact_was_fetched() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
        degraded_frame=_degraded(attempts=1, reason="store_write"),
    )

    assert "was fetched, but the frame store did not confirm the write" in digest
    assert "the write may still land" in digest
    assert "could not be retrieved" not in digest
    assert "frame_not_held" in digest
    assert "uses the frame if the store holds it by then" in digest


def test_a_fetch_after_an_unanswered_store_read_does_not_claim_nothing_is_held() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
        degraded_frame=_degraded(attempts=2, reason="store_unchecked"),
    )

    assert "could not be retrieved after 2 attempt(s)" in digest
    assert "could not check whether it already holds the frame" in digest
    assert "No bytes are held yet" not in digest


def test_degraded_fetch_zero_attempts_states_none_landed_rather_than_after_0() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
        degraded_frame=_degraded(attempts=0, reason="fetch"),
    )

    assert "without a single attempt reaching the artifact" in digest
    assert "after 0 attempt(s)" not in digest
    assert "frame_not_held" in digest


def test_no_degraded_frame_renders_no_note() -> None:
    manifest = strip_manifest(_manifest_payload())

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
        degraded_frame=None,
    )

    assert "could not be retrieved" not in digest


def test_ordinary_success_digest_is_unchanged_by_the_new_parameters() -> None:
    """The reference line and every other section render exactly as they do with no
    optional arguments supplied at all -- only an explicit `degraded_frame` adds
    anything new.
    """
    manifest = strip_manifest(_manifest_payload())

    with_defaults = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
    )
    with_explicit_none = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
        remint_persist_failed=False,
        degraded_frame=None,
    )

    assert with_defaults == with_explicit_none
    assert "Frame: spec_id=" in with_defaults
    assert "-- pass both to downstream tools." in with_defaults


# Entity and source caps hold digest length steady against frame breadth


def test_acknowledgement_caps_entities_per_role_with_an_and_more_tail() -> None:
    dcids = [f"country/C{i}" for i in range(12)]
    manifest = strip_manifest(_manifest_payload(entities={"recipient": dcids}))

    digest = render_get_data_digest(
        manifest=manifest,
        frame_ref=_frame_ref(manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
    )

    for i in range(8):
        assert f"country/C{i}" in digest
    for i in range(8, 12):
        assert f"country/C{i}" not in digest
    assert "and 4 more" in digest


def test_citation_sources_are_capped_with_an_and_more_tail() -> None:
    sources = [{"ref": f"SRC{i}", "label": f"Source {i}"} for i in range(8)]

    digest = _digest(citation_sources=sources)

    for i in range(5):
        assert f"Source {i}" in digest
    for i in range(5, 8):
        assert f"Source {i}" not in digest
    assert "and 3 more" in digest


def test_digest_length_does_not_track_frame_breadth() -> None:
    """A frame ten times wider than another must not make the digest ten times longer.

    Both fixtures sit past the entity and source caps already, so growing the
    underlying frame tenfold only changes the "and N more" remainder counts,
    not the digest's shape -- the same standard `test_digest_budget.py` holds
    `data_mcp.digest` to.
    """
    lean_manifest = strip_manifest(
        _manifest_payload(
            entities={"recipient": [f"country/C{i}" for i in range(9)]},
            citation_sources=[{"ref": f"SRC{i}", "label": f"Source {i}"} for i in range(6)],
        )
    )
    fat_manifest = strip_manifest(
        _manifest_payload(
            entities={"recipient": [f"country/C{i}" for i in range(90)]},
            citation_sources=[{"ref": f"SRC{i}", "label": f"Source {i}"} for i in range(60)],
        )
    )

    lean_digest = render_get_data_digest(
        manifest=lean_manifest,
        frame_ref=_frame_ref(lean_manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
    )
    fat_digest = render_get_data_digest(
        manifest=fat_manifest,
        frame_ref=_frame_ref(fat_manifest),
        entity_labels={},
        dropped_roles=[],
        unheld_regions=[],
    )

    assert abs(len(fat_digest) - len(lean_digest)) <= 10
