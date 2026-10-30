"""Tests for the `sources` tool: the catalogue digest, one dataset's documentation, and refusals."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder, sources_tool
from data_mcp.catalogue_counts import COUNTS_UNAVAILABLE
from data_mcp.server import mcp
from data_mcp.sources_tool import RESULT_CAP_CHARS
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client, _problem_body, text_of

Handler = Callable[[httpx2.Request], httpx2.Response]

CRS = "provenance/OECDCreditorReportingSystem"


@pytest.fixture(autouse=True)
def _fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clock before the fixture's next release, so no result depends on the run date."""
    monkeypatch.setattr(sources_tool, "_today", lambda: date(2026, 10, 11))


def _catalogue() -> dict[str, Any]:
    """A source catalogue as `GET /api/graph/sources` serves it: two publishers,
    one with a description-less dataset and an exclusion."""
    return {
        "sources": [
            {
                "source": {
                    "ref": "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
                    "label": "Organisation for Economic Co-operation and Development (OECD)",
                },
                "data_commons": {"id": "dfo", "label": "Development Finance Observatory"},
                "datasets": [
                    {
                        "ref": CRS,
                        "label": "OECD Creditor Reporting System (CRS)",
                        "provenances": [{"ref": CRS, "label": "OECD CRS"}],
                        "description": (
                            "The Creditor Reporting System (CRS) is the OECD's activity-level "
                            "database of development finance. Each record is a project."
                        ),
                        "license_type": "OECD Terms and Conditions",
                        "update_cadence": "Quarterly",
                        "years_covered": "1973-2024",
                        "source_url": "https://data-explorer.oecd.org/",
                    }
                ],
                "exclusions": [],
            },
            {
                "source": {"ref": "dc/s/WorldBank", "label": "World Bank"},
                "data_commons": {"id": "base", "label": "Data Commons by Google"},
                "datasets": [
                    {
                        "ref": "dc/base/WorldDevelopmentIndicators",
                        "label": "World Development Indicators",
                        "provenances": [],
                        "description": None,
                        "license_type": "Creative Commons Attribution License",
                        "update_cadence": None,
                        "years_covered": "1960-2024",
                        "source_url": None,
                    }
                ],
                "exclusions": [
                    {"ref": "dc/base/WorldBankIDS", "reason": "superseded_by_dfo"},
                ],
            },
        ]
    }


def _documentation(**overrides: Any) -> dict[str, Any]:
    """A source document as `GET /api/graph/documentation/{dcid}` serves it."""
    doc: dict[str, Any] = {
        "details": {
            "name_of_source": "OECD Creditor Reporting System (CRS)",
            "publisher": "Organisation for Economic Co-operation and Development (OECD)",
            "source": {"ref": "dc/s/Oecd", "label": "OECD"},
            "data_commons": {"id": "dfo", "label": "Development Finance Observatory"},
            "description": "The CRS is the OECD's activity-level database of development finance.",
            "source_url": "https://data-explorer.oecd.org/",
            "source_data_url": "https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",
            "citation": "OECD (2026), Creditor Reporting System (CRS).",
        },
        "license": {
            "type": "OECD Terms and Conditions",
            "url": "https://www.oecd.org/terms",
            "update_cadence": "Quarterly",
            "last_update": "2026-09-11",
            "next_expected_update": "2026-12",
        },
        "scope_and_units": {
            "countries": [],
            "years_covered": "1973-2024",
            "granularity": "Annual, by provider and recipient",
            "currency": "USD",
        },
        "methodology": {"description": None},
        "data_notes": [],
        "usage": {"api_access": None, "python": None},
        "transformation_log": [
            {
                "transformation": "Download the source data",
                "description": "The complete CRS file is downloaded each run. It replaces the copy.",
            }
        ],
        "version_history": [
            {
                "date": "2026-09-22",
                "title": "Exclude donor core contributions",
                "changes": [{"kind": "changed", "note": "Rows flagged bi_multi = 2 are excluded."}],
            }
        ],
        "gaps": [
            "scope_and_units.countries",
            "methodology.description",
            "data_notes",
            "usage.api_access",
            "usage.python",
        ],
        "document_errors": [],
    }
    doc.update(overrides)
    return doc


def _search_catalogue() -> dict[str, Any]:
    """The search catalogue as `GET /api/search/catalogue` serves it."""
    return {
        "catalogue_version": "2026-10-11T04:00:00Z",
        "counts": {
            "indicators": 108905,
            "datasets": 92,
            "publishers": 42,
            "countries": 230,
            "aid": {"donors": 172, "recipients": 235, "years": {"first": 1960, "last": 2024}},
        },
        "examples": [],
    }


_COUNTS_SENTENCE = (
    "DFO holds 108,905 indicators in 92 datasets from 42 publishers, with data for 230 "
    "countries. Aid data covers 172 donors and 235 recipients from 1960 to 2024."
)


def _serving(
    body: dict[str, Any],
    requests: list[httpx2.Request] | None = None,
    *,
    search_catalogue: Handler | None = None,
) -> Handler:
    """Serve `body` on the Graph API's doors and `search_catalogue` (by default
    `_search_catalogue()`) on the search catalogue route; `requests` records
    the Graph API reads."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/search/catalogue":
            if search_catalogue is not None:
                return search_catalogue(request)
            return httpx2.Response(200, json=_search_catalogue())
        if requests is not None:
            requests.append(request)
        return httpx2.Response(200, json=body)

    return handler


def _refusing(status: int, code: str, detail: str | None = None) -> Handler:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        extra = {"detail": detail} if detail is not None else {}
        return httpx2.Response(status, content=_problem_body(code=code, retryable=False, **extra))

    return handler


async def _call_sources(handler: Handler, ref: str | None = None) -> CallToolResult:
    arguments: dict[str, object] = {} if ref is None else {"ref": ref}
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            return await client.call_tool("sources", arguments, raise_on_error=False)


# The catalogue


async def test_the_catalogue_is_grouped_by_publisher_with_one_line_per_dataset() -> None:
    requests: list[httpx2.Request] = []

    result = await _call_sources(_serving(_catalogue(), requests))

    assert result.is_error is False
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/graph/sources"
    text = text_of(result)
    assert text.startswith(_COUNTS_SENTENCE + "\nDatasets by publisher, one line per dataset")
    assert (
        "## Organisation for Economic Co-operation and Development (OECD) "
        "[Development Finance Observatory]" in text
    )
    assert (
        f"- OECD Creditor Reporting System (CRS) [{CRS}] "
        "(1973-2024; Quarterly; OECD Terms and Conditions) -- The Creditor Reporting System "
        "(CRS) is the OECD's activity-level database of development finance." in text
    )
    # The second sentence of a description stays out of the catalogue.
    assert "Each record is a project" not in text


async def test_a_dataset_with_no_description_or_cadence_states_only_what_it_has() -> None:
    result = await _call_sources(_serving(_catalogue()))

    text = text_of(result)
    assert (
        "- World Development Indicators [dc/base/WorldDevelopmentIndicators] "
        "(1960-2024; Creative Commons Attribution License)\n" in text
    )
    assert "None" not in text


async def test_an_exclusion_superseded_by_dfo_is_counted_under_its_publisher() -> None:
    result = await _call_sources(_serving(_catalogue()))

    assert (
        "- 1 dataset(s) from this publisher are served through DFO's own version instead."
        in text_of(result)
    )


async def test_the_catalogue_answer_carries_the_counts_and_the_publishers() -> None:
    result = await _call_sources(_serving(_catalogue()))

    assert result.structured_content == {
        "counts": _search_catalogue()["counts"],
        "publishers": [
            "Organisation for Economic Co-operation and Development (OECD)",
            "World Bank",
        ],
    }


@pytest.mark.parametrize(
    "search_catalogue",
    [
        _refusing(503, "service_unavailable"),
        lambda _request: httpx2.Response(200, json={"catalogue_version": "v", "counts": {}}),
    ],
    ids=["no_catalogue_published", "wrong_shape"],
)
async def test_a_search_catalogue_that_cannot_be_read_opens_the_digest_with_no_count(
    search_catalogue: Handler,
) -> None:
    """The source catalogue still answers; the counts line says the counts are
    unavailable, and no count comes from anywhere else."""
    result = await _call_sources(_serving(_catalogue(), search_catalogue=search_catalogue))

    assert result.is_error is False
    assert text_of(result).startswith(COUNTS_UNAVAILABLE + "\n")
    assert result.structured_content is not None
    assert result.structured_content["counts"] is None


async def test_a_catalogue_of_the_wrong_shape_refuses_upstream_response_unusable() -> None:
    result = await _call_sources(_serving({"sources": [{"nope": True}]}))

    assert result.is_error is True
    assert result.structured_content is not None
    assert "did not match the shape this service expects" in result.structured_content["message"]


# One dataset's documentation


async def test_documentation_reads_the_documentation_door_for_the_ref() -> None:
    requests: list[httpx2.Request] = []

    result = await _call_sources(_serving(_documentation(), requests), ref=CRS)

    assert result.is_error is False
    assert requests[0].url.path == f"/api/graph/documentation/{CRS}"
    text = text_of(result)
    assert text.startswith(f"# OECD Creditor Reporting System (CRS) [{CRS}]")
    assert (
        "Publisher: Organisation for Economic Co-operation and Development (OECD); "
        "part of Development Finance Observatory." in text
    )
    assert (
        "Licence: OECD Terms and Conditions; updated quarterly. Terms: https://www.oecd.org/terms"
        "\n\nDownloaded by DFO 2026-09-11; Next release expected 2026-12.\n\n" in text
    )
    assert "Coverage: 1973-2024; Annual, by provider and recipient; USD." in text
    assert "Citation: OECD (2026), Creditor Reporting System (CRS)." in text
    assert (
        "How DFO transforms it:\n- Download the source data: The complete CRS file is "
        "downloaded each run." in text
    )
    assert (
        "Releases, newest first:\n- 2026-09-22: Exclude donor core contributions (1 change(s))"
        in text
    )
    assert "Data download: https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet." in text


@pytest.mark.parametrize(
    ("last_update", "next_expected_update", "today", "line"),
    [
        (
            "2026-09-11T04:12:00Z",
            "2026-12",
            date(2026, 12, 31),
            "Downloaded by DFO 2026-09-11; Next release expected 2026-12.",
        ),
        (
            "2026-09-11",
            "2026-12",
            date(2027, 1, 1),
            "Downloaded by DFO 2026-09-11; Next release expected 2026-12 (overdue).",
        ),
        (None, "2026-12-15", date(2027, 1, 1), "Next release expected 2026-12-15 (overdue)."),
        (None, "2026-10-01", date(2026, 10, 12), "Next release expected 2026-10-01 (overdue)."),
        (None, "2026-10-12", date(2026, 10, 12), "Next release expected 2026-10-12."),
        (None, "early 2027", date(2028, 1, 1), "Next release expected early 2027."),
        (None, None, date(2027, 1, 1), None),
    ],
    ids=[
        "expected_this_month",
        "overdue",
        "overdue_date",
        "overdue_day_this_month",
        "due_today",
        "unparsed_month",
        "neither",
    ],
)
async def test_licence_dates_are_labelled_for_what_they_hold(
    monkeypatch: pytest.MonkeyPatch,
    last_update: str | None,
    next_expected_update: str | None,
    today: date,
    line: str | None,
) -> None:
    """`last_update` is DFO's download; the next release is overdue once it is
    before today on the tool's clock, by day for a full date and by month for
    a `YYYY-MM` value."""
    monkeypatch.setattr(sources_tool, "_today", lambda: today)
    licence = {
        **_documentation()["license"],
        "last_update": last_update,
        "next_expected_update": next_expected_update,
    }

    result = await _call_sources(_serving(_documentation(license=licence)), ref=CRS)

    text = text_of(result)
    if line is None:
        assert "Downloaded by DFO" not in text
        assert "Next release expected" not in text
    else:
        assert f"\n{line}" in text


async def test_documentation_names_its_gaps_in_plain_words_instead_of_empty_fields() -> None:
    result = await _call_sources(_serving(_documentation()), ref=CRS)

    text = text_of(result)
    assert (
        "Not published for this dataset: API usage example, Python usage example, "
        "countries covered, data notes, methodology." in text
    )
    assert "Methodology:" not in text
    assert "Data notes:" not in text


async def test_documentation_renders_methodology_and_data_notes_when_published() -> None:
    doc = _documentation(
        methodology={"description": "Amounts are deflated to constant prices."},
        data_notes=["Negative values are repayments.", "Totals exclude core multilateral."],
        gaps=["scope_and_units.countries"],
    )

    result = await _call_sources(_serving(doc), ref=CRS)

    text = text_of(result)
    assert "Methodology:\nAmounts are deflated to constant prices." in text
    assert (
        "Data notes:\n- Negative values are repayments.\n- Totals exclude core multilateral."
        in text
    )
    assert "Not published for this dataset: countries covered." in text


async def test_the_documentation_answer_carries_the_facts_and_gaps() -> None:
    result = await _call_sources(_serving(_documentation()), ref=CRS)

    assert result.structured_content == {
        "ref": CRS,
        "name": "OECD Creditor Reporting System (CRS)",
        "publisher": "Organisation for Economic Co-operation and Development (OECD)",
        "license_type": "OECD Terms and Conditions",
        "years_covered": "1973-2024",
        "gaps": [
            "scope_and_units.countries",
            "methodology.description",
            "data_notes",
            "usage.api_access",
            "usage.python",
        ],
        "document_errors": [],
    }


# Refusals


async def test_a_ref_the_deployment_does_not_hold_refuses_node_not_held() -> None:
    result = await _call_sources(_refusing(404, "node_not_held"), ref="provenance/Nope")

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "node_not_held"
    assert "provenance/Nope" in result.structured_content["message"]


async def test_a_ref_that_is_not_a_source_node_refuses_request_invalid_with_the_doors_detail() -> (
    None
):
    result = await _call_sources(
        _refusing(400, "request_invalid", detail="country/KEN is a Place, not a source"),
        ref="country/KEN",
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert result.structured_content["fix"]["problems"] == [
        "ref: country/KEN is a Place, not a source"
    ]


async def test_a_malformed_ref_is_refused_before_any_read() -> None:
    requests: list[httpx2.Request] = []

    result = await _call_sources(_serving(_documentation(), requests), ref="../etc/passwd")

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert requests == []


async def test_an_unmounted_graph_surface_refuses_capability_unavailable() -> None:
    result = await _call_sources(_refusing(404, "not_found"))

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "capability_unavailable"


async def test_a_member_the_door_could_not_read_is_not_called_unpublished() -> None:
    """The door holds a version history it could not parse: that is a read failure,
    named with the door's reason, not a dataset that publishes no version history."""
    doc = _documentation(
        version_history=[],
        gaps=["version_history", "data_notes"],
        document_errors=[
            {"member": "version_history", "line": 12, "reason": "missing_release_title"}
        ],
    )

    result = await _call_sources(_serving(doc), ref=CRS)

    text = text_of(result)
    assert "Published but could not be read: version history (missing release title)." in text
    assert "Not published for this dataset: data notes." in text
    assert result.structured_content is not None
    assert result.structured_content["document_errors"] == ["version_history"]


async def test_a_catalogue_description_is_cut_at_its_earliest_sentence_end() -> None:
    catalogue = _catalogue()
    catalogue["sources"][0]["datasets"][0]["description"] = "Is it aid? It is. Really."

    result = await _call_sources(_serving(catalogue))

    assert "-- Is it aid?\n" in text_of(result)


async def test_documentation_caps_its_lists_and_counts_the_rest() -> None:
    doc = _documentation(
        methodology={"description": "m" * 7_000},
        data_notes=[f"note {i}" for i in range(12)],
        transformation_log=[
            {"transformation": f"step {i}", "description": f"does {i}."} for i in range(12)
        ],
        version_history=[
            {"date": f"2026-0{i}-01", "title": f"release {i}", "changes": []} for i in range(1, 6)
        ],
        gaps=[],
    )

    result = await _call_sources(_serving(doc), ref=CRS)

    text = text_of(result)
    assert (
        "Methodology:\n" + "m" * 6_000 + "\n[Methodology cut at 6,000 of 7,000 characters.]" in text
    )
    assert "- note 9\n- and 2 more" in text
    assert "- step 9: does 9.\n- and 2 more steps" in text
    assert "- 2026-03-01: release 3 (0 change(s))" in text
    assert "release 4" not in text


async def test_a_request_invalid_with_no_detail_names_the_ref_itself() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        # A problem body whose `detail` is null, not the helper's default prose.
        return httpx2.Response(
            400, content=_problem_body(code="request_invalid", retryable=False, detail=None)
        )

    result = await _call_sources(handler, ref="country/KEN")

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["fix"]["problems"] == [
        "ref: 'country/KEN' is not a dataset the catalogue names"
    ]


def _wide_catalogue(datasets: int) -> dict[str, Any]:
    """One publisher holding `datasets` datasets, each with a long first sentence."""
    catalogue = _catalogue()
    entry = catalogue["sources"][0]
    entry["datasets"] = [
        {
            **entry["datasets"][0],
            "ref": f"provenance/Dataset{i}",
            "label": f"Dataset {i}",
            "description": "word " * 100 + "end. Second sentence.",
        }
        for i in range(datasets)
    ]
    return catalogue


async def test_a_description_sentence_is_cut_at_240_characters() -> None:
    result = await _call_sources(_serving(_wide_catalogue(1)))

    line = next(row for row in text_of(result).splitlines() if row.startswith("- Dataset 0 "))
    sentence = line.split(" -- ", 1)[1]
    assert len(sentence) == 240
    assert sentence.endswith("…")


async def test_a_catalogue_over_the_result_cap_drops_its_descriptions_first() -> None:
    # 150 datasets with descriptions run past the cap; without them they fit.
    result = await _call_sources(_serving(_wide_catalogue(150)))

    text = text_of(result)
    assert len(text) <= RESULT_CAP_CHARS
    assert "Descriptions are left out to fit one result." in text
    assert " -- word" not in text
    assert "- Dataset 149 [provenance/Dataset149]" in text
    assert result.structured_content is not None
    assert result.structured_content["descriptions_left_out"] is True


async def test_a_catalogue_still_over_the_cap_is_cut_with_a_count() -> None:
    result = await _call_sources(_serving(_wide_catalogue(600)))

    text = text_of(result)
    assert len(text) <= RESULT_CAP_CHARS
    assert text.endswith("more lines of datasets left out.]")
    assert "- Dataset 0 [provenance/Dataset0]" in text
    assert "- Dataset 599 " not in text


async def test_a_catalogue_under_the_cap_keeps_its_descriptions() -> None:
    result = await _call_sources(_serving(_catalogue()))

    assert result.structured_content is not None
    assert "descriptions_left_out" not in result.structured_content
