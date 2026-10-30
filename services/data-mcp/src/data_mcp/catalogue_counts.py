"""DFO's counts of what it holds, read from the Search API's catalogue route.

`GET /api/search/catalogue` serves the published search catalogue's counts:
indicators, datasets, publishers, countries, and the donors, recipients and
years of aid data. It answers 503 while no catalogue is published. This
module is data-mcp's one reader of that route. The `sources` tool opens its
catalogue digest with `counts_sentence`, and the harness reads the same
sentence through the `catalogue_counts` tool, which the combined server
outside clients reach leaves out (`harness.combined_mcp.HARNESS_ONLY_TOOLS`).

A failed read gives no count. `COUNTS_UNAVAILABLE` says so in place of the
sentence, so every figure quoted comes from a live catalogue.
"""

from __future__ import annotations

from fastmcp.tools import ToolResult
from pydantic import BaseModel, ConfigDict

from data_mcp.client_holder import get_client
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import ToolRefusal, from_upstream_error, to_tool_result

CATALOGUE_ROUTE = "/api/search/catalogue"

# What a model reads in place of the counts when the catalogue could not be read.
COUNTS_UNAVAILABLE = (
    "DFO's counts of what it holds (indicators, datasets, publishers, countries, aid donors, "
    "recipients and years) are unavailable on this deployment just now. Quote no count."
)


class _YearRange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    first: int
    last: int


class _AidCounts(BaseModel):
    model_config = ConfigDict(extra="ignore")

    donors: int
    recipients: int
    years: _YearRange | None = None


class CatalogueCounts(BaseModel):
    """The global counts of `GET /api/search/catalogue`."""

    model_config = ConfigDict(extra="ignore")

    indicators: int
    datasets: int
    publishers: int
    countries: int
    aid: _AidCounts


class _CatalogueBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    catalogue_version: str
    counts: CatalogueCounts


async def read_catalogue_counts() -> tuple[str, CatalogueCounts]:
    """The live catalogue's version and global counts, in one read.

    The read is not retried: a 503 means no catalogue is published, which a
    retry within the same call does not change, and the callers give no
    count rather than wait.

    Raises:
        UpstreamError: The route refused, or never answered.
        ValueError: The body did not match the route's shape.
    """
    response = await get_client().request(
        "GET", CATALOGUE_ROUTE, route=CATALOGUE_ROUTE, retry=False
    )
    body = _CatalogueBody.model_validate(response.json())
    return body.catalogue_version, body.counts


def counts_sentence(counts: CatalogueCounts) -> str:
    """The counts as the sentences a model quotes."""
    aid = counts.aid
    years = f" from {aid.years.first} to {aid.years.last}" if aid.years is not None else ""
    return (
        f"DFO holds {counts.indicators:,} indicators in {counts.datasets:,} datasets from "
        f"{counts.publishers:,} publishers, with data for {counts.countries:,} countries. "
        f"Aid data covers {aid.donors:,} donors and {aid.recipients:,} recipients{years}."
    )


async def catalogue_counts() -> ToolResult:
    """DFO's counts of what it holds, from the live search catalogue.

    Returns the counts as sentences, with the catalogue version and the
    counts themselves as the structured answer, or the refusal of a read
    that failed.
    """
    try:
        version, counts = await read_catalogue_counts()
    except ValueError as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the search catalogue did not match the shape this service expects: {exc}"
            )
        )
    except UpstreamError as exc:
        return to_tool_result(from_upstream_error(exc, tool="catalogue_counts"))
    return ToolResult(
        content=counts_sentence(counts),
        structured_content={"catalogue_version": version, "counts": counts.model_dump()},
        is_error=False,
    )


__all__ = [
    "CATALOGUE_ROUTE",
    "COUNTS_UNAVAILABLE",
    "CatalogueCounts",
    "catalogue_counts",
    "counts_sentence",
    "read_catalogue_counts",
]
