"""A composite frame's components, as a `get_data` manifest's trace lists them, and the
years they serve.

dfo-apis mints a composite definition, such as bilateral plus imputed multilateral aid,
as one frame whose rows each belong to one component, and lists the components under
the manifest's `trace.components` (apis' `materialise.composite.ComponentSummary`).
data-mcp's `get_data` digest, the harness's served scope and analysis-mcp's refusals
of a composite's StatVars all read them through `manifest_components`, and state the
years a component serves through `year_spans` and `years_of`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# The trace member apis lists a composite frame's components under, absent on a frame of
# one table.
COMPONENTS_KEY = "components"


class ManifestComponent(BaseModel):
    """One component of a composite frame: one entry of the manifest's `trace.components`.

    A composite definition binds a declared composite provenance, which no graph node
    holds, and apis mints it from the tables its components name: stacked components
    (bilateral plus imputed multilateral aid) share years and each row is one
    component's, and spliced ones (net flows to 2017, grant equivalents from 2018) serve
    disjoint years. `role` names what the component holds, `provenance` the table it was
    read from, `periods` the years it serves as closed spans, `dropped_years` the years
    left out of it, outside the years the composite serves, `stat_vars` its members,
    `rows` its rows in the frame and `zero_rows` how many of them are zeros written where
    a table that reads absence as zero holds nothing. A field apis adds later is dropped,
    so a reader keeps working across an upstream schema change.
    """

    model_config = ConfigDict(extra="ignore")
    role: str
    provenance: str | None = None
    content_version: str | None = None
    periods: list[tuple[int, int]] = Field(default_factory=list)
    dropped_years: list[int] = Field(default_factory=list)
    stat_vars: list[str] = Field(default_factory=list)
    rows: int = 0
    zero_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


_COMPONENTS = TypeAdapter(list[ManifestComponent])


def manifest_components(trace: Mapping[str, object]) -> list[ManifestComponent]:
    """The components a manifest's `trace` says its composite frame was minted from, in
    apis' order, and none for a frame of one table.

    Raises:
        pydantic.ValidationError: When the trace lists them in another shape.
    """
    return _COMPONENTS.validate_python(trace.get(COMPONENTS_KEY) or [])


def years_of(periods: Iterable[tuple[int, int]]) -> set[int]:
    """Every year the closed spans `periods` hold."""
    return {year for first, last in periods for year in range(first, last + 1)}


def year_spans(years: Iterable[int]) -> list[tuple[int, int]]:
    """`years`, sorted and distinct, as closed runs of consecutive years."""
    spans: list[tuple[int, int]] = []
    for year in sorted(set(years)):
        if spans and year == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], year)
        else:
            spans.append((year, year))
    return spans
