"""The `sources` tool: what DFO holds, and how one dataset is made.

`search` answers a question that names data. A question about the data
itself -- what DFO holds, who publishes it, how a dataset is built -- names
no indicator to resolve, so it has no spec_id and `search` cannot answer it.
This tool reads the Graph API's two source doors instead and renders each
as a digest a model can quote from without inventing a dataset's name:

- with no `ref`, the source catalogue (`GET /api/graph/sources`): every
  publisher, each with the datasets DFO holds from it, one line per dataset
  with its years, cadence, licence and the first sentence of its
  description;
- with a `ref`, one dataset's documentation (`GET /api/graph/documentation/
  {dcid}`): what it is, who publishes it, its licence and cadence, its
  coverage, its methodology and data notes when the publisher wrote any,
  how DFO transforms it, and, in plain words, what the door holds but
  could not read and what is not published for it.

Upstream access goes through the shared `UpstreamClient`; this module reads
the two doors' wire shapes and nothing else of the Graph API.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.tools import ToolResult
from pydantic import BaseModel, ConfigDict, Field

from data_mcp.client_holder import get_client
from data_mcp.describe_tool import DCID_PATTERN
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import ToolRefusal, from_upstream_error, to_tool_result

_CATALOGUE_ROUTE = "/api/graph/sources"
_DOCUMENTATION_ROUTE = "/api/graph/documentation/{dcid}"

# The Graph API's own ceiling on a dcid in a path (`graph_api.routing.
# DCID_MAX_LENGTH`); a longer ref answers 400 before any read.
_REF_MAX_CHARS = 512

# A dataset's description is cut to its first sentence in the catalogue, and
# that sentence to this many characters, so the whole catalogue stays one tool result.
_CATALOGUE_DESCRIPTION_CAP = 240
# The documentation digest's own caps: a methodology past this is cut with a
# line saying so; data notes, transformation steps and releases past these
# counts collapse to a count.
_METHODOLOGY_CAP = 6_000
_DATA_NOTE_CAP = 10
_TRANSFORMATION_CAP = 10
_RELEASE_CAP = 3
# One result, catalogue or documentation, stays under this many characters,
# the same bound `get_skill` puts on a skill body. A catalogue over it drops
# its description sentences first; anything still over is cut at a line break.
RESULT_CAP_CHARS = 32_000


class _Ref(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ref: str
    label: str


class _DataCommons(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    label: str


class _Dataset(BaseModel):
    """One Dataset group of the catalogue, with the card facts the catalogue job computed."""

    model_config = ConfigDict(extra="ignore")

    ref: str
    label: str
    description: str | None = None
    license_type: str | None = None
    update_cadence: str | None = None
    years_covered: str | None = None
    source_url: str | None = None


class _Exclusion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ref: str
    reason: str


class _CatalogueEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: _Ref
    data_commons: _DataCommons
    datasets: list[_Dataset] = []
    exclusions: list[_Exclusion] = []


class _Catalogue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sources: list[_CatalogueEntry]


class _Details(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name_of_source: str | None = None
    publisher: str | None = None
    data_commons: _DataCommons | None = None
    description: str | None = None
    source_url: str | None = None
    source_data_url: str | None = None
    citation: str | None = None


class _License(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    url: str | None = None
    update_cadence: str | None = None
    last_update: str | None = None
    next_expected_update: str | None = None


class _ScopeAndUnits(BaseModel):
    model_config = ConfigDict(extra="ignore")

    countries: list[str] = []
    years_covered: str | None = None
    granularity: str | None = None
    currency: str | None = None


class _Methodology(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str | None = None


class _Transformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transformation: str
    description: str


class _Change(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str
    note: str


class _Release(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    title: str
    changes: list[_Change] = []


class _DocumentError(BaseModel):
    """A member of the stored document the door holds but could not read."""

    model_config = ConfigDict(extra="ignore")

    member: str
    line: int | None = None
    reason: str


class _Documentation(BaseModel):
    """The fields of the documentation door's `SourceDocumentation` this tool reads."""

    model_config = ConfigDict(extra="ignore")

    details: _Details
    license: _License
    scope_and_units: _ScopeAndUnits = _ScopeAndUnits()
    methodology: _Methodology = _Methodology()
    data_notes: list[str] = []
    transformation_log: list[_Transformation] = []
    version_history: list[_Release] = []
    gaps: list[str] = []
    document_errors: list[_DocumentError] = []


def _ref_refusal(ref: str) -> ToolResult | None:
    """A refusal for a ref the documentation door could never serve, or `None`."""
    if not DCID_PATTERN.fullmatch(ref) or any(segment in (".", "..") for segment in ref.split("/")):
        return to_tool_result(
            ToolRefusal.request_invalid(
                pointer="/ref",
                detail=(
                    f"{ref!r} is not a well-formed dataset ref; expected one or more "
                    "'/'-separated segments of letters, digits, '_', '.', or '-', as the "
                    "catalogue names them"
                ),
            )
        )
    return None


def _refusal_from_upstream(exc: UpstreamError, ref: str | None) -> ToolResult:
    """Map a failed Graph source door read onto the refusal grammar.

    `node_not_held` is the door's answer for a dataset this deployment does
    not hold; `request_invalid` for a node that is not a source. A bare
    `not_found` comes from the platform's router, which answers it for every
    path under a surface this deployment does not mount.
    """
    if exc.code == "node_not_held" and ref is not None:
        refusal = ToolRefusal.node_not_held(
            dcids=[ref], trace_id=exc.trace_id, capability_version=exc.capability_version
        )
    elif exc.code == "request_invalid" and ref is not None:
        refusal = ToolRefusal.request_invalid(
            pointer="/ref",
            detail=exc.detail or f"{ref!r} is not a dataset the catalogue names",
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        )
    elif exc.code == "not_found":
        refusal = ToolRefusal.capability_unavailable(
            detail="this deployment does not serve the Graph API's source doors",
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        )
    else:
        refusal = from_upstream_error(exc)
    return to_tool_result(refusal)


def _first_sentence(text: str, cap: int) -> str:
    """`text` up to its first sentence end, then cut to `cap` characters."""
    flat = " ".join(text.split())
    ends = [at for at in (flat.find(end) for end in (". ", "! ", "? ")) if at != -1]
    if ends:
        flat = flat[: min(ends) + 1]
    if len(flat) <= cap:
        return flat
    return flat[: cap - 1].rstrip() + "…"


def _facts(*values: str | None) -> str:
    """The stated facts of `values`, joined; `""` when none is stated."""
    stated = [value for value in values if value]
    return "; ".join(stated)


def _render_catalogue(catalogue: _Catalogue) -> tuple[str, dict[str, Any]]:
    """The catalogue as a digest grouped by publisher, and its small answer,
    under `RESULT_CAP_CHARS`: without description sentences when the full
    digest is over it, and cut at a line break when that is still over."""
    content, answer = _catalogue_digest(catalogue, with_descriptions=True)
    if len(content) > RESULT_CAP_CHARS:
        content, answer = _catalogue_digest(catalogue, with_descriptions=False)
        answer["descriptions_left_out"] = True
    return _bounded(content, what="datasets"), answer


def _bounded(content: str, *, what: str) -> str:
    """`content` cut at its last line break under `RESULT_CAP_CHARS`, with a
    closing line counting the lines left out; `content` itself when it fits."""
    if len(content) <= RESULT_CAP_CHARS:
        return content
    cut = content.rfind("\n", 0, RESULT_CAP_CHARS - 200)
    kept = content[:cut]
    left = content[cut:].count("\n")
    return f"{kept}\n[Cut to fit one result: {left} more lines of {what} left out.]"


def _catalogue_digest(
    catalogue: _Catalogue, *, with_descriptions: bool
) -> tuple[str, dict[str, Any]]:
    """The catalogue digest, with or without each dataset's first description sentence."""
    dataset_count = sum(len(entry.datasets) for entry in catalogue.sources)
    buckets: dict[str, int] = {}
    for entry in catalogue.sources:
        buckets[entry.data_commons.label] = buckets.get(entry.data_commons.label, 0) + 1
    bucket_line = ", ".join(f"{count} under {label}" for label, count in buckets.items())
    lines = [
        f"DFO holds {dataset_count} datasets from {len(catalogue.sources)} publishers "
        f"({bucket_line}). One line per dataset: label [ref] (years; cadence; licence)"
        + (
            " -- the first sentence of its description."
            if with_descriptions
            else ". Descriptions are left out to fit one result."
        )
        + " Pass a dataset's ref to this tool for its full documentation."
    ]
    for entry in catalogue.sources:
        lines.append("")
        lines.append(f"## {entry.source.label} [{entry.data_commons.label}]")
        for dataset in entry.datasets:
            facts = _facts(dataset.years_covered, dataset.update_cadence, dataset.license_type)
            head = f"- {dataset.label} [{dataset.ref}]"
            if facts:
                head += f" ({facts})"
            if with_descriptions and dataset.description:
                head += f" -- {_first_sentence(dataset.description, _CATALOGUE_DESCRIPTION_CAP)}"
            lines.append(head)
        superseded = [x for x in entry.exclusions if x.reason == "superseded_by_dfo"]
        if superseded:
            lines.append(
                f"- {len(superseded)} dataset(s) from this publisher are served through DFO's "
                "own version instead."
            )
    answer = {
        "datasets": dataset_count,
        "publishers": [entry.source.label for entry in catalogue.sources],
    }
    return "\n".join(lines), answer


def _plain_gap(path: str) -> str:
    """A `gaps` entry, a dotted wire path, in plain words: `scope_and_units.countries` ->
    `countries covered`; the rest read as their last segment."""
    named = {
        "scope_and_units.countries": "countries covered",
        "scope_and_units.years_covered": "years covered",
        "scope_and_units.granularity": "granularity",
        "scope_and_units.currency": "currency",
        "methodology.description": "methodology",
        "data_notes": "data notes",
        "usage.api_access": "API usage example",
        "usage.python": "Python usage example",
        "license.type": "licence",
        "license.update_cadence": "update cadence",
        "details.citation": "citation",
        "transformation_log": "transformation log",
        "version_history": "version history",
    }
    return named.get(path, path.rsplit(".", 1)[-1].replace("_", " "))


def _render_documentation(ref: str, doc: _Documentation) -> tuple[str, dict[str, Any]]:
    """One dataset's documentation as a digest, and its small answer."""
    details, licence, scope = doc.details, doc.license, doc.scope_and_units
    title = details.name_of_source or ref
    lines = [f"# {title} [{ref}]"]
    who = _facts(
        f"Publisher: {details.publisher}" if details.publisher else None,
        f"part of {details.data_commons.label}" if details.data_commons else None,
    )
    if who:
        lines.append(who + ".")
    if details.description:
        lines.append(details.description.strip())
    licence_facts = _facts(
        licence.type,
        f"updated {licence.update_cadence.lower()}" if licence.update_cadence else None,
        f"last update {licence.last_update}" if licence.last_update else None,
        f"next expected {licence.next_expected_update}" if licence.next_expected_update else None,
    )
    if licence_facts:
        line = f"Licence: {licence_facts}."
        if licence.url:
            line += f" Terms: {licence.url}"
        lines.append(line)
    coverage = _facts(scope.years_covered, scope.granularity, scope.currency)
    if scope.countries:
        coverage = _facts(coverage, f"{len(scope.countries)} countries")
    if coverage:
        lines.append(f"Coverage: {coverage}.")
    if details.citation:
        lines.append(f"Citation: {details.citation}")
    methodology = (doc.methodology.description or "").strip()
    if methodology:
        if len(methodology) > _METHODOLOGY_CAP:
            cut = methodology[:_METHODOLOGY_CAP].rstrip()
            methodology = (
                f"{cut}\n[Methodology cut at {len(cut):,} of {len(methodology):,} characters.]"
            )
        lines.append(f"Methodology:\n{methodology}")
    if doc.data_notes:
        shown = doc.data_notes[:_DATA_NOTE_CAP]
        notes = "\n".join(f"- {note.strip()}" for note in shown)
        remaining = len(doc.data_notes) - len(shown)
        tail = f"\n- and {remaining} more" if remaining > 0 else ""
        lines.append(f"Data notes:\n{notes}{tail}")
    if doc.transformation_log:
        shown_steps = doc.transformation_log[:_TRANSFORMATION_CAP]
        steps = "\n".join(
            f"- {step.transformation}: {_first_sentence(step.description, _CATALOGUE_DESCRIPTION_CAP)}"
            for step in shown_steps
        )
        remaining = len(doc.transformation_log) - len(shown_steps)
        tail = f"\n- and {remaining} more steps" if remaining > 0 else ""
        lines.append(f"How DFO transforms it:\n{steps}{tail}")
    if doc.version_history:
        shown_releases = doc.version_history[:_RELEASE_CAP]
        releases = "\n".join(
            f"- {release.date}: {release.title} ({len(release.changes)} change(s))"
            for release in shown_releases
        )
        lines.append(f"Releases, newest first:\n{releases}")
    links = _facts(
        f"Source page: {details.source_url}" if details.source_url else None,
        f"Data download: {details.source_data_url}" if details.source_data_url else None,
    )
    if links:
        lines.append(links + ".")
    unreadable = {error.member: error.reason for error in doc.document_errors}
    if unreadable:
        # The door holds these members but could not read them; saying they
        # are not published would be false, so they get their own line.
        read_failures = ", ".join(
            f"{_plain_gap(member)} ({reason.replace('_', ' ')})"
            for member, reason in sorted(unreadable.items())
        )
        lines.append(f"Published but could not be read: {read_failures}.")
    absent = sorted({_plain_gap(path) for path in doc.gaps if path not in unreadable})
    if absent:
        lines.append(f"Not published for this dataset: {', '.join(absent)}.")
    answer = {
        "ref": ref,
        "name": details.name_of_source,
        "publisher": details.publisher,
        "license_type": licence.type,
        "years_covered": scope.years_covered,
        "gaps": list(doc.gaps),
        "document_errors": [error.member for error in doc.document_errors],
    }
    return "\n\n".join(lines), answer


async def sources(
    ref: Annotated[
        str | None,
        Field(
            default=None,
            min_length=1,
            max_length=_REF_MAX_CHARS,
            description=(
                "A dataset's ref as the catalogue names it, for that dataset's documentation; "
                "leave unset for the catalogue of everything DFO holds."
            ),
        ),
    ] = None,
) -> ToolResult:
    """What data DFO holds, and how one of its datasets is made.

    Call this with no `ref` to learn what DFO holds: every publisher and, under
    each, the datasets DFO serves from it, one line per dataset with its years,
    update cadence, licence and the first sentence of its description. Use it
    for a question about the data itself rather than about a measure -- what
    datasets exist, who publishes them, how far back they go -- which `search`
    cannot answer because it names no indicator to resolve. Never invent a
    dataset the catalogue does not list.

    Call it with a `ref` from the catalogue for that dataset's documentation:
    what it is, who publishes it, its licence and cadence, its coverage, its
    methodology and data notes when the publisher wrote any, how DFO transforms
    it, and what is not published for it, named plainly. A ref the catalogue
    does not name refuses `node_not_held`.

    Args:
        ref: A dataset's ref from the catalogue, or unset for the catalogue.

    Returns:
        The catalogue, or one dataset's documentation, as a digest.
    """
    if ref is None:
        return await _catalogue_result()
    refusal = _ref_refusal(ref)
    return refusal if refusal is not None else await _documentation_result(ref)


async def _catalogue_result() -> ToolResult:
    """The catalogue digest, or the refusal for a read that failed."""
    try:
        response = await get_client().request("GET", _CATALOGUE_ROUTE, route=_CATALOGUE_ROUTE)
    except UpstreamError as exc:
        return _refusal_from_upstream(exc, None)
    try:
        catalogue = _Catalogue.model_validate(response.json())
    except ValueError as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the source catalogue did not match the shape this service expects: {exc}"
            )
        )
    content, answer = _render_catalogue(catalogue)
    return ToolResult(content=content, structured_content=answer, is_error=False)


async def _documentation_result(ref: str) -> ToolResult:
    """One dataset's documentation digest, or the refusal for a read that failed."""
    try:
        response = await get_client().request(
            "GET", f"/api/graph/documentation/{ref}", route=_DOCUMENTATION_ROUTE
        )
    except UpstreamError as exc:
        return _refusal_from_upstream(exc, ref)
    try:
        doc = _Documentation.model_validate(response.json())
    except ValueError as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the source documentation did not match the shape this service expects: {exc}"
            )
        )
    content, answer = _render_documentation(ref, doc)
    content = _bounded(content, what="documentation")
    return ToolResult(content=content, structured_content=answer, is_error=False)


__all__ = ["sources"]
