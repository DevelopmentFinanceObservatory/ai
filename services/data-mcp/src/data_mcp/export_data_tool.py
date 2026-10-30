"""The `export_data` tool: a frame as a file the user keeps, behind a public link.

The tool asks dfo-apis to prepare the export (`POST /api/datasets/exports`). dfo-apis
renders the exact frame version as CSV, xlsx or Parquet, alone or in a zip bundle,
copies the file to its pinned tier and answers the file's facts with the link's path,
`/api/datasets/exports/{export_id}`. The tool puts that path on the public site
(`UpstreamClient.public_url`), where a GET answers 302 to a URL signed for five
minutes. The tool never calls the link, so no signed URL reaches its result.

The prepare spends one unit of the caller's daily pin allowance for a new file, and
nothing for a repeat. The caller the website named rides on the call
(`UpstreamClient.caller_headers`).

A fetched frame (`spec_`) exports its long layout, with a `_label` column beside each
code. A derived frame (`ops_`) exports its own columns, which carry no labels. A bare
`spec_id` goes to dfo-apis alone, which exports the newest version it holds and names the
pair it resolved; the result names that pair, and the tool writes its mint record
(`common.materialisation.record_bare_read`), so a later call can export it by name. Any
other bare id refuses `frame_ref_incomplete`.

A `spec_` pair `get_data` fetched with options, such as the type columns, is held under a
pair of its own, which dfo-apis never stored. The mint record the frame store keeps beside
that frame names the dataset version dfo-apis holds (`MintRecord.dataset`), and the tool
exports that version (`_exported_pair`). The result names the pair the caller passed and
the version exported. That version holds the frame's rows only when the mint selected
columns or cut party totals. A `groupby` (a `breakdown` or a `stat_vars` sum) can keep the
base version's pair, whose export holds the unsummed rows, so such a frame refuses before
any call to dfo-apis. So does a pair no record states the rendering of
(`common.materialisation.frame_rendering`), since a file of that pair might hold other rows
than the frame. A failed read of the record refuses `service_unavailable`.

dfo-apis answers `snapshot_not_found`, `artifact_expired` or `spec_not_found` for a frame
it cannot export, and the tool refuses `frame_unavailable`. It never exports
another version in place of the one named.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal, NotRequired, TypedDict

from common.derived_replay import DEFAULT_RETRY_AFTER_S
from common.materialisation import (
    FrameRef,
    FrameRendering,
    RenderingUnknown,
    frame_rendering,
    record_bare_read,
)
from fastmcp.tools import ToolResult
from mcp.types import ResourceLink, TextContent
from pydantic import BaseModel, ConfigDict, Field, ValidationError, with_config

from data_mcp.client import DATASETS_PREFIX
from data_mcp.client_holder import get_client
from data_mcp.deadline import (
    EXTERNAL_CALL_LIMIT_S,
    EXTERNAL_CALL_MARGIN_S,
    CallDeadline,
    current_call_deadline,
)
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import ToolRefusal, from_upstream_error, to_tool_result

logger = logging.getLogger(__name__)

_PREPARE_ROUTE = f"{DATASETS_PREFIX}/exports"

# The prepare must answer before the combined server cancels an outside call
# (`EXTERNAL_CALL_LIMIT_S`). dfo-apis finishes a prepare the tool stopped waiting
# for, so a repeat call then finds the file and renders nothing.
_PREPARE_DEADLINE_S = EXTERNAL_CALL_LIMIT_S - EXTERNAL_CALL_MARGIN_S

_FORMAT_NAMES = {"csv": "CSV", "xlsx": "Excel (xlsx)", "parquet": "Parquet"}

_DERIVED_PREFIX = "ops_"
_FETCHED_PREFIX = "spec_"

# The codes dfo-apis answers for a frame it cannot find or read back.
_FRAME_UNAVAILABLE_CODES = frozenset({"snapshot_not_found", "artifact_expired", "spec_not_found"})


@with_config(ConfigDict(extra="forbid"))
class ExportFrameRef(TypedDict):
    """The frame to export: `spec_id`, and its `content_version`, which a `spec_` id may
    leave out."""

    spec_id: str
    content_version: NotRequired[str]


class ExportLicence(BaseModel):
    """One source of the exported frame and its licence, as dfo-apis names them."""

    source: str
    label: str
    licence: str | None
    url: str | None


class ExportFacts(BaseModel):
    """The facts dfo-apis answers for a prepared export. A repeat answers the same."""

    url_path: str
    file_name: str
    media_type: str
    format: Literal["csv", "xlsx", "parquet"]
    bundle: bool
    bytes: int
    rows: int
    columns: int
    sha256: str
    frame: dict[str, str]
    citation: str
    licences: list[ExportLicence]
    pinned: bool


def _size(n: int) -> str:
    """A byte count as a reader scans it: `812 B`, `5.5 KB`, `3.2 MB`."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _licence_line(licence: ExportLicence) -> str:
    terms = licence.licence or "licence not stated"
    return f"- {licence.label}: {terms}{f' ({licence.url})' if licence.url else ''}"


async def _exported_pair(spec_id: str, content_version: str) -> tuple[str, str] | ToolRefusal:
    """The pair dfo-apis exports for the frame `{spec_id, content_version}`, or the refusal
    for a frame no stored version is known to hold the rows of.

    An `ops_` pair is its own stored version. A `spec_` pair maps to its mint record's
    dataset pair (`MintRecord.dataset`). A summed frame refuses, and so does a frame whose
    rendering cannot be established (`frame_rendering`). A pair that is no frame's goes to
    dfo-apis, which refuses it.
    """
    if not spec_id.startswith(_FETCHED_PREFIX):
        return spec_id, content_version
    try:
        ref = FrameRef(spec_id=spec_id, content_version=content_version)
    except ValidationError:
        return spec_id, content_version
    match await frame_rendering(ref, logger=logger):
        case FrameRendering(record=record) as known:
            if known.summed:
                return ToolRefusal.summed_frame_not_exportable(spec_id=spec_id)
            return record.dataset.spec_id, record.dataset.content_version
        case RenderingUnknown(options=options):
            return ToolRefusal.rendering_not_recorded(
                spec_id=spec_id, content_version=content_version, options=options
            )
        case _:
            return ToolRefusal.service_unavailable(
                service="the frame store",
                retry_after_s=DEFAULT_RETRY_AFTER_S,
                detail=f"the record of {spec_id}, {content_version} could not be read",
            )


def _render_digest(facts: ExportFacts, *, url: str, frame_ref: dict[str, str]) -> str:
    """The digest: file, format, shape, size, hash, link, citation, licences, how long the
    file is kept, the frame and the stored version exported, and which columns the file
    holds."""
    file_format = _FORMAT_NAMES[facts.format]
    if facts.bundle:
        file_format += " in a zip with README.md, datapackage.json and CITATION.cff"
    lines = [
        f"File: {facts.file_name}",
        f"Format: {file_format}",
        f"Rows and columns: {facts.rows:,} rows, {facts.columns:,} columns",
        f"Size: {_size(facts.bytes)}",
        f"SHA-256: {facts.sha256}",
        f"Link: {url}",
        f"Citation: {facts.citation}",
        "Licences:",
        *(_licence_line(licence) for licence in facts.licences),
        "Kept: permanently. The link needs no key and does not expire.",
        f"Frame: {frame_ref['spec_id']} at content_version {frame_ref['content_version']}",
    ]
    if facts.frame != frame_ref:
        lines.append(
            f"Exported version: {facts.frame['spec_id']} at content_version "
            f"{facts.frame['content_version']}, the stored version the frame was fetched from"
        )
    if facts.frame["spec_id"].startswith(_DERIVED_PREFIX):
        lines.append("Columns: the derived frame's own. Code labels come with fetched frames only.")
    else:
        lines.append("Columns: the long layout, with a _label column beside each code.")
    return "\n".join(lines)


def _refusal(exc: UpstreamError, *, frame: ExportFrameRef, export_format: str) -> ToolRefusal:
    """The refusal for a prepare dfo-apis refused. Codes the exports door owns map here.
    Every other code maps as it does for `get_data`."""
    match exc.code:
        case "over_ceiling":
            count = exc.extensions.get("count")
            cap = exc.extensions.get("cap")
            refusal = ToolRefusal.export_over_ceiling(
                export_format=export_format,
                count=count if isinstance(count, int) else None,
                cap=cap if isinstance(cap, int) else None,
            )
        case "allowance_exhausted":
            refusal = ToolRefusal.allowance_exhausted(
                retry_after_s=exc.retry_after_s, detail=exc.detail
            )
        case "provenance_unavailable":
            refusal = ToolRefusal.provenance_unavailable(spec_id=frame["spec_id"])
        case "licence_not_redistributable":
            refusal = ToolRefusal.licence_not_redistributable(detail=exc.detail)
        case "not_found":
            # The exports door never answers `not_found`, so one from its path means
            # the deployment serves no exports route.
            refusal = ToolRefusal.capability_unavailable(
                detail="this deployment of dfo-apis serves no exports route",
                trace_id=exc.trace_id,
                capability_version=exc.capability_version,
            )
        case code if code in _FRAME_UNAVAILABLE_CODES:
            refusal = ToolRefusal.frame_unavailable(spec_id=frame["spec_id"], upstream_code=code)
        case "service_unavailable" if exc.retry_after_s is not None:
            refusal = ToolRefusal.service_unavailable(
                service="dfo-apis exports",
                retry_after_s=exc.retry_after_s,
                detail=exc.detail,
                trace_id=exc.trace_id,
                capability_version=exc.capability_version,
            )
        case _:
            refusal = from_upstream_error(exc, tool="export_data")
    return refusal


async def export_data(
    frame_ref: Annotated[
        ExportFrameRef,
        Field(
            description=(
                "The frame to export, as a get_data or verb result names it. A bare "
                "spec_id exports its newest version."
            )
        ),
    ],
    format: Annotated[
        Literal["csv", "xlsx", "parquet"], Field(description="The file format.")
    ] = "csv",
    bundle: Annotated[
        bool,
        Field(
            description="Wrap the file in a zip with README.md, datapackage.json and CITATION.cff."
        ),
    ] = False,
) -> ToolResult:
    """Export a frame, fetched or derived, as a file and return a public link to it.

    The link needs no key and does not expire. CSV and xlsx hold up to 250,000 rows;
    Parquet has no limit.
    """
    spec_id = frame_ref["spec_id"]
    content_version = frame_ref.get("content_version")
    if not content_version and not spec_id.startswith(_FETCHED_PREFIX):
        return to_tool_result(
            ToolRefusal.frame_ref_incomplete(argument="frame_ref", spec_id=spec_id)
        )
    if content_version:
        exported = await _exported_pair(spec_id, content_version)
        if isinstance(exported, ToolRefusal):
            return to_tool_result(exported)
        frame = {"spec_id": exported[0], "content_version": exported[1]}
    else:
        frame = {"spec_id": spec_id}
    client = get_client()
    body = {"frame": frame, "format": format, "bundle": bundle}
    try:
        response = await client.request(
            "POST",
            _PREPARE_ROUTE,
            route=_PREPARE_ROUTE,
            json=body,
            headers=client.caller_headers(),
            deadline=current_call_deadline() or CallDeadline.after(_PREPARE_DEADLINE_S),
        )
    except UpstreamError as exc:
        return to_tool_result(_refusal(exc, frame=frame_ref, export_format=format))
    try:
        facts = ExportFacts.model_validate_json(response.content)
    except ValidationError as exc:
        return to_tool_result(
            ToolRefusal.upstream_response_unusable(
                detail=f"the export response did not match the shape this service expects: {exc}"
            )
        )
    if not content_version:
        await record_bare_read(
            facts.frame["spec_id"], facts.frame["content_version"], logger=logger
        )
    url = client.public_url(facts.url_path)
    # The caller's pair, or the one dfo-apis resolved a bare spec_id to.
    named = (
        {"spec_id": spec_id, "content_version": content_version} if content_version else facts.frame
    )
    structured = {
        "url": url,
        **facts.model_dump(
            include={
                "file_name",
                "media_type",
                "bytes",
                "rows",
                "columns",
                "sha256",
                "citation",
                "licences",
                "pinned",
            }
        ),
        "frame_ref": named,
        "exported_frame_ref": facts.frame,
    }
    link = ResourceLink(
        type="resource_link",
        uri=url,
        name=facts.file_name,
        mimeType=facts.media_type,
        size=facts.bytes,
    )
    return ToolResult(
        content=[
            TextContent(type="text", text=_render_digest(facts, url=url, frame_ref=named)),
            link,
        ],
        structured_content=structured,
        is_error=False,
    )
