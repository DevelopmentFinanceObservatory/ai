"""Renders `get_data`'s mint manifest into the model-facing digest.

On a successful call, this digest is the entire model-facing surface. The
harness reads `structuredContent` in code, so a model only ever sees this
digest. `harness.mcp_client.McpExecutor._get_data` takes the mint manifest
for the provenance store and the frame reference, then answers the graph
with `_manifest_summary`'s one line, and `executor.run` records this digest
text as its step's `StepRecord.digest` for `narrate` to read. So whatever
this module doesn't render into the digest text, a successful `get_data`
call never puts in front of the model at all.

A refusal is the other half of that boundary. The actionable recovery
grammar rides `structuredContent`, and `McpExecutor` merges it into the
graph's typed refusal (`code`, `retry_mode`, `fix`), and that is what
reaches the model.

Scope for this module: the acknowledgement line, the coverage line, the
components of a composite frame, any warnings that qualify that coverage, the period note, the frame reference, the citation
line, the version description's steps, caveats and unknowns, per-series statistics, any dropped-role,
remint-persistence, operation-poll, or degraded-frame notes, and any
remaining manifest warnings.

Deliberately independent of `data_mcp.digest`: that module parses the Search
API's resolve envelope (`resolved.coverage["kind"]`/`["has_data"]`, `.get()`
on a facet value that is always a mapping there), shapes a manifest doesn't
have. `strip_manifest`'s `CoverageModel` carries `{rows, dimensions, window}`
with neither `kind` nor `has_data`, and a manifest's facet values are plain
strings, so reusing `data_mcp.digest`'s renderers here would either silently
misrender ("no data" on every frame) or raise (`AttributeError` on a bare
string). `render_get_data_digest` is the public seam.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

from common.materialisation import FrameRef
from common.period_note import period_note
from common.place_type import WHOLE_SIDE_POPULATIONS
from common.price_basis import price_basis_reading

from data_mcp.frame_materialisation import MaterialisationOutcome
from data_mcp.manifest import (
    CitationModel,
    CoverageModel,
    DefinitionModel,
    Manifest,
    ManifestComponent,
    PartyTotalsLeftOut,
    SeriesStatisticsModel,
    WarningModel,
    manifest_components,
    party_totals_left_out,
)
from data_mcp.series_statistics import continued_series, continuing_method
from data_mcp.tool_errors import ENTITY_DCID_MARKER
from data_mcp.version_description import VersionDescription

# How the digest's frame reference line and operation poll note begin. Both
# address a caller handling frame references by hand; `harness.graph` drops
# lines starting with either from a plan step's summary.
FRAME_REFERENCE_PREFIX = "Frame: spec_id="
OPERATION_POLL_PREFIX = "Note: this mint exceeded the synchronous budget"

# The digest carries statistics for up to this many series; past it, only the
# series count is reported. Matches the platform contract's own ceiling: a
# wide frame (dozens of series across several units) has no single
# meaningful summary, and a model holding forty sets of figures is likelier
# to quote the wrong one than to reason across them.
_STATISTICS_SERIES_CAP = 8

# At most this many manifest warnings are rendered individually; the rest
# are only counted. A frame can carry more warnings than are worth reading
# one by one, and a run-on trailer is as unreadable as no trailer at all.
# This is one shared budget across coverage and non-coverage warnings together,
# not a per-group cap: coverage warnings draw from it first, and whatever is
# left over goes to the rest.
_MANIFEST_WARNING_CAP = 5

# At most this many entity dcids are named per role in the acknowledgement
# line; the rest are only counted. Resolving a region can expand to dozens
# of member dcids (e.g. "Africa" resolves to roughly 54 countries), and
# naming all of them would let one role's breadth dominate the sentence.
_ACKNOWLEDGEMENT_ENTITY_CAP = 8

# At most this many group totals are named per role in the line that says which apis
# left out of the frame; the rest are only counted.
_LEFT_OUT_TOTALS_CAP = 10

# At most this many source labels are named in the citation line; the rest
# are only counted. A frame merged across several providers carries one
# source per provider, and naming all of them would let the citation line
# grow with the frame instead of summarising it.
_CITATION_SOURCE_CAP = 5

# At most this many of a version description's steps, caveats and unknown members are
# rendered; the rest are only counted. A description can carry 50 caveats and 100
# gaps, and the digest names the leading few.
_DESCRIPTION_STEP_CAP = 5
_DESCRIPTION_CAVEAT_CAP = 5
_DESCRIPTION_GAP_CAP = 6

# The SDG/OECD price-basis unit-code family: current prices (`CU`) or
# constant prices carrying an optional base year (`CON`, `CON_1999`),
# a three-letter currency, and an optional magnitude suffix.
_SDG_PRICE_BASIS_RE = re.compile(
    r"^SDG_(CU|CON(?:_(\d{4}))?)_(?P<currency>[A-Z]{3})(?:_(?P<magnitude>K|M|B))?$"
)

# This deployment's own current-price unit refs, read from
# `search.engine.unit.refs._UNIT_REF_TABLE`.
_DEPLOYMENT_CURRENT_REFS = frozenset({"USD", "EUR", "GBP"})

# This deployment's own constant-price unit refs, same source: a currency
# and the base year it is constant against.
_DEPLOYMENT_CONSTANT_RE = re.compile(r"^Constant(?P<currency>[A-Z]{3})_(?P<year>\d{4})$")

_MAGNITUDE_WORDS = {"K": "thousands", "M": "millions", "B": "billions"}

# The manifest warning apis added, on versions stored before descriptions, when a
# summed mint added more than one StatVar into the series it labels `dcid:<spec_id>`.
# Its message carries the count of StatVars summed, so it renders outside
# `_MANIFEST_WARNING_CAP`.
_STAT_VARS_SUMMED = "stat_vars_summed"

# What the digest calls each component role apis composes a frame from.
_COMPONENT_NOUNS = {
    "bilateral": "bilateral",
    "imputed": "imputed multilateral",
    "headline_net": "net flows",
    "headline_ge": "grant equivalents",
}

# The coverage dimension apis counts after every declared role column.
_YEARS_DIMENSION = "years"

# A reader's noun for each flow axis a role column can sit on; a role with no
# axis (`observationAbout`) reads "entity".
_GRAIN_NOUNS = {"origin": "provider", "destination": "recipient"}


@dataclass(frozen=True, kw_only=True)
class OperationPollNote:
    """This call's own mint crossed apis' sync budget and ran as a queued
    operation, polled to completion before this call returned. Built by
    `get_data_tool.py` from `data_mcp.operations.ResolvedOperation`, whose own
    docstring explains why the digest reports this rather than leaving it to be
    inferred from how long the call took.
    """

    operation_id: str
    poll_attempts: int
    elapsed_s: float


@dataclass(frozen=True, kw_only=True)
class BreakdownLevel:
    """The code system and hierarchy level a `get_data(breakdown=...)` mint summed the
    family within. Built by `get_data_tool.py` from its `Breakdown` argument."""

    code_system: str
    level: int


@dataclass(frozen=True, kw_only=True)
class SummedFamily:
    """The family a summed mint added into one series per entity.

    `ref` is the `stat_var` every summed series carries, `dcid:<spec_id>`. `label` is
    the family's name, the version description's title, `None` when the description
    was not read or its title is a gap. `count` is how many StatVars with data were
    summed, `None` when the trace records no member's rows.
    """

    ref: str
    label: str | None
    count: int | None


def summed_family(
    manifest: Manifest,
    *,
    frame_summed: bool | None,
    series_stats: Sequence[SeriesStatisticsModel] | None,
    description: VersionDescription | None,
) -> SummedFamily | None:
    """The family `manifest`'s version summed, or `None` when it summed nothing.

    apis gives a summed series the family ref `dcid:<spec_id>` as its `stat_var`,
    and no other series carries it, so a frame holding that ref summed its family.
    `frame_summed` says whether the frame this call read holds it
    (`MaterialisationOutcome.family_summed`); when this call read no frame, the
    fetched statistics rows say instead. The count is the trace's members with rows,
    the count apis' `sum_variables` step states. Only the name comes from
    `description`, its title unless the title is a gap.
    """
    ref = f"dcid:{manifest.definition.spec_id}"
    if frame_summed is None:
        frame_summed = any(series.stat_var == ref for series in series_stats or ())
    if not frame_summed:
        return None
    label = None
    if description is not None and "title" not in description.gaps:
        label = description.title or None
    return SummedFamily(ref=ref, label=label, count=_members_with_rows(manifest.trace) or None)


def _members_with_rows(trace: Mapping[str, object]) -> int:
    """How many of the trace's `resolved_stat_vars` hold rows, by their
    `observation_count`."""
    entries = trace.get("resolved_stat_vars")
    if not isinstance(entries, list):
        return 0
    return sum(
        1
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(rows := entry.get("observation_count"), int)
        and rows > 0
    )


def statistics_needs_fetch(series_count: int | None) -> bool:
    """True when `render_get_data_digest` will use fetched per-series rows for `series_count`.

    `get_data_tool.py` calls this before touching the network, so a call
    whose count is `None` (an old record), `0`, or past
    `_STATISTICS_SERIES_CAP` never fetches the statistics artifact at all --
    the digest renders a count-only line, or nothing, from `series_count`
    alone in every one of those cases.
    """
    return series_count is not None and 0 < series_count <= _STATISTICS_SERIES_CAP


def render_get_data_digest(
    *,
    manifest: Manifest,
    frame_ref: FrameRef,
    entity_labels: Mapping[str, str],
    dropped_roles: Sequence[str],
    remint_persist_failed: bool = False,
    degraded_frame: MaterialisationOutcome | None = None,
    series_stats: Sequence[SeriesStatisticsModel] | None = None,
    statistics_fetch_failed: bool = False,
    operation_poll: OperationPollNote | None = None,
    breakdown: BreakdownLevel | None = None,
    description: VersionDescription | None = None,
    summed: SummedFamily | None = None,
) -> str:
    """Render `manifest` into the model-facing digest for `get_data`.

    Args:
        manifest: The stripped mint-route manifest this call retrieved.
        frame_ref: The reference the frame is held under
            (`frame_materialisation.frame_ref_for`).
        entity_labels: dcid -> name, for every entity this call's own
            resolution named. A dcid outside this map (e.g. one carried over
            unchanged from the base definition, which this call never
            resolved) renders as itself.
        dropped_roles: Roles the base definition bound that an incoming,
            non-empty `entities` override silently left unbound.
        remint_persist_failed: This call re-minted under a new `spec_id` and
            could not confirm the definition store write -- the store was
            unreachable, or the write outran the deadline and may still land.
            The frame reference below is genuine and usable right now, but a
            later `get_data` call passing that same `spec_id` may not find it
            registered.
        degraded_frame: The `"degraded"` outcome of this call's
            `data_mcp.frame_materialisation.materialise_and_register`: the mint
            succeeded and the frame reference below is real, but the frame is not
            confirmed held. Its `degraded_reason` picks the note's wording, and a
            `"fetch"` or `"store_unchecked"` note reports its `attempts`, which may
            be `0` (a sign failure or a timeout before the first GET completed).
            `None` (the default) renders no note.
        series_stats: The rows `data_mcp.series_statistics.fetch_series_statistics`
            read off `manifest`'s own statistics artifact, already fetched
            by the caller -- this function does no I/O of its own. Only
            meaningful, and only ever read, when
            `statistics_needs_fetch(manifest.series_count)` was true;
            ignored otherwise, since a count of `0`, past the cap, or `None`
            renders from `manifest.series_count` alone.
        statistics_fetch_failed: The caller tried the fetch
            `statistics_needs_fetch` called for and it raised -- an artifact
            fetch or a Parquet decode failure. Renders a count-only line with
            a note rather than raising: the mint itself succeeded and the
            frame reference above is real, so a sidecar-artifact hiccup
            degrades this section alone, exactly like a degraded frame fetch
            degrades the frame reference's own section.
        operation_poll: Set when this call's own mint crossed apis' sync
            budget and was queued as an operation, then polled to completion
            here before returning. `None` (the default) renders no note,
            which is every call's outcome except this one -- an ordinary
            synchronous mint never touches `data_mcp.operations` at all.
        breakdown: Set when this call summed the family within each code of one
            code system at one hierarchy level. Renders a line after the
            coverage that names the frame's `stat_var` and
            `stat_var_<code_system>` columns.
        description: apis' description of the manifest's version
            (`data_mcp.version_description.fetch_version_description`), the one
            the version's cards carry. Its citation, which names the sources, and its
            latest period replace the manifest's citation line, and its steps, caveats and gaps
            follow as one line each. `None` (the read failed, or the default)
            renders the manifest's citation line alone.
        summed: The family the frame summed (`summed_family`), `None` (the default) for
            a frame that summed none.

    Returns:
        A short prose digest: an acknowledgement of what was retrieved, its
        coverage, the components of a composite frame (`trace.components`), the group totals apis left out of the frame
        (`trace.party_totals_left_out`), any warnings that qualify that coverage, the period note
        of a one-year frame (`common.period_note.period_note`), the frame
        reference for downstream tools, then the citation, per-series statistics, any
        dropped-role, remint-persistence, or operation-poll
        notes, any degraded-fetch note, the `stat_vars_summed` warning, and
        any remaining manifest warnings. On a summed version the acknowledgement
        names the family and the grain it was summed to, and each summed series is
        headed as the family's total.
    """
    definition = manifest.definition
    family = None if summed is None else _family_name(summed, manifest.citation)
    # A stored version's `stat_vars_summed` warning only renders its message here;
    # whether the frame summed comes from `summed`.
    summed_warnings = [w for w in manifest.warnings if w.code == _STAT_VARS_SUMMED]
    coverage_warnings, other_warnings = _split_coverage_warnings(
        [w for w in manifest.warnings if w.code != _STAT_VARS_SUMMED]
    )
    shown_coverage, shown_other, warning_remainder = _budget_warnings(
        coverage_warnings, other_warnings
    )
    sections = [
        _acknowledgement_line(
            definition, manifest.coverage, entity_labels, family=family, breakdown=breakdown
        ),
        _coverage_line(manifest.coverage, definition),
    ]
    sections.extend(_components_lines(manifest_components(manifest.trace)))
    if breakdown is not None:
        sections.append(_breakdown_line(breakdown, spec_id=definition.spec_id))
    sections.extend(_left_out_totals_lines(party_totals_left_out(manifest.trace), entity_labels))
    sections.extend(_warning_lines(shown_coverage))
    if note := period_note(manifest.coverage.model_dump(mode="json")):
        sections.append(note)
    sections.append(_frame_reference_line(frame_ref))
    if description is not None:
        sections.extend(_description_lines(description))
    elif manifest.citation.text or manifest.citation.sources:
        sections.append(_citation_line(manifest.citation))
    sections.extend(
        _statistics_sections(
            manifest,
            series_stats,
            fetch_failed=statistics_fetch_failed,
            summed_ref=None if summed is None else summed.ref,
            family=family,
            breakdown=breakdown,
        )
    )
    sections.extend(_dropped_role_line(role) for role in dropped_roles)
    if remint_persist_failed:
        sections.append(_remint_persist_failed_line())
    if operation_poll is not None:
        sections.append(_operation_poll_line(operation_poll))
    if degraded_frame is not None:
        sections.append(_degraded_frame_note(degraded_frame))
    sections.extend(_warning_lines(summed_warnings))
    sections.extend(_warning_lines(shown_other))
    if warning_remainder > 0:
        sections.append(f"{warning_remainder} more warning(s) not shown.")
    return "\n\n".join(sections)


def _family_name(summed: SummedFamily, citation: CitationModel) -> str:
    """What the digest calls a summed family: its label, else the family of the sources
    the manifest cites, else the requested StatVar family."""
    if summed.label:
        return summed.label
    sources = list(dict.fromkeys(source.label for source in citation.sources if source.label))
    if sources:
        return f"the {_joined_with_cap(sources, _ACKNOWLEDGEMENT_ENTITY_CAP)} family"
    return "the requested StatVar family"


def _breakdown_line(breakdown: BreakdownLevel, *, spec_id: str) -> str:
    """How a breakdown frame's columns read: one series per code at the level, entity
    and year, every row's `stat_var` the family ref."""
    column = f"stat_var_{breakdown.code_system}"
    return (
        f"Summed within each {breakdown.code_system} code at level {breakdown.level}: "
        f"{column} names the code each row's StatVars roll up to, and stat_var is "
        f"dcid:{spec_id} on every row."
    )


def _split_coverage_warnings(
    warnings: Sequence[WarningModel],
) -> tuple[list[WarningModel], list[WarningModel]]:
    """`warnings` split into ones that qualify the served coverage and the rest.

    A coverage-scope warning's code starts with `coverage_` (for example
    `coverage_below_scope`, raised when the served window falls short of the
    requested one) and belongs beside the coverage line it qualifies, ahead
    of statistics computed only over the served rows. Every other warning
    keeps its place at the end of the digest.
    """
    coverage_warnings = [w for w in warnings if w.code.startswith("coverage_")]
    other_warnings = [w for w in warnings if not w.code.startswith("coverage_")]
    return coverage_warnings, other_warnings


def _acknowledgement_line(
    definition: DefinitionModel,
    coverage: CoverageModel,
    entity_labels: Mapping[str, str],
    *,
    family: str | None,
    breakdown: BreakdownLevel | None,
) -> str:
    """A factual line built only from the identity document and its own entities.

    No prose source exists here the way `search`'s `rendered_sentence` does --
    `StoredDefinition` carries only `{spec_id, schema_version, canonical_bytes}`
    -- so this states only what the identity document itself says: what shape
    was minted, and for whom. A summed mint instead names its `family`
    (`_family_name`) and says the family's StatVars were summed into one total series
    per entity role, or, on a `breakdown` mint, into one series per entity role and code
    at the breakdown's level. apis sums away the
    StatVar axis and keeps every declared role column, a role the question
    left open included, so the roles come from `coverage.dimensions`, one per
    frame role column plus `years`, less a breakdown's `stat_var_<code_system>`
    column, which the code names. The identity's `entities` hold only the
    bound roles, keyed by flow axis, and its `open_roles` the axes it covers
    every member of, each named "every <noun>" by the role column the response
    declares on that axis (`creditor` on IDS, `financingProvider` on CRS), or by
    `_GRAIN_NOUNS` when the response declares none.
    """
    identity = definition.identity
    shape = identity.get("shape")
    shape_id = shape.get("id") if isinstance(shape, Mapping) else None
    if family is not None:
        subject = family
    elif isinstance(shape_id, str) and shape_id:
        subject = shape_id
    else:
        subject = "the requested data"

    entities = identity.get("entities")
    pieces = []
    if isinstance(entities, Mapping):
        for role, dcids in entities.items():
            if isinstance(dcids, list) and dcids:
                pieces.append(f"{role} {render_dcids(dcids, entity_labels)}")
    open_roles = identity.get("open_roles")
    if isinstance(open_roles, list):
        axis_labels = {dim.axis: dim.label for dim in coverage.dimensions if dim.axis}
        pieces.extend(
            f"every {axis_labels.get(role) or _GRAIN_NOUNS.get(role, role)}" for role in open_roles
        )
    scope_text = " · ".join(pieces)

    line = f"Retrieved {subject} for {scope_text}" if scope_text else f"Retrieved {subject}"
    if family is None:
        return f"{line}."
    code_column = None if breakdown is None else f"stat_var_{breakdown.code_system}"
    nouns = dict.fromkeys(
        _GRAIN_NOUNS.get(dim.axis or "", "entity")
        for dim in coverage.dimensions
        if dim.label not in (_YEARS_DIMENSION, code_column)
    )
    grain = " and ".join(nouns) or "entity"
    if breakdown is None:
        return f"{line}, with its StatVars summed into one total series per {grain}."
    return (
        f"{line}, with its StatVars summed into one series per {grain} and "
        f"{breakdown.code_system} code at level {breakdown.level}."
    )


def render_dcids(dcids: Sequence[Any], entity_labels: Mapping[str, str]) -> str:
    names = [dcid if not isinstance(dcid, str) else entity_labels.get(dcid, dcid) for dcid in dcids]
    return _joined_with_cap([str(name) for name in names], _ACKNOWLEDGEMENT_ENTITY_CAP)


def _joined_with_cap(items: Sequence[str], cap: int) -> str:
    """Comma-join `items`, naming at most `cap` and summarising the rest.

    Shared by every digest line whose item count can grow with the frame
    rather than with what the model needs to know, so the line's length
    stays bounded instead of tracking frame breadth.
    """
    shown = items[:cap]
    remainder = len(items) - len(shown)
    text = ", ".join(shown)
    if remainder <= 0:
        return text
    return f"{text}, and {remainder} more" if text else f"{remainder} more"


def _coverage_line(coverage: CoverageModel, definition: DefinitionModel) -> str:
    row_word = "row" if coverage.rows == 1 else "rows"
    pieces = [f"{coverage.rows} {row_word}"]
    dims_text = ", ".join(f"{dim.count} {dim.label}" for dim in coverage.dimensions)
    if dims_text:
        pieces.append(dims_text)
    window_text = _coverage_window_text(
        coverage.window, definition.identity.get("requested_window")
    )
    if window_text:
        pieces.append(window_text)
    return "Coverage: " + " · ".join(pieces)


def _coverage_window_text(window: Mapping[str, object] | None, requested_window: object) -> str:
    """The coverage line's window piece, naming a requested range the served window fell short of.

    `requested_window` is the identity document's own record of what the caller
    asked for. Only its `range` kind carries years to compare. Its `all` and `latest`
    kinds name no specific span, so the served window is the only fact worth stating.
    """
    served = _window_text(window)
    if isinstance(requested_window, Mapping) and requested_window.get("kind") == "range":
        requested = _window_text(cast("Mapping[str, object]", requested_window))
        if requested and requested != served:
            return f"requested {requested}, served {served}" if served else f"requested {requested}"
    return served


def _window_text(window: Mapping[str, object] | None) -> str:
    if not window:
        return ""
    start, end = window.get("start_year"), window.get("end_year")
    if start is not None and end is not None:
        return f"{start}-{end}"
    if start is not None:
        return f"since {start}"
    if end is not None:
        return f"through {end}"
    return ""


def _components_lines(components: Sequence[ManifestComponent]) -> list[str]:
    """The line stating a composite frame's components, none for a frame of one table.

    Each component reads as its role, its rows, the years it serves, the zeros written
    for it and the years left out of it, all as counts and year spans. Components serving
    disjoint years are spliced, and the line names each year the measure changes at, so
    a change across it is read as comparing two measures.
    """
    if not components:
        return []
    line = "Components: " + " · ".join(_component_piece(c) for c in components) + "."
    served = sorted(
        (component for component in components if component.periods),
        key=lambda component: component.periods[0][0],
    )
    years = [_years_of(component.periods) for component in served]
    spliced = len(served) > 1 and sum(len(held) for held in years) == len(set().union(*years))
    if spliced:
        breaks = [
            f"between {max(before)} and {min(after)}, from {_component_noun(earlier)} to "
            f"{_component_noun(later)}"
            for (earlier, before), (later, after) in itertools.pairwise(
                zip(served, years, strict=True)
            )
        ]
        line += (
            f" The measure changes {'; and '.join(breaks)}: a change across those years "
            "compares two measures."
        )
    return [line]


def _component_piece(component: ManifestComponent) -> str:
    """One component as the components line states it."""
    row_word = "row" if component.rows == 1 else "rows"
    piece = f"{_component_noun(component)} {component.rows} {row_word}"
    if component.periods:
        piece += f", {_spans_text(component.periods)}"
    if component.zero_rows:
        piece += f", {component.zero_rows} of them zeros written where its table reports nothing"
    if component.dropped_years:
        dropped = sorted(set(component.dropped_years))
        year_word = "year" if len(dropped) == 1 else "years"
        piece += (
            f", {len(dropped)} {year_word} left out ({_spans_text(_spans(dropped))}) as "
            "outside the years every component covers"
        )
    return piece


def _component_noun(component: ManifestComponent) -> str:
    return _COMPONENT_NOUNS.get(component.role, component.role)


def _years_of(periods: Sequence[tuple[int, int]]) -> set[int]:
    return {year for first, last in periods for year in range(first, last + 1)}


def _spans(years: Sequence[int]) -> list[tuple[int, int]]:
    """Sorted, distinct `years` as closed runs of consecutive years."""
    spans: list[tuple[int, int]] = []
    for year in years:
        if spans and year == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], year)
        else:
            spans.append((year, year))
    return spans


def _spans_text(spans: Sequence[tuple[int, int]]) -> str:
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in spans)


def _frame_reference_line(frame_ref: FrameRef) -> str:
    return (
        f"{FRAME_REFERENCE_PREFIX}{frame_ref.spec_id}, content_version={frame_ref.content_version} "
        "-- pass both to downstream tools."
    )


def _citation_line(citation: CitationModel) -> str:
    labels = _joined_with_cap([source.label for source in citation.sources], _CITATION_SOURCE_CAP)
    if not citation.text:
        return f"Source: {labels}."
    return f"Source: {citation.text} ({labels})." if labels else f"Source: {citation.text}."


def _description_lines(description: VersionDescription) -> list[str]:
    """The citation line with the latest period, then one line each for the steps,
    the caveats and the members apis could not fill, when there are any."""
    # `cite.text` is the composer's citation line, which names every source.
    citation = f"Source: {description.cite.text}"
    latest = description.freshness.latest_period
    lines = [f"{citation}; latest period {latest}." if latest else f"{citation}."]
    if description.steps:
        steps = [step.text for step in description.steps]
        lines.append(f"Steps: {_sentences_with_cap(steps, _DESCRIPTION_STEP_CAP, 'step')}")
    if description.caveats:
        caveats = [caveat.text for caveat in description.caveats]
        lines.append(f"Caveats: {_sentences_with_cap(caveats, _DESCRIPTION_CAVEAT_CAP, 'caveat')}")
    unknown = list(dict.fromkeys(_gap_words(path) for path in description.gaps))
    if unknown:
        lines.append(f"Not known: {_joined_with_cap(unknown, _DESCRIPTION_GAP_CAP)}.")
    return lines


def _sentences_with_cap(sentences: Sequence[str], cap: int, noun: str) -> str:
    """`sentences` joined by spaces, naming at most `cap` and counting the rest."""
    text = " ".join(sentence.strip() for sentence in sentences[:cap])
    remainder = len(sentences) - cap
    if remainder <= 0:
        return text
    return f"{text} {remainder} more {noun}{'' if remainder == 1 else 's'} not shown."


def _gap_words(path: str) -> str:
    """A gap's dotted path as words: the members below its block, without list
    indexes, so `freshness.source_release` reads "source release" and a bare
    `steps` reads "steps"."""
    segments = path.split(".")
    named = [segment for segment in segments[1:] or segments if not segment.isdigit()]
    return " ".join(named).replace("_", " ")


def _left_out_totals_lines(
    entries: Sequence[PartyTotalsLeftOut], entity_labels: Mapping[str, str]
) -> list[str]:
    """The lines naming the group totals apis left out of the frame on each party role
    (`options.without_party_totals`), one per entry, each total by its label when this call
    resolved it. The population a search names for a side's whole total follows the last line
    of the role, once however many datasets the role lost totals in."""
    last_of_role = {entry.role: index for index, entry in enumerate(entries)}
    lines = []
    for index, entry in enumerate(entries):
        line = _left_out_totals_line(entry, entity_labels)
        population = WHOLE_SIDE_POPULATIONS.get(entry.role)
        if population is not None and last_of_role[entry.role] == index:
            line = (
                f"{line} For the total across the whole side, search again naming the "
                f'population ("{population}") with no operation over that side.'
            )
        lines.append(line)
    return lines


def _left_out_totals_line(left_out: PartyTotalsLeftOut, entity_labels: Mapping[str, str]) -> str:
    """The line naming the group totals one entry left out of its party role. The entry's
    scope follows the role: a provenance names the dataset by its dcid, since the manifest
    carries no label for it; a null provenance says the rows carry no dataset; an entry
    recorded without the key comes from a trace whose cut covered every dataset, so it
    names none."""
    totals = [entity_labels.get(ref, ref) for ref in left_out.entities]
    noun = "group total" if len(totals) == 1 else "group totals"
    names = _joined_with_cap(totals, _LEFT_OUT_TOTALS_CAP)
    if "provenance" not in left_out.model_fields_set:
        scope = ""
    elif left_out.provenance is None:
        scope = " in the rows that carry no dataset"
    else:
        scope = f" in {left_out.provenance}"
    return (
        f"Left out of the frame: {len(totals)} {noun} on {left_out.role}{scope} "
        f"({names}); the frame holds that side's other parties."
    )


def _dropped_role_line(role: str) -> str:
    return (
        f"Note: this call's entities override did not restate {role!r}, so that "
        "role is no longer bound in the re-minted definition."
    )


def _remint_persist_failed_line() -> str:
    return (
        "Note: this call re-minted a new spec_id, and could not confirm that the definition "
        "store holds it -- the frame reference above is real and usable now, but a later "
        "get_data call passing this spec_id may refuse definition_not_held rather than "
        "find it."
    )


def _operation_poll_line(note: OperationPollNote) -> str:
    attempt_word = "attempt" if note.poll_attempts == 1 else "attempts"
    return (
        f"{OPERATION_POLL_PREFIX} and ran as a queued operation "
        f"({note.operation_id}) -- {note.poll_attempts} poll {attempt_word} over "
        f"{note.elapsed_s:.0f}s before completing. The frame reference above is the same "
        "as any other successful mint."
    )


def _degraded_frame_note(outcome: MaterialisationOutcome) -> str:
    # `materialise_and_register` reads the frame store before it fetches, so a repeat
    # call finds a frame that has landed since without fetching it, and otherwise
    # fetches and writes it again.
    retry = (
        "Call get_data again with this spec_id to retry: it uses the frame if the store "
        "holds it by then, and otherwise fetches and writes it again."
    )
    if outcome.degraded_reason == "store_write":
        return (
            "Note: the artifact behind the frame reference above was fetched, but the "
            "frame store did not confirm the write before this call's deadline. The mint "
            "succeeded and the reference names a real, durable result, and the write may "
            "still land -- until it does, any analysis verb given this frame_ref refuses "
            f"frame_not_held. {retry}"
        )
    # `attempts` can be `0`: a sign failure or a crossed deadline before the first GET
    # ever completed means no attempt actually landed, and saying "after 0 attempt(s)"
    # would misstate that as a GET that was tried and failed.
    retrieval = (
        "without a single attempt reaching the artifact"
        if outcome.attempts == 0
        else f"after {outcome.attempts} attempt(s)"
    )
    if outcome.degraded_reason == "store_unchecked":
        held = (
            "The frame store did not answer in time either, so this call could not check "
            "whether it already holds the frame -- if it does not, any analysis verb given "
            "this frame_ref refuses frame_not_held."
        )
    else:
        held = (
            "No bytes are held yet -- any analysis verb given this frame_ref will refuse "
            "frame_not_held until a retry succeeds."
        )
    return (
        f"Note: the artifact behind the frame reference above could not be retrieved "
        f"{retrieval}. The mint succeeded and the reference names a real, durable result. "
        f"{held} {retry}"
    )


def _statistics_sections(
    manifest: Manifest,
    series_stats: Sequence[SeriesStatisticsModel] | None,
    *,
    fetch_failed: bool,
    summed_ref: str | None,
    family: str | None,
    breakdown: BreakdownLevel | None,
) -> list[str]:
    """One rendered block per series, or a single count-only line, or nothing.

    A series the mint's estimates continue (`continuing_method`) holds two
    rows, one per method, and renders as one block, its reported row's
    figures followed by a line naming the estimates' years
    (`continued_series`).

    `manifest.series_count` is `None` for a manifest minted before the field existed
    -- that snapshot renders everything else and simply carries no
    statistics section, the same as `0` (a mint that resolved to no series
    at all): neither is worth a line. Past `_STATISTICS_SERIES_CAP`, or on a
    fetch failure the caller already tried and reports via `fetch_failed`,
    only the count is rendered -- `series_stats` is never read in either
    case. Otherwise `series_stats` is exactly what the caller fetched from
    the manifest's own statistics artifact; series membership comes from
    those rows alone, never from `coverage.dimensions` or any other reading
    of the frame's own rows. `summed_ref` is the `stat_var` of a summed
    family's series and `family` its name (`_family_name`), both `None` for a
    frame that summed none. On a `breakdown` mint each row holds the family summed
    within one code, which the row's `stat_var_<code_system>` entity names.
    """
    series_count = manifest.series_count
    if not series_count:
        return []
    if series_count > _STATISTICS_SERIES_CAP:
        return [
            f"Statistics: {series_count} series retrieved -- too many to "
            f"summarise individually (cap is {_STATISTICS_SERIES_CAP})."
        ]
    if fetch_failed:
        return [
            f"Statistics: {series_count} series retrieved -- could not fetch or decode "
            "the statistics artifact, showing the count only."
        ]
    return [
        _series_statistics_block(
            series.reported,
            family if series.reported.stat_var == summed_ref else None,
            estimates=series.estimates,
            breakdown=breakdown,
        )
        for series in continued_series(series_stats or (), continuing=continuing_method(manifest))
    ]


def _series_statistics_block(
    series: SeriesStatisticsModel,
    family: str | None,
    *,
    estimates: SeriesStatisticsModel | None = None,
    breakdown: BreakdownLevel | None = None,
) -> str:
    """One series' block, headed as the summed total of `family` when it is one, or
    on a `breakdown` as `family` summed within the code its `stat_var_<code_system>`
    entity names, with a line for the `estimates` continuing it past its reported end."""
    code_column = None if breakdown is None else f"stat_var_{breakdown.code_system}"
    code = next((entity.ref for entity in series.entities if entity.role == code_column), None)
    if family is None:
        header = series.stat_var
    elif breakdown is not None and code is not None:
        header = f"Sum of {family} within {breakdown.code_system} code {code}"
    else:
        header = f"Summed total of {family}"
    header_pieces = [header]
    header_pieces.extend(
        f"{entity.role} {entity.ref}"
        for entity in series.entities
        if family is None or entity.role != code_column
    )
    unit = series.facet.get("unit")
    if isinstance(unit, str):
        reading = _unit_reading(unit)
        header_pieces.append(f"unit {unit} ({reading})" if reading else f"unit {unit}")
    first = f"first {_format_magnitude(series.first.value)} ({series.first.period})"
    latest = f"latest {_format_magnitude(series.latest.value)} ({series.latest.period})"
    minimum = f"min {_format_magnitude(series.minimum.value)} ({series.minimum.period})"
    maximum = f"max {_format_magnitude(series.maximum.value)} ({series.maximum.period})"
    lines = [
        " · ".join(header_pieces),
        f"  {first} · {latest} · {minimum} · {maximum}",
    ]
    if series.change is not None:
        change_line = f"  change over window {_format_magnitude(series.change.absolute)}"
        if series.change.percent is not None:
            change_line += f" ({_format_percent(series.change.percent)}%)"
        lines.append(change_line)
    if estimates is not None:
        method = estimates.facet.get("measurementMethod")
        lines.append(
            f"  continued {estimates.first.period} to {estimates.latest.period} by its "
            f"source's {method} values, part of this one series: "
            f"first {_format_magnitude(estimates.first.value)} ({estimates.first.period}) · "
            f"latest {_format_magnitude(estimates.latest.value)} ({estimates.latest.period})"
        )
    return "\n".join(lines)


def _format_magnitude(value: float) -> str:
    """Scale a statistic value to a short magnitude string, e.g. `12.4m`, `-2.6m`.

    Always scaled to thousand/million/billion at one decimal place, so a
    statistic (a citable figure the materialisation library computed) never
    reads back as a literal observation value.
    """
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{sign}{magnitude / 1_000_000_000:.1f}b"
    if magnitude >= 1_000_000:
        return f"{sign}{magnitude / 1_000_000:.1f}m"
    if magnitude >= 1_000:
        return f"{sign}{magnitude / 1_000:.1f}k"
    return f"{sign}{magnitude:.1f}"


def _format_percent(value: float) -> str:
    return f"{value:.0f}"


def _unit_reading(unit: str) -> str | None:
    """A plain-language current-vs-constant-prices reading of a recognised unit code.

    Matches only the SDG/OECD price-basis form and this deployment's own
    current- and constant-price refs. Every other code, recognised elsewhere
    or not, has no reading here and renders as itself.
    """
    match = _SDG_PRICE_BASIS_RE.match(unit)
    if match:
        return _price_basis_reading(
            current=match.group(1) == "CU",
            year=match.group(2),
            currency=match.group("currency"),
            magnitude=match.group("magnitude"),
        )
    if unit in _DEPLOYMENT_CURRENT_REFS:
        return _price_basis_reading(current=True, year=None, currency=unit, magnitude=None)
    match = _DEPLOYMENT_CONSTANT_RE.match(unit)
    if match:
        return _price_basis_reading(
            current=False,
            year=match.group("year"),
            currency=match.group("currency"),
            magnitude=None,
        )
    return None


def _price_basis_reading(
    *, current: bool, year: str | None, currency: str, magnitude: str | None
) -> str:
    reading = price_basis_reading(
        currency=currency, prices="current" if current else "constant", base_year=year
    )
    return f"{reading}, {_MAGNITUDE_WORDS[magnitude]}" if magnitude else reading


def _budget_warnings(
    coverage_warnings: Sequence[WarningModel], other_warnings: Sequence[WarningModel]
) -> tuple[list[WarningModel], list[WarningModel], int]:
    """Split one shared budget of `_MANIFEST_WARNING_CAP` across both warning groups.

    Coverage warnings keep their position beside the coverage line and draw
    from the budget first; whatever budget remains goes to the other
    warnings at the end. Whatever isn't shown from either group is combined
    into a single remainder count, reported once.
    """
    shown_coverage = list(coverage_warnings[:_MANIFEST_WARNING_CAP])
    remaining_budget = _MANIFEST_WARNING_CAP - len(shown_coverage)
    shown_other = list(other_warnings[:remaining_budget]) if remaining_budget > 0 else []
    remainder = (len(coverage_warnings) - len(shown_coverage)) + (
        len(other_warnings) - len(shown_other)
    )
    return shown_coverage, shown_other, remainder


def warning_refs(warnings: Sequence[WarningModel]) -> list[str]:
    """Every ref the counted fields of `warnings` name, once each, in the order
    `labelled_warnings` names them."""
    refs = (
        ref for warning in warnings for field in _counted_fields(warning) for ref, _ in field.named
    )
    return list(dict.fromkeys(refs))


class _CountedField(NamedTuple):
    """One counted field of a warning: the noun its sentence opens with, how many things
    it counts, and the `(ref, years)` of those it names, `years` empty for every field
    but `omitted_entity_years`."""

    noun: str
    count: int
    named: Sequence[tuple[str, Sequence[str]]]


def _counted_fields(warning: WarningModel) -> list[_CountedField]:
    """The counted fields `warning` carries, in the order they are rendered."""
    fields = [
        _CountedField(noun, counted.count, [(ref, ()) for ref in counted.refs])
        for noun, counted in (
            ("Series at another frequency", warning.off_cadence_members),
            ("Series with no observation in the requested years", warning.out_of_window_members),
            ("Entities with no observation", warning.unserved_entities),
            ("Entities left out", warning.omitted_entities),
        )
        if counted is not None
    ]
    if (entity_years := warning.omitted_entity_years) is not None:
        named = [(entity.ref, entity.years) for entity in entity_years.entities]
        fields.append(_CountedField("Entities left out", entity_years.count, named))
    return fields


def _counted_field_sentence(field: _CountedField, labels: Mapping[str, str]) -> str:
    """`field`'s noun and the names of the things it names, each with its years, then how
    many of its count it names none of."""
    names = []
    for ref, years in field.named:
        name = labels.get(ref) or ref.removeprefix(ENTITY_DCID_MARKER)
        names.append(f"{name} ({', '.join(years)})" if years else name)
    text = ", ".join(names)
    if (unnamed := field.count - len(field.named)) > 0:
        text = f"{text}, and {unnamed} more" if text else f"{unnamed} more"
    return f"{field.noun}: {text}."


def labelled_warnings(
    warnings: Sequence[WarningModel], labels: Mapping[str, str]
) -> list[WarningModel]:
    """`warnings`, each message followed by a sentence for each counted field its warning
    carries, naming the field's refs by `labels` and a ref outside them by its bare dcid.

    apis' messages state counts only and leave the naming to the reader. `get_data`
    labels the manifest once, and its digest and returned manifest both carry these
    messages, so the harness's card copy reads the names the digest does. A warning with
    no counted field keeps its message, and every warning keeps its fields.
    """
    return [
        warning.model_copy(update={"message": _labelled_message(warning, labels)})
        for warning in warnings
    ]


def _labelled_message(warning: WarningModel, labels: Mapping[str, str]) -> str:
    sentences = (_counted_field_sentence(field, labels) for field in _counted_fields(warning))
    return " ".join([warning.message, *sentences])


def _warning_lines(warnings: Sequence[WarningModel]) -> list[str]:
    """One line per manifest warning, with no cap or remainder of its own."""
    return [
        f"Warning [{warning.code}] ({warning.severity}): {warning.message}" for warning in warnings
    ]
