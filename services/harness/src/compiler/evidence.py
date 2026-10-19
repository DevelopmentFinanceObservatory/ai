"""Does what was found satisfy what a requirement asks for?

`check_evidence` compares one `DatasetRequirement` with `RetrievalEvidence`,
the facts known about a retrieval. Only a dataset requirement has evidence to
check -- a derived requirement rests on one through the chain the plan's
requirement graph already validates. Before approval that evidence comes from
the source discovery selected (`evidence_from_source`), so a requirement a
source cannot meet is refused before anyone approves the plan. At run time it
comes from the manifest `get_data` returned (`evidence_from_manifest`), which
also carries each series' observation count and first and last periods, so
the same function then checks what was retrieved against what was approved:
the identity is still the one approved, the required entities came back,
each series covers the period with one observation per year, and rows were
returned.

`measure` is not compared here: it is prose, and the structural stand-in for
it is the `spec_id` and the identity document it hashes, which this module
already holds stable through `SOURCE_DRIFT`. What is new is that the identity
describes a measure rather than enumerating one -- a schema-shaped identity
names up to five schema properties and pins no dcid, which is how one
definition legitimately covers hundreds of stat_vars -- so the definition's
own binding under a stable identity is checked separately, by
`STAT_VAR_MEMBERSHIP_DRIFT`: the digest of the family `search` stated at
approval (`resolved.shape.member_digest`) against the digest of the family
the mint admitted at execution (the manifest trace's `family_member_digest`).
Both digest every member the bindings admit, before the mint reduces the
family to the members no other member dominates. A side that states no
digest leaves membership unverified (`_digest_issues`).
The stat_vars a retrieval's per-series statistics name are a different
thing, coverage rather than binding: a series absent from them may
simply hold no data for the entities and years asked, which is sparsity, not
drift, so `STAT_VAR_MEMBERSHIP_DRIFT` never reads that set. Two residuals are
unchecked: membership can drift between two `get_data` calls against one
`spec_id` inside a single run, which an approval-versus-execution comparison
does not see; and this reads the manifest's resolved binding, not the
stat_var that reached analysis, so which series the mint picked when it
bound several stays unverified.

`evidence_from_manifest` also reads `provenance_ref`/`provenance_label` off
the manifest's own `citation.sources`, so a requirement's stated source is
compared against what was actually retrieved, not only against what
discovery offered before approval.

Every finding is an `EvidenceIssue` with a severity. An `ERROR` means the
requirement is not met -- a missing, role-ambiguous or wrong-sided entity, a
population of every member that no open axis of the source holds, a currency
or price basis that names something other than what was asked for, a source
pinned or stated that does not match what was found, a definition whose
identity is unknown, is no longer the one approved, or does not bind a
required entity, a retrieval that returned no rows or whose grain is not
annual, or (when `coverage.on_missing` is `"fail"`) a coverage window or a
series short of the period. A `WARNING` means it could not be verified from
this evidence -- accepted, but recorded for the approval summary. What a
sparse envelope leaves out (its data, its entities, its window) and the years
a retrieved series is missing are unverified at the severity the
requirement's `coverage` asks for, at either origin, so a requirement that
fails on gaps is not defeated by a source that states nothing. The manifest,
though, is the record of what was actually retrieved, so a check that cannot
run there is not a passed check: at manifest origin an unrecognisable or
absent unit is an `ERROR` where at discovery it is a `WARNING`, and missing
per-series statistics -- `None` or an empty list alike -- are always an
`ERROR`, whatever `coverage.on_missing` says, since missing mandatory
metadata is not a gap to report but a result that cannot be checked. `INFO`
records a check that passed and is worth showing.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Literal, NamedTuple, cast

import pyarrow as pa
import pyarrow.compute as pc
from common.price_basis import price_basis_reading
from data_mcp.manifest import Manifest
from data_mcp.series_statistics import continuing_method, series_index
from dfo_frame_grammar.columnar import array_kernel, scalar_kernel, unit_codes
from planner.parties import Party, across_party, flow_side, place_qualified_party
from planner.phrasing import pluralize
from planner.plan import DatasetRequirement, Entity, Unit

# `Severity`, `EvidenceIssueCode` and `EvidenceIssue` live in `planner.records`
# so a `Workflow` can carry its findings; they are re-exported here so the
# compiler and executor keep importing them from this module.
from planner.records import (
    EvidenceIssue,
    EvidenceIssueCode,
    Period,
    ResolvedSourceMention,
    Severity,
    SpecId,
    StrictModel,
)
from planner.workflow import OpenEntityAxis, ResolvedEntity, ResolvedGroup, SelectedSource
from pydantic import ConfigDict, Field, JsonValue, ValidationError

from compiler.frame_columns import columns_on_axes
from compiler.sources import pinned_stat_var, source_constraint_matches


class SeriesEvidence(StrictModel):
    """One series of a retrieved frame whose coverage `_series_issues` reports on."""

    stat_var: str
    entity_refs: tuple[str, ...]
    first_year: int
    latest_year: int
    observations: int


class RetrievalEvidence(StrictModel):
    """The facts known about a retrieval, from whichever side supplied them.

    Discovery supplies the source's own account of itself; the manifest, what
    was retrieved. `rows` and `series` are the manifest's alone. `has_data`
    and `identity` are `None` when the side that supplied the evidence stated
    nothing, which is unverified rather than fine; a manifest always counts
    its rows, so only discovery leaves `has_data` unknown.

    `stat_vars` is coverage: the dcids of the series the fetched statistics
    actually carried observations for, used only by `_series_issues`. `resolved_stat_vars`
    is the definition's binding at execution -- every stat_var the manifest's
    `trace` says the spec resolved to, whether or not that stat_var's series
    holds any observations -- and is what `STAT_VAR_MEMBERSHIP_DRIFT` compares
    against the binding recorded at approval. It is `None` at discovery
    origin, where there is no execution trace yet, and `None` at manifest
    origin when the trace carries no `resolved_stat_vars` at all; either way
    membership treats that as unverified, never as an empty binding.

    `family_member_digest` and `family_member_count` are the manifest
    `trace`'s account of the family the mint admitted before it reduced it
    to `resolved_stat_vars`: the digest `search` states as
    `resolved.shape.member_digest`, over the same population. Both are
    `None` when the trace states none, as an apis service that predates the
    fields does and as one that cannot state the whole family does.

    `series` is the manifest's statistics artifact itself, in
    `data_mcp.series_statistics.SERIES_STATISTICS_ARROW_SCHEMA` plus a
    `series` column, kept as Arrow: a retrieval can carry tens of thousands
    of series, and the checks read them column by column. A row is one series
    under one method, so a series the mint's estimates continue holds two
    rows; `series` names each row's series by its head row's position
    (`data_mcp.series_statistics.series_index`).

    `open_entity_axes` are the axes the source states it left open, discovery's
    alone: a manifest names the entities it returned, and the approved
    source says which of its columns was open. `None` means they were never
    read, which is the state of every manifest and of a source recorded
    without them.

    `groups` are the named sets the source binds (`SelectedSource.groups`),
    and `entity_refs` the refs the graph resolved the requirement's
    unmatched names to (`SelectedSource.entity_refs`), both discovery's
    alone: at run time the approved source's are read instead, as its open
    axes are.

    `axis_by_column` is the manifest's alone: each coverage dimension's label
    and the flow axis apis states for it. A source names a flow
    dataset's role columns by flow axis (`origin`), and the series
    statistics name each entity's role by the column the frame carries
    (`financingProvider`), so this is what ties the two together. `None` at
    discovery origin.

    A unit is stated in one of two forms. `unit_ref` and `unit_label` hold a
    unit named by reference (a Data Commons unit dcid such as `CurrencyUSD`)
    and its text. `stated_unit` holds a unit stated field by field, the
    frame grammar's `{currency, prices, base_year}` object a mint serves when
    its definition binds the unit by request. Only a manifest carries it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    origin: Literal["discovery", "manifest"]
    spec_id: SpecId
    has_data: bool | None
    stat_vars: tuple[str, ...] = ()
    resolved_stat_vars: tuple[str, ...] | None = None
    family_member_digest: str | None = None
    family_member_count: int | None = None
    entities: tuple[ResolvedEntity, ...] = ()
    open_entity_axes: tuple[OpenEntityAxis, ...] | None = None
    groups: tuple[ResolvedGroup, ...] = ()
    entity_refs: dict[str, str] = Field(default_factory=dict)
    axis_by_column: dict[str, str | None] | None = None
    unit_ref: str | None = None
    unit_label: str | None = None
    stated_unit: Unit | None = None
    provenance_ref: str | None = None
    provenance_label: str | None = None
    resolved_sources: tuple[ResolvedSourceMention, ...] = ()
    window: Period | None = None
    rows: int | None = None
    series: pa.Table | None = None
    identity: dict[str, JsonValue] | None = None


def evidence_from_source(source: SelectedSource) -> RetrievalEvidence:
    """The evidence a discovered source offers before anything is retrieved."""
    return RetrievalEvidence(
        origin="discovery",
        spec_id=source.spec_id,
        has_data=source.has_data,
        entities=source.entities,
        open_entity_axes=source.open_entity_axes,
        groups=source.groups,
        entity_refs=source.entity_refs,
        unit_ref=source.unit_ref,
        unit_label=source.unit_label,
        provenance_ref=source.provenance_ref,
        provenance_label=source.provenance_label,
        resolved_sources=source.resolved_sources,
        window=source.coverage_window,
        identity=source.identity,
    )


def evidence_from_manifest(
    manifest: Manifest, *, series_stats: pa.Table | None = None
) -> RetrievalEvidence:
    """The evidence a `get_data` manifest gives about what was retrieved.

    `series_stats` is the manifest's own statistics artifact, already fetched
    by the caller (`executor.run`, through `data_mcp.series_statistics
    .fetch_series_statistics_table`) -- per-series figures live in a Parquet
    artifact rather than on the manifest, so this function does no I/O of
    its own and reads whatever the caller resolved. Entities and the unit come
    from it, the window from the earliest first and latest last period
    across it, each read over the columns and their distinct values rather
    than series by series; `None` (the manifest carries no statistics artifact,
    or the fetch failed) or a table with no rows (`series_count == 0`) leaves
    those unknown, and the checks that read `series` treat both the same as
    unavailable, per `_series_issues`. Provenance comes from
    `citation.sources`: one source's `ref`/`label`, several joined with
    `"; "`. `axis_by_column` comes from `coverage.dimensions`, each label with
    its flow axis. An empty identity is recorded as none held, so it compares as an
    approved source that held none does. `resolved_stat_vars` comes from
    `trace.resolved_stat_vars` -- upstream's own record of every StatVar the
    spec resolved to at mint time -- read separately from `series_stats`
    since a stat_var the spec resolved to but that carried no observations
    for the entities and years asked is still part of the binding, not
    absent from it.
    """
    units = _series_units(series_stats) if series_stats is not None else set()
    unit = next(iter(units)) if len(units) == 1 else None
    sources = manifest.citation.sources
    return RetrievalEvidence(
        origin="manifest",
        spec_id=manifest.definition.spec_id,
        has_data=manifest.coverage.rows > 0,
        stat_vars=_series_stat_vars(series_stats) if series_stats is not None else (),
        resolved_stat_vars=_resolved_stat_vars(manifest.trace),
        family_member_digest=_family_member_digest(manifest.trace),
        family_member_count=_family_member_count(manifest.trace),
        entities=_series_entities(series_stats) if series_stats is not None else (),
        axis_by_column={
            dimension.label: dimension.axis for dimension in manifest.coverage.dimensions
        },
        unit_ref=unit if isinstance(unit, str) else None,
        stated_unit=unit if isinstance(unit, Unit) else None,
        provenance_ref="; ".join(one.ref for one in sources) or None,
        provenance_label="; ".join(one.label for one in sources) or None,
        window=_window_of(series_stats),
        rows=manifest.coverage.rows,
        series=None
        if series_stats is None
        else series_stats.append_column(
            "series", series_index(series_stats, continuing=continuing_method(manifest))
        ),
        identity=cast(dict[str, JsonValue], dict(manifest.definition.identity)) or None,
    )


def check_evidence(
    requirement: DatasetRequirement,
    evidence: RetrievalEvidence,
    *,
    approved: SelectedSource | None = None,
) -> list[EvidenceIssue]:
    """Compare `evidence` with `requirement`, reporting every finding.

    Args:
        requirement: What an acceptable result has to be.
        evidence: What is known about the retrieval.
        approved: The source the plan was approved with, when checking a
            retrieval at run time; its identity is what the manifest's has to
            match, and its resolutions are how a manifest's bare refs are
            matched to the requirement's entity names.

    Returns:
        Every issue found, errors and warnings alike, so one pass reports
        everything the planner or the user has to weigh.
    """
    issues: list[EvidenceIssue] = []
    rid = requirement.id
    at_discovery = evidence.origin == "discovery"
    if evidence.has_data is False:
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.NO_DATA_AT_SOURCE
                if at_discovery
                else EvidenceIssueCode.ROWS_EMPTY,
                severity=Severity.ERROR,
                requirement_id=rid,
                detail=f"source {evidence.spec_id} "
                + ("reports no data" if at_discovery else "returned no rows"),
            )
        )
    elif evidence.has_data is None:
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.DATA_UNVERIFIED,
                severity=_coverage_severity(requirement),
                requirement_id=rid,
                detail=f"source {evidence.spec_id} states no coverage, so whether it holds data cannot be confirmed",
            )
        )
    if at_discovery and evidence.identity is None:
        # Drift detection compares an approved identity against the manifest's;
        # a source whose identity was never known cannot be protected by it.
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.IDENTITY_UNVERIFIED,
                severity=Severity.ERROR,
                requirement_id=rid,
                detail=f"source {evidence.spec_id} carries no identity document, so drift from it can never be detected",
            )
        )
    if approved is not None:
        issues.extend(_drift_issues(requirement, evidence, approved))
    if not at_discovery:
        issues.extend(_series_issues(requirement, evidence))
    issues.extend(_entity_issues(requirement, evidence, approved))
    issues.extend(_period_issues(requirement, evidence))
    issues.extend(_scope_issues(requirement, evidence))
    issues.extend(_unit_issues(requirement, evidence))
    issues.extend(_source_issues(requirement, evidence, approved))
    return issues


def errors(issues: list[EvidenceIssue]) -> list[EvidenceIssue]:
    """Return the issues that mean the requirement is not met."""
    return [issue for issue in issues if issue.severity is Severity.ERROR]


def entity_matches(entity: Entity, resolved: Sequence[ResolvedEntity]) -> list[ResolvedEntity]:
    """Return every resolved entity whose `label` matches `entity.name`, case-folded.

    Matched by name alone, before role narrows the field -- what
    `resolve_entity` starts from, and what a finding reports as what was
    found when the name alone does not pick out one resolution.
    """
    wanted = entity.name.casefold()
    return [one for one in resolved if one.label.casefold() == wanted]


def resolve_entity(entity: Entity, resolved: Sequence[ResolvedEntity]) -> ResolvedEntity | None:
    """Return the one resolved entity that answers `entity`, or `None` when none or several do.

    Matches `Entity.name` against `label`, case-folded. When the requirement
    states a role, a match whose role column names it, or sits on the same
    side of the flow (`donor` and `origin`), is preferred, so a name
    resolving under several role columns is told apart.
    """
    wanted = entity.name.casefold()
    matches = [one for one in resolved if one.label.casefold() == wanted]
    if len(matches) > 1:
        matches = [one for one in matches if _role_fits(entity.role, one.role)]
    return matches[0] if len(matches) == 1 else None


def resolve_name(
    entity: Entity, source: SelectedSource | RetrievalEvidence
) -> ResolvedEntity | None:
    """Return the one entity of `source` that `entity` names, by label or else by ref.

    A name no label matches (`resolve_entity`) is looked up in the refs the
    graph resolved the requirement's names to (`SelectedSource.entity_refs`),
    so "United States" names the entity labelled "United States of
    America". A stated role tells apart a ref resolved under several role
    columns, as it does a label.
    """
    resolved = resolve_entity(entity, source.entities)
    if resolved is not None or (ref := source.entity_refs.get(entity.name.casefold())) is None:
        return resolved
    matches = [one for one in source.entities if one.ref == ref]
    if len(matches) > 1:
        matches = [one for one in matches if _role_fits(entity.role, one.role)]
    return matches[0] if len(matches) == 1 else None


def resolve_group(entity: Entity, groups: Sequence[ResolvedGroup]) -> ResolvedGroup | None:
    """Return the one named set in `groups` that `entity` names, or `None` when none or several do.

    Matches `Entity.name` against the set's `label`, case-folded. A stated
    role has to name the set's role column or sit on its side of the flow
    (`donor` and `origin`), so "G7" asked for as a recipient is not the G7
    whose members give.
    """
    wanted = entity.name.casefold()
    matches = [
        group
        for group in groups
        if group.label.casefold() == wanted and _role_fits(entity.role, group.role)
    ]
    return matches[0] if len(matches) == 1 else None


def named_members(
    entity: Entity, source: SelectedSource | RetrievalEvidence
) -> tuple[ResolvedEntity, ...]:
    """The entities of `source` that `entity` names, empty when it names none.

    A name matching one of the source's named sets (`resolve_group`) names
    every entity the source resolves on that set's role column, which is the
    set's members or the one party that is its published total
    (`ResolvedGroup.total`). Any other name names the one entity
    `resolve_name` finds, by label or by the ref the graph resolved the
    name to. Callers compare the result by `ref`, so a member is named
    whatever label it carries.
    """
    group = resolve_group(entity, source.groups)
    if group is not None:
        return tuple(one for one in source.entities if one.role == group.role)
    resolved = resolve_name(entity, source)
    return () if resolved is None else (resolved,)


def named_refs(entities: Sequence[Entity], source: SelectedSource | RetrievalEvidence) -> set[str]:
    """The refs of every entity of `source` that one of `entities` names (`named_members`)."""
    return {one.ref for entity in entities for one in named_members(entity, source)}


def _role_fits(role: str | None, column: str) -> bool:
    """Whether the requirement's `role` phrase, `None` when unstated, allows the role `column`."""
    if role is None or role.casefold() == column.casefold():
        return True
    side = flow_side(role)
    return side is not None and flow_side(column) == side


def _coverage_severity(requirement: DatasetRequirement) -> Severity:
    """How a gap the evidence cannot rule out bears on `requirement`."""
    return Severity.ERROR if requirement.coverage.on_missing == "fail" else Severity.WARNING


# --- Entities -----------------------------------------------------------------

# The role column that holds the subject of a measure, the place.
_PLACE_COLUMN = "observationAbout"
# The fewest members a population holds: one member is an entity, not a
# population to rank or total.
_POPULATION_MINIMUM = 2


def _roles_conflict(phrase: str, column: str) -> bool:
    """Whether `phrase` and `column` name opposite sides of the flow.

    Only a conflict the lexicon can see is reported: a phrase or a column it
    does not know, and two words on the same side, are not a conflict.
    """
    phrase_side, column_side = flow_side(phrase), flow_side(column)
    return phrase_side is not None and column_side is not None and phrase_side != column_side


def open_axes_for(
    across: Sequence[str], open_entity_axes: Sequence[OpenEntityAxis] | None
) -> tuple[OpenEntityAxis, ...] | str:
    """The open entity axis each `across` word covers, in `across` order, or why no such set exists.

    The one place a plan's party words ("donor", "country") are matched to a
    source's open columns. `planner.parties.across_party` reads each word: a
    side of a flow, or the place. A word it cannot read describes a narrower
    population than a party ("African country", "bilateral donor", "donor
    countries"), which an open axis does not hold: the axis ranks every
    member.

    A word naming a side of a flow covers the one open axis on that side. A
    place word covers the source's place axis, `observationAbout`, when that
    axis is open over several members. Otherwise it covers the one open
    flow axis holding several members that no other word covers, which is
    how a source keyed by debtor or recipient states its countries. Every
    covered axis has to hold at least two members by the source's own count:
    an axis `search` did not count, or one holding a single member, is no
    population. Every open axis no word covers has to hold exactly one
    member. One holding several, or an uncounted one, is a party the
    requirement neither names nor states, and every row of the population
    would repeat once per member of it. `open_entity_axes` of `None`, a
    source whose axes were never read, matches nothing.

    Returns:
        The covered `OpenEntityAxis` of each word, or the reason no set of axes
        matches, as a sentence a finding can carry.
    """
    parties: list[Party] = []
    for word in across:
        party = across_party(word)
        if party is None:
            return _not_a_party_reason(word)
        parties.append(party)
    if open_entity_axes is None:
        return (
            "the source was recorded without the axes it leaves open, so whether it holds "
            f"every {_quoted(across)} is unknown; discover the source again"
        )
    covered: dict[str, OpenEntityAxis] = {}
    for word, party in zip(across, parties, strict=True):
        if party == "place":
            continue
        matching = [axis for axis in open_entity_axes if _axis_party(axis.column) == party]
        if len(matching) != 1:
            return _off_party_reason(word, party, open_entity_axes)
        covered[word] = matching[0]
    for word, party in zip(across, parties, strict=True):
        if party == "place":
            place = _place_axis(word, open_entity_axes, taken=covered.values())
            if isinstance(place, str):
                return place
            covered[word] = place
    axes = tuple(covered[word] for word in across)
    return _no_population_reason(across, axes, open_entity_axes) or axes


def _place_axis(
    word: str, open_entity_axes: Sequence[OpenEntityAxis], *, taken: Iterable[OpenEntityAxis]
) -> OpenEntityAxis | str:
    """The open axis the place `word` covers (`open_axes_for`), or why no one axis is it.

    `observationAbout`, open over several members, holds the places a measure
    is about, at every level. Without it the word names the open party of a
    flow, the one flow axis open over several members that no other word
    covers, as IDS holds its debtors on `destination`. That axis keeps its
    flow role, and every member it holds stays in, organisations included.
    A CRS donor axis covered by "country" holds the agencies beside the
    donor countries. The retrieval serves the axis without its group totals
    (`get_data`'s `without_party_totals`), and the data's residuals stay in.
    Two such flow axes leave the word naming neither, and the
    reason asks for each party to be stated.
    """
    if not open_entity_axes:
        return f"the source leaves no axis open, so it holds no population of every {word!r}"
    place = next((axis for axis in open_entity_axes if _axis_party(axis.column) == "place"), None)
    if place is not None and place.member_count != 1:
        return place
    taken_columns = {axis.column for axis in taken}
    flows = [
        axis
        for axis in open_entity_axes
        if axis.member_count != 1
        and axis.column not in taken_columns
        and _axis_party(axis.column) is not None
    ]
    if len(flows) == 1:
        return flows[0]
    columns = ", ".join(repr(axis.column) for axis in open_entity_axes)
    if flows:
        return (
            f"the source leaves {columns} open, each over several members, so {word!r} could "
            "be any of them; state each party the requirement covers in `across`, such as "
            "'donor' and 'recipient'"
        )
    # A place axis holding one place is named, so the population check says why it is none.
    if place is not None:
        return place
    return (
        f"the source leaves {columns} open, and none of them holds several places a "
        f"requirement over every {word!r} could cover"
    )


def _no_population_reason(
    across: Sequence[str],
    axes: Sequence[OpenEntityAxis],
    open_entity_axes: Sequence[OpenEntityAxis],
) -> str | None:
    """Why the open `axes` the `across` words cover hold no population of every member, or `None`."""
    covered = {axis.column for axis in axes}
    if extra := [
        axis for axis in open_entity_axes if axis.column not in covered and axis.member_count != 1
    ]:
        columns = ", ".join(repr(axis.column) for axis in extra)
        return (
            f"the source also leaves {columns} open over several members, a party `across` "
            f"does not state; add its party word to `across` to cover every member of it too, "
            f"or name the members the answer needs in `entities`, so each {_quoted(across)} is "
            "not repeated once per member of it"
        )
    for word, axis in zip(across, axes, strict=True):
        if axis.member_count is None:
            return (
                f"the source does not count the members of its open axis {axis.column!r}, so "
                f"it cannot be shown to hold every {word!r}; name the entities the answer needs "
                "in `entities` instead of stating that party in `across`"
            )
        if axis.member_count < _POPULATION_MINIMUM:
            return (
                f"the source holds {axis.member_count} "
                f"member{'' if axis.member_count == 1 else 's'} on its open axis "
                f"{axis.column!r}, which is no population of every {word!r}; name the entities "
                "the answer needs in `entities` instead of stating that party in `across`"
            )
    return None


def _quoted(across: Sequence[str]) -> str:
    """The `across` words for a sentence: "'donor'" or "'donor' and 'recipient'"."""
    quoted = [repr(word) for word in across]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def _axis_party(column: str) -> Party | None:
    """The party an open column holds: a side of a flow, the place for `observationAbout`, or `None`."""
    return flow_side(column) or ("place" if column == _PLACE_COLUMN else None)


def _not_a_party_reason(across: str) -> str:
    """Why the `across` word names no party an open axis holds whole."""
    party_word = place_qualified_party(across)
    if party_word is not None:
        return (
            f"{across!r} narrows a party to its countries, and a source's open axis holds every "
            f"{party_word}, agencies included; state `across` as {pluralize(party_word)!r} to "
            "cover them all, or name the countries in `entities`"
        )
    return (
        f"{across!r} is not a party a source's open axis holds: an open axis ranks every "
        "member it holds, and each `across` word states that whole party, such as "
        "'donor', 'recipient' or 'country'; a narrower population (a region, an income "
        "group, a donor type) needs its members named in `entities`, or a filter the plan "
        "can express"
    )


def _off_party_reason(across: str, party: Party, open_entity_axes: Sequence[OpenEntityAxis]) -> str:
    """Why the `across` word, naming the `party` side of a flow, matches no one open axis of the source."""
    if not open_entity_axes:
        return f"the source leaves no axis open, so it holds no population of every {across!r}"
    if len(open_entity_axes) > 1:
        columns = ", ".join(repr(axis.column) for axis in open_entity_axes)
        return (
            f"the source leaves {columns} open, and exactly one of them has to hold the "
            f"{party} side of the flow {across!r} names"
        )
    column = open_entity_axes[0].column
    column_party = _axis_party(column)
    if column_party is None:
        return (
            f"the source's one open axis, {column!r}, is a column no party word names, so it "
            f"cannot be read as holding every {across!r}"
        )
    if column_party == "place":
        return (
            f"{across!r} names the {party} side of a flow, and the source's one open axis, "
            f"{column!r}, is no side of a flow; state `across` as the place noun, such as "
            "'country'"
        )
    return (
        f"{across!r} names the other side of the flow from the source's one open axis, {column!r}"
    )


def _resolved_by_ref(
    entity: Entity, evidence: RetrievalEvidence, approved: SelectedSource | None
) -> ResolvedEntity | None:
    """The entity `entity` names by ref when no label of `evidence` matches it, or `None`.

    At discovery the source's own refs answer (`resolve_name`). A
    manifest names only refs, so at run time `entity` is resolved through
    the approved source, by label or by ref, and the retrieval has to hold
    that ref.
    """
    if evidence.origin == "discovery":
        return resolve_name(entity, evidence)
    resolved = resolve_name(entity, approved) if approved is not None else None
    if resolved is None or all(one.ref != resolved.ref for one in evidence.entities):
        return None
    return resolved


def _named_group(
    entity: Entity, evidence: RetrievalEvidence, approved: SelectedSource | None
) -> ResolvedGroup | None:
    """The named set `entity` names (`resolve_group`), or `None`.

    At discovery the source's own `groups` answer. A manifest states none,
    so at run time the set is the approved source's.
    """
    if evidence.origin == "discovery":
        return resolve_group(entity, evidence.groups)
    return resolve_group(entity, approved.groups) if approved is not None else None


def _group_present(
    entity: Entity, evidence: RetrievalEvidence, approved: SelectedSource | None
) -> bool:
    """Whether the retrieval holds the set `entity` names (`_named_group`), as its members or its published total.

    Called once `_named_group` found the set. At discovery naming one of the
    source's sets is enough. A manifest names only refs, so at run time the
    retrieval has to hold at least one of the entities the approved source
    resolved on the set's column (`named_members`).
    """
    if evidence.origin == "discovery":
        return True
    assert approved is not None  # `_named_group` finds no set without it at run time
    members = {one.ref for one in named_members(entity, approved)}
    return any(one.ref in members for one in evidence.entities)


def _group_reading_issues(
    requirement: DatasetRequirement, entity: Entity, group: ResolvedGroup
) -> list[EvidenceIssue]:
    """The issues, if any, with how the source binds `group`, the set `entity` names, against what it asks for.

    An entity with `members` set asks for each member, so a set the source
    binds as its published total, or as the totals of a whole population
    (`ResolvedGroup.reads_total`), does not answer it. One without accepts
    either reading.
    """
    if not entity.members or not group.reads_total:
        return []
    bound = (
        f"set's published total, {group.total!r}"
        if group.total is not None
        else f"published totals of {group.label!r}"
    )
    return [
        EvidenceIssue(
            code=EvidenceIssueCode.GROUP_READ_AS_TOTAL,
            severity=Severity.ERROR,
            requirement_id=requirement.id,
            detail=(
                f"{entity.name!r} is asked for member by member, but the source binds the "
                f"{bound}, and no member"
            ),
            found=group.total if group.total is not None else group.label,
            required=entity.name,
        )
    ]


def _entity_issues(
    requirement: DatasetRequirement,
    evidence: RetrievalEvidence,
    approved: SelectedSource | None = None,
) -> list[EvidenceIssue]:
    """The findings on the entities `requirement` names and the population it leaves open."""
    issues = _named_entity_issues(requirement, evidence, approved) if requirement.entities else []
    if requirement.across:
        issues.extend(_open_axis_issues(requirement, requirement.across, evidence, approved))
    else:
        issues.extend(_unstated_place_issues(requirement, evidence, approved))
    return issues


def opens_place_axis(open_entity_axes: Sequence[OpenEntityAxis] | None) -> bool:
    """Whether a source leaves its place axis open over more than one place.

    Such a retrieval holds every place with data at every level, the world,
    regions, countries and their subdivisions, so a rank, a total or a share
    over it keeps one level first (`planner.shapes`). The place axis is the
    one `observationAbout` holds, the column `open_axes_for` reads as the
    place party. One member is a single place, such as Earth for a worldwide
    total, and an uncounted axis may hold every place.
    """
    return any(
        _axis_party(axis.column) == "place" and axis.member_count != 1
        for axis in open_entity_axes or ()
    )


def _unstated_place_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence, approved: SelectedSource | None
) -> list[EvidenceIssue]:
    """Whether a source leaves the place open under a requirement whose `across` does not say so.

    `across` naming the place party is how a plan states that its retrieval
    holds every level of place, which the plan's level rule, the compiler's
    `place_types` request and the card all read, so a source that
    `opens_place_axis` under a requirement with no `across` is refused.
    """
    open_entity_axes = (
        approved.open_entity_axes if approved is not None else evidence.open_entity_axes
    )
    if not opens_place_axis(open_entity_axes):
        return []
    return [
        EvidenceIssue(
            code=EvidenceIssueCode.OPEN_AXIS_UNRESOLVED,
            severity=Severity.ERROR,
            requirement_id=requirement.id,
            detail=(
                f"source {evidence.spec_id} leaves its place axis {_PLACE_COLUMN!r} open, so "
                "its retrieval holds every place with data at every level, and the requirement "
                "states no `across`; state the place party, such as 'country', so the plan keeps "
                "one level before a rank or a total, or name the places in `entities`"
            ),
            found=[axis.column for axis in open_entity_axes or ()],
        )
    ]


def _open_axis_issues(
    requirement: DatasetRequirement,
    across: Sequence[str],
    evidence: RetrievalEvidence,
    approved: SelectedSource | None,
) -> list[EvidenceIssue]:
    """Whether the source holds every member of each population `across` names.

    At discovery the source's own open entity axes have to answer `across`
    (`open_axes_for`). A manifest states none, so at run time the approved
    source's answer it, and the retrieval has to hold at least two distinct
    entities under the column it carries each covered axis in
    (`_retrieved_column`), the fewest a population has. Each axis holding
    fewer is its own finding. The count is not compared with the axis's
    `member_count`: `search` counts members over every year, and the
    requirement's period may hold fewer.
    """
    open_entity_axes = (
        approved.open_entity_axes if approved is not None else evidence.open_entity_axes
    )
    axes = open_axes_for(across, open_entity_axes)
    if isinstance(axes, str):
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.OPEN_AXIS_UNRESOLVED,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=axes,
                found=[one.column for one in open_entity_axes or ()],
                required=list(across),
            )
        ]
    if evidence.origin == "discovery":
        return []
    issues: list[EvidenceIssue] = []
    for word, axis in zip(across, axes, strict=True):
        column = _retrieved_column(axis.column, evidence)
        found = sorted({one.ref for one in evidence.entities if one.role == column})
        if len(found) >= _POPULATION_MINIMUM:
            continue
        held = (
            "no entity" if not found else f"{len(found)} entit{'y' if len(found) == 1 else 'ies'}"
        )
        where = f"under the {column!r} column" if column else f"on the {axis.column!r} axis"
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.OPEN_AXIS_UNRESOLVED,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"the retrieval holds {held} {where}, which the plan was approved to read "
                    f"as every {word!r}; a population holds at least {_POPULATION_MINIMUM}"
                ),
                found=found,
                required=word,
            )
        )
    return issues


def _retrieved_column(column: str, evidence: RetrievalEvidence) -> str | None:
    """The role column of the retrieval carrying the approved source's open `column`, or `None`.

    A column the series statistics name binds to itself, as a place's
    `observationAbout` does. A flow axis (`origin`) binds to the one column
    the manifest's coverage dimensions place on it (`financingProvider`), by
    the rule the executor binds a step's columns with
    (`compiler.frame_columns.columns_on_axes`).
    """
    if any(entity.role == column for entity in evidence.entities):
        return column
    axes = evidence.axis_by_column or {}
    return columns_on_axes(axes, axes.keys()).get(column)


def _named_entity_issues(
    requirement: DatasetRequirement,
    evidence: RetrievalEvidence,
    approved: SelectedSource | None,
) -> list[EvidenceIssue]:
    """Whether the source resolves each entity `requirement` names, under the role it asks for.

    A name is met by a named set the source binds (`_group_present`), as
    its members or, unless the entity asks for its members, as its
    published total (`_group_reading_issues`); by a resolved entity's
    label; or by the ref the graph resolved the name to, or at run time the
    ref the approved source resolved it to (`_resolved_by_ref`).
    """
    if not evidence.entities:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.ENTITY_UNVERIFIED,
                severity=_coverage_severity(requirement),
                requirement_id=requirement.id,
                detail="the source resolves no entities, so none of the required ones can be confirmed",
                required=[entity.name for entity in requirement.entities],
            )
        ]
    labels = [entity.label for entity in evidence.entities]
    issues: list[EvidenceIssue] = []
    for entity in requirement.entities:
        group = _named_group(entity, evidence, approved)
        if group is not None and _group_present(entity, evidence, approved):
            issues.extend(_group_reading_issues(requirement, entity, group))
            continue
        matches = entity_matches(entity, evidence.entities)
        if not matches:
            if (by_ref := _resolved_by_ref(entity, evidence, approved)) is not None:
                if evidence.origin == "discovery":
                    issues.extend(_role_issues(requirement, entity, by_ref))
                continue
            issues.append(
                EvidenceIssue(
                    code=EvidenceIssueCode.ENTITY_MISSING,
                    severity=Severity.ERROR,
                    requirement_id=requirement.id,
                    detail=f"the source resolves no entity named {entity.name!r}",
                    found=labels,
                    required=entity.name,
                )
            )
            continue
        resolved = resolve_entity(entity, evidence.entities)
        if resolved is None:
            columns = sorted({one.role for one in matches})
            issues.append(
                EvidenceIssue(
                    code=EvidenceIssueCode.ENTITY_ROLE_AMBIGUOUS,
                    severity=Severity.ERROR,
                    requirement_id=requirement.id,
                    detail=(
                        f"{entity.name!r} resolves under more than one role ({', '.join(columns)}); "
                        + (
                            f"none matches the requirement's stated role {entity.role!r}"
                            if entity.role is not None
                            else "the requirement states no role to tell them apart"
                        )
                    ),
                    found=columns,
                    required=entity.role,
                )
            )
            continue
        issues.extend(_role_issues(requirement, entity, resolved))
    return issues


def _role_issues(
    requirement: DatasetRequirement, entity: Entity, resolved: ResolvedEntity
) -> list[EvidenceIssue]:
    """Whether the role column `entity` resolved under answers the role it asked for.

    Compares sides of the flow, not words: `debtor` against a `recipient`
    column is the same party and stays silent, `creditor` against `debtor`
    is an error. A phrase or a column the lexicon cannot place is unverified
    rather than fine -- silence was the bug this check exists to close. The exact check
    needs `search` to echo which requested entity each resolved entity
    answers, which is filed as issue #196.
    """
    if entity.role is None or entity.role.casefold() == resolved.role.casefold():
        return []
    if _roles_conflict(entity.role, resolved.role):
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.ENTITY_ROLE_MISMATCH,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"{entity.name!r} is asked for as {entity.role!r} but resolves under the "
                    f"{resolved.role!r} column, the other side of the flow"
                ),
                found=resolved.role,
                required=entity.role,
            )
        ]
    unplaced = [word for word in (entity.role, resolved.role) if flow_side(word) is None]
    if unplaced:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.ENTITY_ROLE_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id=requirement.id,
                detail=(
                    f"{entity.name!r} is asked for as {entity.role!r} and resolves under the "
                    f"{resolved.role!r} column; the check cannot place "
                    f"{' or '.join(repr(word) for word in unplaced)} on either side of the "
                    "flow, so the column cannot be confirmed to answer the phrase"
                ),
                found=resolved.role,
                required=entity.role,
            )
        ]
    return []


# --- Period -------------------------------------------------------------------


def _period_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence
) -> list[EvidenceIssue]:
    period = requirement.period
    if evidence.series is not None and evidence.series.num_rows:
        # Non-empty per-series statistics: each series is checked against the
        # period on its own, in `_series_issues`, since the union of their
        # windows can hide one entity's shortfall behind another's reach. A
        # manifest whose statistics are `None` or `[]` falls through instead
        # -- `_series_issues` reports that as `STATISTICS_UNAVAILABLE`, and
        # this still checks whatever coverage window the evidence states.
        return []
    if evidence.window is None:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.PERIOD_UNVERIFIED,
                severity=_coverage_severity(requirement),
                requirement_id=requirement.id,
                detail="the source reports no coverage window",
                required=period.model_dump(),
            )
        ]
    if not evidence.window.covers(period):
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.WINDOW_SHORT,
                severity=_coverage_severity(requirement),
                requirement_id=requirement.id,
                detail=(
                    f"the source covers {evidence.window.start}-{evidence.window.end}, "
                    f"the requirement asks for {period.start}-{period.end}"
                ),
                found=evidence.window.model_dump(),
                required=period.model_dump(),
            )
        ]
    return []


# --- Scope: what the definition itself binds ---------------------------------


def _scope_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence
) -> list[EvidenceIssue]:
    """Compare the identity document's own scope with the requirement.

    The identity is what `spec_id` hashes, so what it binds is what a
    retrieval against that spec can return. Only what it actually states is
    compared: a member it omits says nothing.
    """
    if not evidence.identity:
        return []
    issues: list[EvidenceIssue] = []
    bound = _identity_refs(evidence.identity.get("entities"))
    if bound is not None:
        named = named_refs(requirement.entities, evidence)
        issues.extend(_scope_entity_issues(requirement, named, bound))
    requested = evidence.identity.get("requested_window")
    if isinstance(requested, dict):
        issues.extend(_scope_window_issues(requirement, requested))
    return issues


def _identity_refs(entities: JsonValue) -> set[str] | None:
    """The refs an identity's `entities` binds, across every role; `None` when it states none.

    The identity's keys are graph properties, not the source's role columns,
    so only the refs are compared and never the roles. An empty map, or one
    with no list under any key, states none exactly as an absent member
    does: data-mcp writes `entities: {}` to clear the filter, which is an
    unfiltered definition wider than the requirement, not one binding nothing.
    """
    if not isinstance(entities, dict):
        return None
    lists = [values for values in entities.values() if isinstance(values, list)]
    if not lists:
        return None
    return {value for values in lists for value in values if isinstance(value, str)}


def _scope_entity_issues(
    requirement: DatasetRequirement, wanted: set[str], bound: set[str]
) -> list[EvidenceIssue]:
    # A required entity is known by name; the refs it binds to are the ones
    # it names (`named_members`), a set's members included. One naming none
    # is already reported as missing or ambiguous, and has no ref to compare.
    if not wanted:
        return []
    issues: list[EvidenceIssue] = []
    if unbound := wanted - bound:
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.SCOPE_ENTITIES_UNBOUND,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"the definition binds {sorted(bound)!r}, which leaves out "
                    f"{sorted(unbound)!r} that the requirement names"
                ),
                found=sorted(bound),
                required=sorted(wanted),
            )
        )
    if wider := bound - wanted:
        # A region name legitimately expands into its members, so a definition
        # bound wider than the requirement is recorded, not refused.
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.SCOPE_ENTITIES_WIDER,
                severity=Severity.WARNING,
                requirement_id=requirement.id,
                detail=(
                    f"the definition also binds {sorted(wider)!r}, "
                    "which the requirement does not name"
                ),
                found=sorted(bound),
                required=sorted(wanted),
            )
        )
    return issues


def _scope_window_issues(
    requirement: DatasetRequirement, requested: dict[str, JsonValue]
) -> list[EvidenceIssue]:
    period = requirement.period
    kind = requested.get("kind")
    if kind == "latest":
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.SCOPE_WINDOW_SHORT,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    "the definition requests only the latest observation, "
                    f"which can never cover {period.start}-{period.end}"
                ),
                found=requested,
                required=period.model_dump(),
            )
        ]
    if kind != "range":
        return []
    start, end = requested.get("start_year"), requested.get("end_year")
    # A range with no end year holds every year from its start on.
    if not isinstance(start, int) or not (end is None or (isinstance(end, int) and end >= start)):
        return []
    if start <= period.start and (end is None or period.end <= end):
        return []
    disjoint = period.end < start or (end is not None and end < period.start)
    asked = f"{start} on" if end is None else f"{start}-{end}"
    # A range that shares no year with the period can never cover it; one
    # that merely falls short is a gap the requirement's coverage decides.
    return [
        EvidenceIssue(
            code=EvidenceIssueCode.SCOPE_WINDOW_SHORT,
            severity=Severity.ERROR if disjoint else _coverage_severity(requirement),
            requirement_id=requirement.id,
            detail=(
                f"the definition requests {asked}, "
                f"the requirement asks for {period.start}-{period.end}"
            ),
            found=requested,
            required=period.model_dump(),
        )
    ]


# --- Unit ---------------------------------------------------------------------

# Common currency symbols mapped to their ISO-4217 code before the currency
# search runs. Longest symbol first, so "US$" is consumed whole and does not
# leave a stray "$" for the single-character mapping to double-substitute.
_CURRENCY_SYMBOLS: dict[str, str] = {"US$": "USD", "$": "USD", "€": "EUR", "£": "GBP"}

# The ISO-4217 codes a source's unit may name. A closed set rather than any
# three-letter word, so the "per" in "US dollars per capita" is not read as a
# currency; extend it when a source reports in another.
_KNOWN_CURRENCIES = frozenset({"USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD", "XDR"})

# The unit ids that name a currency structurally, read before any text
# heuristic: Data Commons' `Currency<ISO>` (any ISO-4217 code, so the closed
# set above does not apply), its `USDollar`, this deployment's
# constant-price `Constant<ISO>_<year>`, and the SDG/OECD
# `SDG_CU_<ISO>`/`SDG_CON[_<year>]_<ISO>` family with an optional magnitude
# suffix. The first three are one letter run each, so the letter scan below
# cannot see the code inside them, and the scan reads no basis from `CU` or
# `CON`.
_CURRENCY_DCID = re.compile(r"Currency(?P<currency>[A-Z]{3})")
_CONSTANT_PRICE_DCID = re.compile(r"Constant(?P<currency>[A-Z]{3})_(?P<year>\d{4})")
_SDG_PRICE_BASIS_ID = re.compile(
    r"SDG_(?:(?P<current>CU)|CON(?:_(?P<year>\d{4}))?)_(?P<currency>[A-Z]{3})(?:_[KMB])?"
)
_NAMED_CURRENCY_DCIDS = {"USDollar": "USD"}

# What a unit in international dollars reads as. Purchasing-power-parity
# dollars carry no ISO-4217 code and are not US dollars, though the UN's id
# for them names USD (`undata/UNIT_MEASURE-CR_USD_PPP`), so a `PPP` segment
# of a unit id, or Data Commons' `InternationalDollar`, reads as this rather
# than as the currency beside it. A unit's text reads the same way when it
# says "PPP", "purchasing power parity" or "international dollars", whatever
# symbol it carries, before any symbol maps to a currency.
_PPP = "PPP"
_PPP_DCIDS = frozenset({"InternationalDollar"})
_PPP_TEXT = re.compile(
    r"\bPPP\b|purchasing[\s-]+power[\s-]+parity|international\s+(?:dollars?|\$)", re.IGNORECASE
)

# A run of letters, in either case. Not `\b`-delimited words: `_` is a word
# character, so a word boundary cannot see the "USD" in "SDG_CU_USD_M". The
# closed currency set above is what keeps a bare letter scan safe.
_LETTER_RUNS = re.compile(r"[A-Za-z]+")

# The words a unit's text uses for each price basis, and the shape of a base
# year. `PriceBasis` is `Unit.prices`' own vocabulary.
PriceBasis = Literal["current", "constant"]
_CONSTANT_WORDS = frozenset({"constant", "real", "deflated"})
_CURRENT_WORDS = frozenset({"current", "nominal"})
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


class UnitReading(NamedTuple):
    """What a source's unit states: a currency, a price basis, and a base year,
    each `None` when the unit does not state it."""

    currency: str | None = None
    prices: PriceBasis | None = None
    base_year: int | None = None


def _read_unit(evidence: RetrievalEvidence, described: str) -> UnitReading:
    """Read the unit `evidence` states: a field-by-field unit as it stands, a
    unit dcid by its structure, and whatever either leaves unstated from the
    unit's text."""
    if (stated := evidence.stated_unit) is not None:
        return UnitReading(stated.currency, stated.prices, stated.base_year)
    by_ref = read_unit_dcid(evidence.unit_ref)
    prices, base_year = (
        (by_ref.prices, by_ref.base_year)
        if by_ref.prices is not None
        else _detect_price_basis(described)
    )
    currency = by_ref.currency or (
        _PPP
        if _PPP_TEXT.search(described)
        else _detect_currency(evidence.unit_ref, evidence.unit_label)
    )
    return UnitReading(currency, prices, base_year)


def read_unit_dcid(ref: str | None) -> UnitReading:
    """The currency and price basis a unit id's own structure names."""
    if ref is None:
        return UnitReading()
    dcid = ref.removeprefix("dcid:")
    if dcid in _PPP_DCIDS or _PPP in _LETTER_RUNS.findall(dcid):
        return UnitReading(currency=_PPP)
    if match := _CURRENCY_DCID.fullmatch(dcid):
        return UnitReading(currency=match["currency"])
    if match := _CONSTANT_PRICE_DCID.fullmatch(dcid):
        return UnitReading(match["currency"], "constant", int(match["year"]))
    if match := _SDG_PRICE_BASIS_ID.fullmatch(dcid):
        return _sdg_unit_reading(match)
    return UnitReading(currency=_NAMED_CURRENCY_DCIDS.get(dcid))


def read_unit_currency(ref: str | None, label: str | None) -> str | None:
    """The currency a unit names, `None` when it names none.

    Read the way `check_evidence` reads a source's currency: by the unit id's
    structure (`read_unit_dcid`), then `PPP` when `label` says purchasing
    power parity, then the first known ISO-4217 code in the id or `label`.
    """
    return read_unit_dcid(ref).currency or (
        _PPP if label and _PPP_TEXT.search(label) else _detect_currency(ref, label)
    )


def _sdg_unit_reading(match: re.Match[str]) -> UnitReading:
    """The currency and price basis an `_SDG_PRICE_BASIS_ID` match names."""
    if match["current"]:
        return UnitReading(match["currency"], "current")
    year = match["year"]
    return UnitReading(match["currency"], "constant", int(year) if year else None)


def _detect_currency(*parts: str | None) -> str | None:
    """The first known ISO-4217 currency code named in `parts`, if any."""
    text = " ".join(part for part in parts if part)
    for symbol in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True):
        text = text.replace(symbol, f" {_CURRENCY_SYMBOLS[symbol]} ")
    for token in _LETTER_RUNS.findall(text):
        if token.upper() in _KNOWN_CURRENCIES:
            return token.upper()
    return None


def _detect_price_basis(text: str) -> tuple[PriceBasis | None, int | None]:
    """The price basis and base year a unit's text states, either `None` when it does not.

    A text heuristic over the unit's words: a text naming both bases states
    neither. The base year is missing from the search surface entirely
    (issue #193), so it is read here only when the text happens to carry it.
    """
    words = {token.casefold() for token in _LETTER_RUNS.findall(text)}
    constant, current = bool(words & _CONSTANT_WORDS), bool(words & _CURRENT_WORDS)
    basis: PriceBasis | None = None
    if constant != current:
        basis = "constant" if constant else "current"
    years = _YEAR.findall(text)
    return basis, int(years[0]) if len(years) == 1 else None


def read_source_unit(source: SelectedSource) -> UnitReading:
    """What `source`'s own unit states, read as `check_evidence` reads it: by the
    unit id's structure, then by the unit's text for whatever that leaves unstated."""
    evidence = evidence_from_source(source)
    return _read_unit(evidence, _describe_unit(evidence))


def _unit_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence
) -> list[EvidenceIssue]:
    """The findings on whether the unit `evidence` states is the one `requirement` asks for,
    checking its currency, then its price basis. Empty when the requirement states no unit."""
    unit = requirement.unit
    if unit is None:
        return []
    described = _describe_unit(evidence)
    currency, basis, base_year = _read_unit(evidence, described)
    issues: list[EvidenceIssue] = []
    if currency == unit.currency:
        stated = f" in {basis} prices" if basis is not None else ""
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.UNIT_MATCHES,
                severity=Severity.INFO,
                requirement_id=requirement.id,
                detail=f"the source's unit {described!r} names {unit.currency}{stated}",
                found=described,
                required=unit.model_dump(),
            )
        )
    elif currency is not None:
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.UNIT_MISMATCH,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"the source's unit {described!r} names {currency}, "
                    f"the requirement asks for {unit.currency}"
                ),
                found=currency,
                required=unit.currency,
            )
        )
    else:
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.UNIT_UNVERIFIED,
                # At manifest origin the manifest is the record of what was
                # actually retrieved, so an unrecognisable or absent unit there
                # is not merely unconfirmed; at discovery it is only a note for
                # the approval summary.
                severity=Severity.ERROR if evidence.origin == "manifest" else Severity.WARNING,
                requirement_id=requirement.id,
                detail=(
                    f"the source's unit {described!r} names no recognisable currency"
                    if described
                    else "the source states no unit"
                ),
                found=described or None,
                required=unit.model_dump(),
            )
        )
    issues.extend(_price_basis_issues(requirement, unit, described, basis, base_year))
    return issues


def _describe_unit(evidence: RetrievalEvidence) -> str:
    """The unit `evidence` states, as the issues quote it."""
    if (stated := evidence.stated_unit) is not None:
        return price_basis_reading(
            currency=stated.currency, prices=stated.prices, base_year=stated.base_year
        )
    return " ".join(part for part in (evidence.unit_ref, evidence.unit_label) if part)


def _price_basis_issues(
    requirement: DatasetRequirement,
    unit: Unit,
    described: str,
    basis: PriceBasis | None,
    base_year: int | None,
) -> list[EvidenceIssue]:
    """Whether the price basis the unit's text states is the one required.

    A basis the text does not state is a warning, not an error: the base
    year is missing from the search surface (issue #193), so an error here
    would block every constant-price plan. Today only the current-price arm
    runs -- `_check_requirements_meetable` refuses every constant-price
    requirement outright -- and the detection is a text heuristic.
    """
    if basis is None:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id=requirement.id,
                detail=(
                    f"the source's unit {described!r} does not state whether its prices are "
                    f"current or constant; the requirement asks for {unit.prices} prices"
                ),
                found=described or None,
                required=unit.model_dump(),
            )
        ]
    if basis != unit.prices:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.UNIT_PRICE_BASIS_MISMATCH,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"the source's unit {described!r} states {basis} prices, "
                    f"the requirement asks for {unit.prices} prices"
                ),
                found=basis,
                required=unit.prices,
            )
        ]
    if unit.prices == "current":
        return []
    if base_year is None:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id=requirement.id,
                detail=(
                    f"the source's unit {described!r} states constant prices but no base year; "
                    f"the requirement asks for constant {unit.base_year} prices"
                ),
                found=described,
                required=unit.model_dump(),
            )
        ]
    if base_year != unit.base_year:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.UNIT_PRICE_BASIS_MISMATCH,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"the source's unit {described!r} states constant {base_year} prices, "
                    f"the requirement asks for constant {unit.base_year} prices"
                ),
                found=base_year,
                required=unit.base_year,
            )
        ]
    return []


# --- Source -------------------------------------------------------------------


_WORD_CHAR = re.compile(r"[^\W_]", re.UNICODE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Unicode general categories for a combining mark: nonspacing (Mn), spacing
# combining (Mc), and enclosing (Me). Python's `re` has no `\p{M}` class, so
# `_words` below scans `unicodedata.category` directly rather than folding
# marks into a regex.
_COMBINING_CATEGORIES = frozenset({"Mn", "Mc", "Me"})


def _is_latin_base(char: str) -> bool:
    """Whether `char` is a Latin letter, by its Unicode name.

    A single character with no name (an unassigned code point, or one that
    slipped in as an empty string when a combining mark opens `text` with no
    base before it) is not Latin.
    """
    if not char:
        return False
    try:
        return unicodedata.name(char).startswith("LATIN")
    except ValueError:
        return False


def _fold_accents(text: str) -> str:
    """`text` with each Latin accented letter reduced to its base letter.

    NFKD decomposes an accented letter into the base letter plus one or more
    combining marks; dropping marks of category Mn ("Cesky" from "Český")
    lets a requirement typed with diacritics match a source spelled without
    them, or the other way round. A mark is only dropped when the base
    character it decomposed from is Latin: the same decomposition also
    separates a Japanese kana from its voiced-sound mark ("か" from "が"), a
    Devanagari consonant from its nukta ("क" from "क़"), and other scripts'
    marks from their base letters, and those pairs name different characters
    once the mark is gone. What survives is renormalised to NFC, so a kept
    mark recomposes with its base into the precomposed form the rest of this
    module tokenizes.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    kept: list[str] = []
    base = ""
    for char in decomposed:
        if unicodedata.combining(char):
            if not _is_latin_base(base):
                kept.append(char)
            continue
        base = char
        kept.append(char)
    return unicodedata.normalize("NFC", "".join(kept))


def _words(text: str) -> list[str]:
    """`text` split into maximal runs of word characters, each carrying whatever
    combining marks (Mn, Mc, Me) are attached to it.

    A composition exclusion keeps some canonically-decomposed letters (a
    Devanagari consonant plus its nukta, among others) from ever recomposing
    under NFC, so `_fold_accents`'s output can still carry a mark as its own
    code point. `[^\\W_]+` alone does not see it: Python's `re` has no
    `\\p{M}` class, and a bare combining mark is not itself a word character,
    so a naive word-character scan would drop it and merge two distinct
    letters into one token, exactly the collision folding a Latin accent is
    supposed to cause and folding anything else is not. A mark can only
    extend a run already open; one with no preceding word character in this
    text is orphaned, and dropped, as it always was.
    """
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if _WORD_CHAR.fullmatch(char) is not None or (
            current and unicodedata.category(char) in _COMBINING_CATEGORIES
        ):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _tokens(text: str) -> tuple[str, ...]:
    """`text` split into lowercase word tokens, on every run of
    non-word characters, every camelCase boundary, and every
    acronym-to-word boundary.

    A source is spelled differently on the two sides: a requirement names
    "World Development Indicators" while the provenance serving it spells the
    same name "WorldDevelopmentIndicators", and a ref prefixes it further
    ("dc/base/WorldDevelopmentIndicators"). Splitting on word boundaries
    (non-word-character runs, a lower-to-upper transition inside one word,
    and an upper-to-upper-then-lower transition such as "UNICEFData") reads
    "WorldDevelopmentIndicators" as `("world", "development",
    "indicators")`, the same tokens spaces would have given, without
    collapsing the name to one blob a short, unrelated word could contain a
    substring of. Matching is Unicode-aware in both directions: Latin accents
    are folded before splitting, so "Český" and "Cesky" tokenize the same,
    while a mark on a non-Latin base is left alone (recomposed with it into
    one code point by NFC where composition allows, kept as base-plus-mark by
    `_words` where it does not), so "が" and "か", or "क़" and "क", keep
    tokenizing as the distinct characters they are. The word boundary itself
    is a Unicode word character, not `[a-zA-Z0-9]`, so a name in a non-Latin
    script still splits on punctuation and case rather than vanishing letter
    by letter.
    """
    folded = _fold_accents(text)
    tokens = [piece.casefold() for word in _words(folded) for piece in _CAMEL_BOUNDARY.split(word)]
    return tuple(token for token in tokens if token)


def _contains_run(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    """Whether `needle`'s tokens appear contiguously, in order, inside `haystack`."""
    if not needle:
        return False
    width = len(needle)
    return any(
        haystack[start : start + width] == needle for start in range(len(haystack) - width + 1)
    )


def _mentions(field: str | None, wanted: str) -> bool:
    """Whether `field` and `wanted` name the same source, loosely.

    Either side's token run containing the other's counts, so a label naming
    the source and a table ("OECD DAC CRS table 2a") matches the source
    alone. Comparison works on whole tokens: a short source name like "IDA"
    matches the word "IDA", not the three letters that happen to fall
    consecutively inside "Guidance". An empty token run matches nothing: a
    name made only of punctuation, or an empty one, would otherwise be
    contained in every source.
    """
    if not field:
        return False
    left = _tokens(field)
    right = _tokens(wanted)
    if not left or not right:
        return False
    return _contains_run(left, right) or _contains_run(right, left)


def source_requirement_matches(*, requirement: DatasetRequirement, source: SelectedSource) -> bool:
    """Whether this source already confirms the requirement's source constraint and series code.

    A requirement stating a `source_code` is confirmed only by a source whose
    `exact_codes` hold it. A source that states no exact codes confirms none.
    """
    if requirement.source_code is not None and requirement.source_code not in source.exact_codes:
        return False
    return not _source_issues(requirement, evidence_from_source(source))


def _source_issues(
    requirement: DatasetRequirement,
    evidence: RetrievalEvidence,
    approved: SelectedSource | None = None,
) -> list[EvidenceIssue]:
    """`SOURCE_PIN_MISMATCH` when the evidence is not the definition the requirement pins.

    A pin names one exact definition, so a differing `spec_id` is an error.
    """
    issues: list[EvidenceIssue] = []
    if requirement.source_spec_id is not None and evidence.spec_id != requirement.source_spec_id:
        issues.append(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_PIN_MISMATCH,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"the requirement pins {requirement.source_spec_id!r}, "
                    f"the source found is {evidence.spec_id!r}"
                ),
                found=evidence.spec_id,
                required=requirement.source_spec_id,
            )
        )
    if requirement.source is not None:
        mentions = evidence.resolved_sources
        if approved is not None and evidence.spec_id == approved.spec_id:
            mentions = approved.resolved_sources
        # Search resolves aliases and collection names against its catalogue.
        # Only the constraints attached to this answer can confirm the ask.
        confirmed = source_constraint_matches(
            mentions,
            text=requirement.source,
            ref=requirement.source_ref,
            exclude=requirement.source_exclude,
        )
        if confirmed:
            return issues
        if requirement.source_exclude or requirement.source_ref is not None:
            return [
                *issues,
                EvidenceIssue(
                    code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                    severity=Severity.ERROR,
                    requirement_id=requirement.id,
                    detail="Search has not confirmed the requirement's source constraint",
                    required=requirement.source,
                ),
            ]
        known = evidence.provenance_ref is not None or evidence.provenance_label is not None
        if not known:
            issues.append(
                EvidenceIssue(
                    code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                    severity=Severity.WARNING,
                    requirement_id=requirement.id,
                    detail="the source states no provenance, so the requirement's stated source cannot be confirmed",
                    required=requirement.source,
                )
            )
        else:
            named = _mentions(evidence.provenance_ref, requirement.source) or _mentions(
                evidence.provenance_label, requirement.source
            )
            if not named:
                found = " ".join(
                    part for part in (evidence.provenance_ref, evidence.provenance_label) if part
                )
                issues.append(
                    EvidenceIssue(
                        code=EvidenceIssueCode.SOURCE_MISMATCH,
                        severity=Severity.ERROR,
                        requirement_id=requirement.id,
                        detail=(
                            f"the requirement states source {requirement.source!r}, "
                            f"the source found names {found!r}"
                        ),
                        found=found,
                        required=requirement.source,
                    )
                )
    return issues


# --- Run time: what was retrieved against what was approved -------------------


def _drift_issues(
    requirement: DatasetRequirement,
    evidence: RetrievalEvidence,
    approved: SelectedSource,
) -> list[EvidenceIssue]:
    """`SOURCE_DRIFT` when the manifest's spec_id or identity differs from what was approved.

    Either alone is enough: a manifest can carry the approved identity under
    a different spec_id -- or vice versa -- and both mean this is not the
    retrieval the plan was approved to run. Identities are compared only
    when both sides hold one. An approved source that held no identity has
    nothing to drift from, so that check is unverified rather than passed; a
    manifest that carries none is left to the identity check at discovery,
    which already refused the source before approval.

    When the description held stable -- same spec_id, same identity -- this
    also checks `STAT_VAR_MEMBERSHIP_DRIFT`: whether the definition's binding
    under that description held stable too. A description that drifted
    already means "not the retrieval that was approved" on its own, so a
    membership finding on top of it would say nothing new; membership only
    runs once `SOURCE_DRIFT` has not fired. Membership is checked whether or
    not an identity was approved: the binding comparison itself needs no identity, only the
    pinned arm does, and that arm already falls back to the binding
    comparison when a shape is absent or unrecognised. Skipping the check on
    an unknown identity would leave the case with the least protection
    getting the least checking -- exactly where a swapped series would
    otherwise pass unnoticed.
    """
    spec_id_drifted = evidence.spec_id != approved.spec_id
    identity_drifted = (
        approved.identity is not None
        and evidence.identity is not None
        and evidence.identity != approved.identity
    )
    if spec_id_drifted or identity_drifted:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_DRIFT,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail=(
                    f"the retrieval answers as {evidence.spec_id}, not the approved {approved.spec_id}"
                    if spec_id_drifted
                    else f"source {evidence.spec_id} no longer has the identity the plan was approved with"
                ),
                found={"spec_id": evidence.spec_id, "identity": evidence.identity},
                required={"spec_id": approved.spec_id, "identity": approved.identity},
            )
        ]
    if approved.identity is None:
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.IDENTITY_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id=requirement.id,
                detail=(
                    f"source {evidence.spec_id} was approved without an identity document, "
                    "so whether it has drifted since cannot be checked"
                ),
                found=evidence.identity,
            ),
            *_membership_issues(requirement, evidence, approved),
        ]
    return _membership_issues(requirement, evidence, approved)


def _pinned_stat_var(approved: SelectedSource) -> set[str] | None:
    """The single dcid `approved.identity`'s `shape` pins (`compiler.sources.pinned_stat_var`), bare.

    A pinned identity names its one dcid directly, with no envelope to
    consult and nothing that could be missing from it, so the binding it
    supplies is always known and always complete. An identity of any other
    shape, and one with no `shape` at all, falls through to the digest.
    """
    dcid = pinned_stat_var(approved.identity)
    return {_bare_dcid(dcid)} if dcid is not None else None


def _membership_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence, approved: SelectedSource
) -> list[EvidenceIssue]:
    """`STAT_VAR_MEMBERSHIP_DRIFT` when the definition's binding has moved under a stable identity.

    `SOURCE_DRIFT` holds the definition's *description* stable; this holds
    its *binding* under that description stable. What the retrieval is held
    to depends on what the approval knows of its own binding:

    - A pinned identity (`_pinned_stat_var`) names its one dcid, so the
      retrieved binding, `evidence.resolved_stat_vars` off the manifest's
      `trace`, has to be exactly that dcid; anything else is an ERROR.
    - Otherwise the family's digest decides (`_digest_issues`), and a source
      approved as one series (`served_count` of one) has to be served as one
      StatVar (`_single_series_issues`).

    `evidence.stat_vars`, the coverage set, is never compared: it is silent
    about a StatVar the definition binds that carries no observations for
    the entities and years asked.
    """
    pinned = _pinned_stat_var(approved)
    if pinned is not None:
        return _binding_comparison_issues(requirement, evidence, pinned)
    return [
        *_digest_issues(requirement, evidence, approved),
        *_single_series_issues(requirement, evidence, approved),
    ]


def _digest_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence, approved: SelectedSource
) -> list[EvidenceIssue]:
    """The approved family's digest against the one the mint's trace states.

    Both are computed the same way over the same population, every member the
    bindings admit (`compiler.sources.member_digest`), so a different digest
    is drift, an ERROR, and an equal one confirms the family. A side that
    states none (`search` states none for a walk it could not finish, and the
    mint none for a family it could not state whole) leaves membership
    unverified, a WARNING.
    """
    required: JsonValue = {
        "member_count": approved.shape_member_count,
        "member_digest": approved.shape_member_digest,
    }
    found: JsonValue = {
        "member_count": evidence.family_member_count,
        "member_digest": evidence.family_member_digest,
    }
    if approved.shape_member_digest is None or evidence.family_member_digest is None:
        return [_membership_unverified(requirement, found=found, required=required)]
    if evidence.family_member_digest == approved.shape_member_digest:
        return []
    return [
        EvidenceIssue(
            code=EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT,
            severity=Severity.ERROR,
            requirement_id=requirement.id,
            detail=(
                f"the definition was approved bound to {approved.shape_member_count} "
                f"StatVars with member digest {approved.shape_member_digest}; the "
                f"retrieval's family holds {evidence.family_member_count} StatVars with "
                f"member digest {evidence.family_member_digest}"
            ),
            found=found,
            required=required,
        )
    ]


def _single_series_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence, approved: SelectedSource
) -> list[EvidenceIssue]:
    """An ERROR when a source approved as one series is served as another number of StatVars.

    `search` states `served_count` once the membership settled, and one
    makes the plan read the source as a single series. A manifest whose
    trace lists no served StatVars proves nothing.
    """
    served = evidence.resolved_stat_vars
    if approved.served_count != 1 or served is None or len(served) == 1:
        return []
    return [
        EvidenceIssue(
            code=EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT,
            severity=Severity.ERROR,
            requirement_id=requirement.id,
            detail=(
                f"the definition was approved as one StatVar; the retrieval serves {len(served)}"
            ),
            found=len(served),
            required=1,
        )
    ]


def _binding_comparison_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence, approved_set: set[str]
) -> list[EvidenceIssue]:
    """The retrieved binding against the one dcid a pinned identity names: any difference is an ERROR.

    A manifest whose trace states no binding leaves membership unverified.
    """
    retrieved = evidence.resolved_stat_vars
    if retrieved is None:
        return [_membership_unverified(requirement, found=None, required=sorted(approved_set))]
    retrieved_set = {_bare_dcid(v) for v in retrieved}
    if approved_set == retrieved_set:
        return []
    return [
        EvidenceIssue(
            code=EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT,
            severity=Severity.ERROR,
            requirement_id=requirement.id,
            detail=(
                f"the definition was approved bound to {sorted(approved_set)!r}, "
                f"the retrieval binds it to {sorted(retrieved_set)!r}"
            ),
            found=sorted(retrieved_set),
            required=sorted(approved_set),
        )
    ]


def _membership_unverified(
    requirement: DatasetRequirement, *, found: JsonValue, required: JsonValue
) -> EvidenceIssue:
    """The WARNING `STAT_VAR_MEMBERSHIP_DRIFT` for a binding not known on both sides."""
    return EvidenceIssue(
        code=EvidenceIssueCode.STAT_VAR_MEMBERSHIP_DRIFT,
        severity=Severity.WARNING,
        requirement_id=requirement.id,
        detail="the definition's binding is not known on both sides, so membership cannot be confirmed",
        found=found,
        required=required,
    )


def _series_issues(
    requirement: DatasetRequirement, evidence: RetrievalEvidence
) -> list[EvidenceIssue]:
    """Per-series coverage: the requirement's period, one observation per year.

    Each series is held to the period on its own, so one entity's reach
    cannot stand in for another's. The manifest exposes each series' first
    and last period and its count, so a count short of the span means years
    are missing without saying which; a count above it means a grain finer
    than annual, or a facet the series did not split on. Missing years take
    the severity the requirement's `coverage` asks for; missing statistics
    are mandatory metadata the manifest failed to carry, and a grain that is
    not annual is never what was asked for, so both are errors whatever it
    says.

    Reading a count above the span as a finer grain rests on one series
    covering one entity per role. `compute_series_statistics` in apis'
    `dfo-data-tools` groups frame rows by every column but time_period,
    value and status, so two entities of the same role land in separate
    entries and a single entry's count is one entity's observations. A
    producer that merged entities into one entry would present a whole
    population's observations as one series and refuse it here as a grain
    error, which is what a ranking over a definition's full population
    would look like.
    """
    if evidence.series is None or not evidence.series.num_rows:
        # `None` (the manifest carries no statistics at all) and a table with
        # no rows are the same gap: mandatory metadata is missing, and a
        # check that cannot run is not a passed one, whatever
        # `coverage.on_missing` says.
        return [
            EvidenceIssue(
                code=EvidenceIssueCode.STATISTICS_UNAVAILABLE,
                severity=Severity.ERROR,
                requirement_id=requirement.id,
                detail="the manifest carries no per-series statistics, so coverage cannot be checked",
            )
        ]
    issues: list[EvidenceIssue] = []
    period = requirement.period
    for one in _series_off_period(evidence.series, period):
        reach = Period(start=one.first_year, end=one.latest_year)
        if not reach.covers(period):
            issues.append(
                EvidenceIssue(
                    code=EvidenceIssueCode.WINDOW_SHORT,
                    severity=_coverage_severity(requirement),
                    requirement_id=requirement.id,
                    detail=(
                        f"series {one.stat_var} for {list(one.entity_refs)!r} covers "
                        f"{reach.start}-{reach.end}, the requirement asks for "
                        f"{period.start}-{period.end}"
                    ),
                    found=reach.model_dump(),
                    required=period.model_dump(),
                )
            )
        span = one.latest_year - one.first_year + 1
        if one.observations < span:
            issues.append(
                EvidenceIssue(
                    code=EvidenceIssueCode.COVERAGE_GAPS,
                    severity=_coverage_severity(requirement),
                    requirement_id=requirement.id,
                    detail=(
                        f"series {one.stat_var} for {list(one.entity_refs)!r} has "
                        f"{one.observations} observations across {span} years"
                    ),
                    found=one.observations,
                    required=span,
                )
            )
        elif one.observations > span:
            issues.append(
                EvidenceIssue(
                    code=EvidenceIssueCode.GRAIN_AMBIGUOUS,
                    severity=Severity.ERROR,
                    requirement_id=requirement.id,
                    detail=(
                        f"series {one.stat_var} for {list(one.entity_refs)!r} has "
                        f"{one.observations} observations across {span} years; the grain is not annual"
                    ),
                    found=one.observations,
                    required=span,
                )
            )
    return issues


# --- Manifest: reading a `get_data` result into evidence ----------------------


def _bare_dcid(ref: str) -> str:
    """A stat_var as the graph names it, without the `dcid:` a frame cell carries."""
    return ref.removeprefix("dcid:")


def _resolved_stat_vars(trace: Mapping[str, object]) -> tuple[str, ...] | None:
    """The dcids `trace.resolved_stat_vars` binds this spec to, or `None` when the key is absent.

    Upstream states this is every StatVar behind the spec, independent of
    which series actually carried observations, so it is read here as a
    binding and never conflated with `statistics`' coverage. A manifest
    minted before the field existed, or a trace that otherwise carries no
    such key, comes back `None` -- unverified, not an empty binding.
    """
    raw = trace.get("resolved_stat_vars")
    if not isinstance(raw, list):
        return None
    return tuple(
        _bare_dcid(str(ref))
        for entry in raw
        if isinstance(entry, Mapping) and (ref := entry.get("ref")) is not None
    )


def _family_member_digest(trace: Mapping[str, object]) -> str | None:
    """`trace.family_member_digest`, or `None` when the trace states none."""
    digest = trace.get("family_member_digest")
    return digest if isinstance(digest, str) and digest else None


def _family_member_count(trace: Mapping[str, object]) -> int | None:
    """`trace.family_member_count`, or `None` when the trace states no whole number."""
    count = trace.get("family_member_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def _facet_unit(value: object) -> str | Unit | None:
    """The unit a series' facet states: a bare ref, a `{ref, label}` block,
    or the frame grammar's `{currency, prices, base_year}` object. An object
    outside that grammar states no unit."""
    if isinstance(value, str):
        return value or None
    if not isinstance(value, dict):
        return None
    if "currency" in value:
        try:
            return Unit.model_validate(value)
        except ValidationError:
            return None
    ref = value.get("ref") or value.get("label")
    return ref if isinstance(ref, str) and ref else None


def _year_of(period: str) -> int | None:
    """The calendar year a period string starts with, or `None` when it has none."""
    head = period[:4]
    return int(head) if head.isdigit() else None


def _window_of(series: pa.Table | None) -> Period | None:
    """The span from the earliest first to the latest last year across `series`."""
    if series is None:
        return None
    first = scalar_kernel("min", _years(_column(series, "first_period"))).as_py()
    latest = scalar_kernel("max", _years(_column(series, "latest_period"))).as_py()
    if first is None or latest is None:
        return None
    return Period(start=first, end=latest)


def _column(series: pa.Table, name: str) -> pa.Array:
    """Column `name` of `series` as one array."""
    return series.column(name).combine_chunks()


def _years(periods: pa.Array) -> pa.Array:
    """`_year_of` of each period, read once per distinct period."""
    distinct = array_kernel("unique", periods)
    years = pa.array([_year_of(period) for period in distinct.to_pylist()], pa.int64())
    return array_kernel("take", years, array_kernel("index_in", periods, value_set=distinct))


def _series_stat_vars(series: pa.Table) -> tuple[str, ...]:
    """The distinct stat_vars `series` carries, bare and sorted."""
    distinct = array_kernel("unique", _column(series, "stat_var")).to_pylist()
    return tuple(sorted({_bare_dcid(one) for one in distinct}))


def _series_entities(series: pa.Table) -> tuple[ResolvedEntity, ...]:
    """Every entity `series` names, in first-seen order with repeats dropped;
    several series share one."""
    named = array_kernel("list_flatten", _column(series, "entities"))
    distinct = array_kernel("take", named, unit_codes(named).first_rows)
    return tuple(
        ResolvedEntity(role=entity["role"], ref=entity["ref"], label=entity["ref"])
        for entity in distinct.to_pylist()
    )


def _series_units(series: pa.Table) -> set[str | Unit]:
    """The units the facets of `series` state, read once per distinct facet."""
    distinct = array_kernel("unique", _column(series, "facet")).to_pylist()
    facets = (json.loads(facet) for facet in distinct)
    return {unit for facet in facets if (unit := _facet_unit(facet.get("unit")))}


def _series_off_period(series: pa.Table, period: Period) -> Iterator[SeriesEvidence]:
    """The series whose reach misses `period` or whose count differs from its
    span, in the order of their head rows: the only ones `_series_issues`
    reports on.

    A series' rows (`RetrievalEvidence.series`) are read together: it reaches
    from its rows' earliest first year to their latest last year, and counts
    their observations summed, so a series and the estimates continuing it are
    held to the period as one. A series with a first or last period that
    names no year is left out, since neither check can run on it. The rest
    are selected column by column, and only the selected series become
    `SeriesEvidence`.
    """
    unskipped = pc.ScalarAggregateOptions(skip_nulls=False)
    folded = (
        pa.table(
            {
                "series": _column(series, "series"),
                "first_year": _years(_column(series, "first_period")),
                "latest_year": _years(_column(series, "latest_period")),
                "observations": _column(series, "observations"),
            }
        )
        .group_by("series", use_threads=False)
        .aggregate(
            [
                ("first_year", "min", unskipped),
                ("latest_year", "max", unskipped),
                ("observations", "sum"),
            ]
        )
        .sort_by("series")
    )
    heads = _column(folded, "series")
    first = _column(folded, "first_year_min")
    latest = _column(folded, "latest_year_max")
    observations = _column(folded, "observations_sum")
    covers = array_kernel(
        "and_",
        array_kernel("less_equal", first, period.start),
        array_kernel("greater_equal", latest, period.end),
    )
    span = array_kernel("add", array_kernel("subtract", latest, first), 1)
    # Null where a year is missing, which `filter` drops.
    flagged = array_kernel(
        "or_", array_kernel("invert", covers), array_kernel("not_equal", observations, span)
    )
    rows = array_kernel("filter", pa.array(range(folded.num_rows), pa.int64()), flagged)
    selected_heads = array_kernel("take", heads, rows)
    selected = pa.table(
        {
            "stat_var": array_kernel("take", _column(series, "stat_var"), selected_heads),
            "entities": array_kernel("take", _column(series, "entities"), selected_heads),
            "first_year": array_kernel("take", first, rows),
            "latest_year": array_kernel("take", latest, rows),
            "observations": array_kernel("take", observations, rows),
        }
    )
    for row in selected.to_pylist():
        yield SeriesEvidence(
            stat_var=_bare_dcid(row["stat_var"]),
            entity_refs=tuple(entity["ref"] for entity in row["entities"]),
            first_year=row["first_year"],
            latest_year=row["latest_year"],
            observations=row["observations"],
        )
