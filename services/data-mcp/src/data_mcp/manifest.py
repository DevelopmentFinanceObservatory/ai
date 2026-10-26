"""This service's own manifest model, stripped of every artifact download link.

`get_data` mints a manifest from apis' mint route
(`datasets_api.manifest_models.Manifest`). That manifest's `artifacts`
entries carry `url`: a stable, root-relative download link that redirects to
the underlying bytes. A model gets the link through the artifact resource
(`artifact_resource.py`) when it needs one, so this module re-validates the
raw mint payload against a model this service owns rather than reusing
apis'. Reuse isn't an option regardless: upstream's `ArtifactEntryModel`
requires `url` with no default and sets `extra="forbid"`, so there is no way
to express "the same shape minus the URL" as a subset of it.

`strip_manifest` is the public seam: validate a raw mint response through
it and get back a `Manifest` with every artifact's URL gone and
`artifact_digest` intact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

# The templated URI a caller reads to get the download link for one
# artifact. The constant `artifact` host segment is deliberate: `AnyUrl`
# normalisation lowercases the host position but preserves path segments, so
# every variable -- `subject_id`, `content_version`, `artifact_digest` --
# lives in the path, where a mixed-case `spec_` id or a `sha256:`
# digest survives verbatim.
ARTIFACT_URI_TEMPLATE = "dfo://artifact/{subject_id}/{content_version}/{artifact_digest}"


def artifact_resource_uri(*, subject_id: str, content_version: str, artifact_digest: str) -> str:
    """Build the resource URI for one artifact, per `ARTIFACT_URI_TEMPLATE`."""
    return ARTIFACT_URI_TEMPLATE.format(
        subject_id=subject_id, content_version=content_version, artifact_digest=artifact_digest
    )


# Applied to every model in this tree instead of the house `extra="forbid"`
# (see `search_tool.py`). `forbid` would mean any new field apis adds to
# the manifest (`series_count`, for instance) breaks every retrieval until
# this module catches up. `allow` would
# forward a field this service has never reviewed straight to the model,
# defeating the point of owning this shape at all. `ignore` does neither:
# an unrecognised field is silently dropped rather than surfaced, so
# retrieval keeps working across an upstream schema change and nothing
# unreviewed ever reaches the model.
_UNSEEN_FIELDS_ARE_DROPPED = ConfigDict(extra="ignore")


class RoleNoun(BaseModel):
    """What apis calls the members of a role, for the frame's kind of
    flow: a provider and providers of aid, a creditor and creditors of debt."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    singular: str = Field(min_length=1)
    plural: str = Field(min_length=1)


class CoverageDimension(BaseModel):
    """One counted dimension of a mint: a declared role column, or `years`.

    `axis` is the flow axis (`origin`, `destination`) the role column's
    declared property sits on. It is absent for `years`, for a role no flow
    axis covers (`observationAbout`), and from an apis service that predates the
    field. A plain string, so an axis this service has not seen still
    parses.

    `noun` is what apis calls the role's members
    (`common.period_note` counts them by it). It is absent for `years`, for a
    role apis words no noun for, and from an apis service that predates the
    field; a noun of any other shape reads as absent, so it fails no
    retrieval.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    label: str
    count: int
    axis: str | None = None
    noun: RoleNoun | None = None

    @field_validator("noun", mode="wrap")
    @classmethod
    def _noun_or_absent(cls, noun: object, handler: ValidatorFunctionWrapHandler) -> object:
        try:
            return handler(noun)
        except ValidationError:
            return None


class YearCoverage(BaseModel):
    """What one calendar year holds of a mint's admitted slice: its series,
    and the distinct members of each role those series carry."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    year: int
    series: int
    role_counts: dict[str, int] = Field(default_factory=dict)


def _is_none(value: object) -> bool:
    return value is None


class CoverageModel(BaseModel):
    """What a mint served: its rows, counted dimensions and window.

    `recent_years` holds three calendar years, newest first, `elected_year`
    the year apis' election chooses among them, and `fullest_year` the
    fullest of the last ten. apis sends each only for a frame of one
    year, and `fullest_year` only when it judges the served year thin beside
    it. An absent one stays out of a dump, so a manifest without them
    serialises as it arrived.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    rows: int
    dimensions: list[CoverageDimension]
    window: dict[str, object] | None = None
    recent_years: list[YearCoverage] | None = Field(default=None, exclude_if=_is_none)
    elected_year: int | None = Field(default=None, exclude_if=_is_none)
    fullest_year: YearCoverage | None = Field(default=None, exclude_if=_is_none)


class CitationSource(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    ref: str
    label: str
    processing_level: str | None = None
    license: str | None = None


class CitationModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    text: str
    sources: list[CitationSource]


class CountedRefsModel(BaseModel):
    """The things of one kind a warning counts: how many, and the `dcid:` refs of the
    first few (apis' `CountedRefsModel`). The warning's message states the count and
    names none."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    count: int
    refs: list[str]


class EntityYearsModel(BaseModel):
    """One entity a conversion left rows out of, and the years of those rows as runs,
    `1990-2001` or `2004` (apis' `EntityYearsModel`)."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    ref: str
    years: list[str]


class CountedEntityYearsModel(BaseModel):
    """The entities a conversion left rows out of: how many, and the first few, each with
    its years (apis' `CountedEntityYearsModel`)."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    count: int
    entities: list[EntityYearsModel]


class WarningModel(BaseModel):
    """One manifest warning.

    `contributing_member_count` and `stat_var_label` are set on a `stat_vars_summed`
    warning only: the number of StatVars with rows that apis added into the family
    total, and the label apis' presentation door gives the family ref.

    A `coverage_below_scope` warning can carry `member_count`, the series a group
    requested, `off_cadence_members` and `out_of_window_members`, the group members the
    figures leave out, and `unserved_entities`, the bound entities that served nothing.
    `omitted_entities` is set on `currency_unmapped_omitted` and `omitted_entity_years`
    on `deflator_missing_omitted`. Each stays out of a dump when absent, so a warning
    without them serialises as it arrived.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    code: str
    severity: str
    message: str
    contributing_member_count: int | None = None
    stat_var_label: str | None = None
    member_count: int | None = Field(default=None, exclude_if=_is_none)
    off_cadence_members: CountedRefsModel | None = Field(default=None, exclude_if=_is_none)
    out_of_window_members: CountedRefsModel | None = Field(default=None, exclude_if=_is_none)
    unserved_entities: CountedRefsModel | None = Field(default=None, exclude_if=_is_none)
    omitted_entities: CountedRefsModel | None = Field(default=None, exclude_if=_is_none)
    omitted_entity_years: CountedEntityYearsModel | None = Field(default=None, exclude_if=_is_none)


class StatPointModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    value: float
    period: str


class SeriesChangeModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    absolute: float
    # Upstream's own field has no default, but every mint door serialises
    # the manifest with `exclude_none=True` (`doors/mint.py`), so a `None`
    # `percent` is absent from the wire rather than sent as `null`.
    percent: float | None = None


class SeriesEntityModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    role: str
    ref: str


class SeriesStatisticsModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    stat_var: str
    entities: list[SeriesEntityModel]
    facet: dict[str, str | dict[str, object]]
    observations: int
    first: StatPointModel
    latest: StatPointModel
    minimum: StatPointModel
    maximum: StatPointModel
    # Same `exclude_none=True` reasoning as `SeriesChangeModel.percent`
    # above: a series with only one observation ships no `change` key at
    # all rather than `"change": null`.
    change: SeriesChangeModel | None = None


class ArtifactEntry(BaseModel):
    """One artifact, minus its download link.

    `resource_uri` is a model's replacement for the stripped link: read it to
    pull the link at fetch time. `Manifest._attach_artifact_resource_uris` derives and
    overwrites it after validation, so the default here only has to satisfy
    an already-stripped document that predates the field.

    `enrichment_stamp` is the digest of the code-index facts behind the file's
    `{role}_party_kind`, `{role}_party_class` and `{role}_party_whole` cells (apis'
    `CodeIndex.enrichment_digest`). apis sends
    it exactly when the file holds party-kind cells read under the code index serving
    the call. A file with no party-kind cell carries none, and so does a listed file
    rendered under an earlier digest with no current twin (a pinned file, an older
    layout), whose kinds may be out of date. `data_mcp.frame_materialisation.
    frame_ref_for` salts the name of a frame read from the file with it, so a change
    to a party's kind names a new frame, and a frame read from a file without one is
    named unsalted. It stays out of a dump when absent.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    artifact_digest: str
    rendering: dict[str, object]
    bytes: int
    media_type: str
    created_at: str
    tier: str
    resource_uri: str | None = None
    enrichment_stamp: str | None = Field(default=None, exclude_if=_is_none)


class DefinitionModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    schema_version: int
    spec_id: str
    identity: Mapping[str, object]


class PartyTotalsLeftOut(BaseModel):
    """The group totals apis left out of one party role of a frame minted with
    `options.without_party_totals`, in ref order, within one provenance: one entry of the
    manifest's `trace.party_totals_left_out` (`party_totals_left_out`). `provenance` is the
    dcid of the dataset whose rows lost them, and `None` for the rows that carry no
    provenance. An entry with no `provenance` key comes from a trace recorded before apis
    keyed the cut on the dataset, whose cut covered every dataset; `model_fields_set` tells
    that entry from a null one."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    role: str
    provenance: str | None = None
    entities: list[str]


# The trace member apis lists a cut frame's left-out totals under, present only when
# the cut left some out (`datasets_api.party_totals.TRACE_KEY`).
PARTY_TOTALS_LEFT_OUT_KEY = "party_totals_left_out"
_PARTY_TOTALS_LEFT_OUT = TypeAdapter(list[PartyTotalsLeftOut])


def party_totals_left_out(trace: Mapping[str, object]) -> list[PartyTotalsLeftOut]:
    """The group totals a manifest's `trace` says apis left out of its frame, one entry
    per party role and provenance in role order, and none for a frame that left none out.

    Raises:
        pydantic.ValidationError: When the trace lists them in another shape.
    """
    return _PARTY_TOTALS_LEFT_OUT.validate_python(trace.get(PARTY_TOTALS_LEFT_OUT_KEY) or [])


class ManifestComponent(BaseModel):
    """One component of a composite frame: one entry of the manifest's `trace.components`
    (apis' `materialise.composite.ComponentSummary`).

    A composite definition binds a declared composite provenance, which no graph node
    holds, and apis mints it from the tables its components name: stacked components
    (bilateral plus imputed multilateral aid) share years and each row is one
    component's, and spliced ones (net flows to 2017, grant equivalents from 2018) serve
    disjoint years. `role` names what the component holds, `provenance` the table it was
    read from, `periods` the years it serves as closed spans, `dropped_years` the years
    left out of it, outside the years the composite serves, `stat_vars` its members,
    `rows` its rows in the frame and `zero_rows` how many of them are zeros written where
    a table that reads absence as zero holds nothing.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    role: str
    provenance: str | None = None
    content_version: str | None = None
    periods: list[tuple[int, int]] = Field(default_factory=list)
    dropped_years: list[int] = Field(default_factory=list)
    stat_vars: list[str] = Field(default_factory=list)
    rows: int = 0
    zero_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


# The trace member apis lists a composite frame's components under, absent on a frame of
# one table.
COMPONENTS_KEY = "components"
_COMPONENTS = TypeAdapter(list[ManifestComponent])


def manifest_components(trace: Mapping[str, object]) -> list[ManifestComponent]:
    """The components a manifest's `trace` says its composite frame was minted from, in
    apis' order, and none for a frame of one table.

    Raises:
        pydantic.ValidationError: When the trace lists them in another shape.
    """
    return _COMPONENTS.validate_python(trace.get(COMPONENTS_KEY) or [])


class Manifest(BaseModel):
    """The mint route's manifest, re-validated with every artifact URL gone.

    Mirrors `datasets_api.manifest_models.Manifest` field for field, except
    `artifacts` holds `ArtifactEntry` rather than upstream's
    `ArtifactEntryModel`. `series_count` is the number of distinct series.
    Per-series figures live in a `series-statistics/1` Parquet entry:
    whenever `series_count` is not `None`, `artifacts` carries one
    (`data_mcp.series_statistics.find_statistics_artifact`) for a caller
    to read.

    `series_count` defaults to `None` because it is optional upstream too: a
    manifest minted before that field existed has no backfill path, and
    would otherwise fail retrieval forever. Such a record may carry an
    inline `statistics` array upstream once sent; this model does not
    declare that field, so `extra="ignore"` (`_UNSEEN_FIELDS_ARE_DROPPED`)
    silently drops it before a reader ever sees it.

    The stamp of the kind cells sits on each artifact (`ArtifactEntry.enrichment_stamp`),
    since one manifest may list files rendered under different code indexes.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    schema_version: int = 1
    definition: DefinitionModel
    content_version: str
    coverage: CoverageModel
    citation: CitationModel
    trace: dict[str, object]
    warnings: list[WarningModel] = Field(default_factory=list)
    artifacts: list[ArtifactEntry]
    series_count: int | None = None

    @model_validator(mode="after")
    def _attach_artifact_resource_uris(self) -> Manifest:
        """Derive every artifact's `resource_uri`, overwriting whatever the input carried.

        The artifact list is only ever at this top level, so every artifact
        shares this manifest's own `content_version` and is keyed on the
        definition's `spec_id`.
        """
        for artifact in self.artifacts:
            artifact.resource_uri = artifact_resource_uri(
                subject_id=self.definition.spec_id,
                content_version=self.content_version,
                artifact_digest=artifact.artifact_digest,
            )
        return self


_MAX_RENDERED_VALIDATION_ERRORS = 5


def render_validation_errors(exc: ValidationError) -> str:
    """Render `exc`'s errors as `location: message` pairs, without the raw input they failed on.

    `str(exc)` embeds pydantic-core's `input_value` for each failing field. For an
    `ArtifactEntry`, that's the artifact's raw, un-stripped dict -- `extra="ignore"`
    only drops its `url` on a successful build, so a validation failure on the
    same field would otherwise render that link straight into a refusal detail.
    `include_url=False` refers to pydantic's own per-error docs link
    (errors.pydantic.dev/...), not the artifact link; paired with `include_input=False`,
    neither the docs link nor the raw input value is rendered. Capped at
    `_MAX_RENDERED_VALIDATION_ERRORS` so a payload with many failing fields can't blow
    up the refusal string.
    """
    errors = exc.errors(include_url=False, include_input=False)
    rendered = [
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in errors[:_MAX_RENDERED_VALIDATION_ERRORS]
    ]
    omitted = len(errors) - len(rendered)
    if omitted > 0:
        rendered.append(f"... and {omitted} more")
    return "; ".join(rendered)


def strip_manifest(payload: Mapping[str, Any]) -> Manifest:
    """Validate a raw mint-route response into a URL-free `Manifest`.

    Args:
        payload: The mint route's parsed JSON body, exactly as apis
            served it.

    Returns:
        The same manifest, minus every artifact's `url`.

    Raises:
        pydantic.ValidationError: `payload` doesn't match the manifest
            shape this service knows how to serve -- for example missing a
            required field such as `definition`.
    """
    return Manifest.model_validate(payload)
