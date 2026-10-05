"""This service's own manifest model, stripped of every signed artifact URL.

`get_data` mints a manifest from dfo-apis' mint route
(`datasets_api.manifest_models.Manifest`). That manifest's `artifacts`
entries carry `url`/`url_expires_at`: short-lived, signed links straight to
the underlying bytes. Nothing that reaches a model should ever see one, so
this module re-validates the raw mint payload against a model this service
owns rather than reusing dfo-apis'. Reuse isn't an option regardless:
upstream's `ArtifactEntryModel` requires `url`/`url_expires_at` with no
default and sets `extra="forbid"`, so there is no way to express "the same
shape minus the URL" as a subset of it.

`strip_manifest` is the public seam: validate a raw mint response through
it and get back a `Manifest` with every artifact's URL gone and
`artifact_digest` intact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# The templated URI a caller reads to get a freshly signed URL for one
# artifact. The constant `artifact` host segment is deliberate: `AnyUrl`
# normalisation lowercases the host position but preserves path segments, so
# every variable -- `subject_id`, `content_version`, `artifact_digest` --
# lives in the path, where a mixed-case `spec_`/`col_` id or a `sha256:`
# digest survives verbatim.
ARTIFACT_URI_TEMPLATE = "dfo://artifact/{subject_id}/{content_version}/{artifact_digest}"


def artifact_resource_uri(*, subject_id: str, content_version: str, artifact_digest: str) -> str:
    """Build the resource URI for one artifact, per `ARTIFACT_URI_TEMPLATE`."""
    return ARTIFACT_URI_TEMPLATE.format(
        subject_id=subject_id, content_version=content_version, artifact_digest=artifact_digest
    )


# Applied to every model in this tree instead of the house `extra="forbid"`
# (see `search_tool.py`). `forbid` would mean any new field dfo-apis adds to
# the manifest (`series_count`, for instance) breaks every retrieval until
# this module catches up. `allow` would
# forward a field this service has never reviewed straight to the model,
# defeating the point of owning this shape at all. `ignore` does neither:
# an unrecognised field is silently dropped rather than surfaced, so
# retrieval keeps working across an upstream schema change and nothing
# unreviewed ever reaches the model.
_UNSEEN_FIELDS_ARE_DROPPED = ConfigDict(extra="ignore")


class CoverageDimension(BaseModel):
    """One counted dimension of a mint: a declared role column, or `years`.

    `axis` is the flow axis (`origin`, `destination`) the role column's
    declared property sits on. It is absent for `years`, for a role no flow
    axis covers (`observationAbout`), and from a dfo-apis that predates the
    field. A plain string, so an axis this service has not seen still
    parses.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    label: str
    count: int
    axis: str | None = None


class CoverageModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    rows: int
    dimensions: list[CoverageDimension]
    window: dict[str, object] | None = None


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


class WarningModel(BaseModel):
    """One manifest warning.

    `contributing_member_count` and `stat_var_label` are set on a `stat_vars_summed`
    warning only: the number of StatVars with rows that dfo-apis added into the family
    total, and the label dfo-apis' presentation door gives the family ref.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    code: str
    severity: str
    message: str
    contributing_member_count: int | None = None
    stat_var_label: str | None = None


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
    """One artifact, minus its signed URL and that URL's expiry.

    `resource_uri` is a model's replacement for the stripped URL: read it to
    pull a freshly signed URL at fetch time, rather than trusting one that
    could go stale. `Manifest._attach_artifact_resource_uris` derives and
    overwrites it after validation, so the default here only has to satisfy
    an already-stripped document that predates the field.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    artifact_digest: str
    rendering: dict[str, object]
    bytes: int
    media_type: str
    created_at: str
    tier: str
    resource_uri: str | None = None


class DefinitionModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    schema_version: int
    spec_id: str
    identity: Mapping[str, object]


class CollectionMemberModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    definition: DefinitionModel
    content_version: str
    coverage: CoverageModel
    series_count: int | None = None


class CollectionModel(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    collection_id: str
    members: list[CollectionMemberModel]


class Manifest(BaseModel):
    """The mint route's manifest, re-validated with every artifact URL gone.

    Mirrors `datasets_api.manifest_models.Manifest` field for field, except
    `artifacts` holds `ArtifactEntry` rather than upstream's
    `ArtifactEntryModel`. Exactly one of `definition`/`collection` is
    present, matching upstream's own rule. `series_count` is the number of
    distinct series and lives at this top level only for a definition
    manifest; a collection manifest carries its own per-member count on each
    `CollectionMemberModel` instead (see `_series_count_only_at_the_right_level`).
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
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    schema_version: int = 1
    definition: DefinitionModel | None = None
    collection: CollectionModel | None = None
    content_version: str
    coverage: CoverageModel
    citation: CitationModel
    trace: dict[str, object]
    warnings: list[WarningModel] = Field(default_factory=list)
    artifacts: list[ArtifactEntry]
    series_count: int | None = None

    @model_validator(mode="after")
    def _definition_xor_collection(self) -> Manifest:
        if (self.definition is None) == (self.collection is None):
            raise ValueError("exactly one of 'definition' or 'collection' must be present")
        return self

    @model_validator(mode="after")
    def _series_count_only_at_the_right_level(self) -> Manifest:
        if self.collection is not None and self.series_count is not None:
            raise ValueError(
                "top-level 'series_count' is only for a definition manifest; "
                "a collection manifest carries series_count per member"
            )
        return self

    @model_validator(mode="after")
    def _attach_artifact_resource_uris(self) -> Manifest:
        """Derive every artifact's `resource_uri`, overwriting whatever the input carried.

        Runs after `_definition_xor_collection`, whose guarantee that exactly
        one of `definition`/`collection` is present is what makes a single
        subject id always available here. A collection manifest's members
        each carry their own `content_version`, but the artifact list itself
        is only ever at this top level, so every artifact shares this
        manifest's own `content_version` -- one resource_uri per artifact,
        keyed on the collection as a whole.
        """
        if self.definition is not None:
            subject_id = self.definition.spec_id
        elif self.collection is not None:
            subject_id = self.collection.collection_id
        else:
            # Unreachable: `_definition_xor_collection` already refused a
            # manifest with neither present, before this validator runs.
            raise AssertionError("neither 'definition' nor 'collection' is present")
        for artifact in self.artifacts:
            artifact.resource_uri = artifact_resource_uri(
                subject_id=subject_id,
                content_version=self.content_version,
                artifact_digest=artifact.artifact_digest,
            )
        return self


_MAX_RENDERED_VALIDATION_ERRORS = 5


def render_validation_errors(exc: ValidationError) -> str:
    """Render `exc`'s errors as `location: message` pairs, without the raw input they failed on.

    `str(exc)` embeds pydantic-core's `input_value` for each failing field. For an
    `ArtifactEntry`, that's the artifact's raw, un-stripped dict -- `extra="ignore"`
    only drops its signed `url` on a successful build, so a validation failure on the
    same field would otherwise render that URL straight into a refusal detail.
    `include_url=False` refers to pydantic's own per-error docs link
    (errors.pydantic.dev/...), not the artifact URL; paired with `include_input=False`,
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
        payload: The mint route's parsed JSON body, exactly as dfo-apis
            served it.

    Returns:
        The same manifest, minus every artifact's `url`/`url_expires_at`.

    Raises:
        pydantic.ValidationError: `payload` doesn't match the manifest
            shape this service knows how to serve -- missing a required
            field, failing the definition/collection exclusivity rule, or
            carrying top-level `statistics` on a collection manifest.
    """
    return Manifest.model_validate(payload)
