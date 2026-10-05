"""Reads the value a step was compiled to produce out of what the tool returned.

Each `ToolOutput` in the capability table names how its value is extracted
from a tool's structured result. The executor reads through here and never
through the digest text, and a result that does not carry what its capability
declares is an `ArtifactError` for the run to stop on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from common.materialisation import FrameRef
from compiler.capabilities import Extract
from compiler.frame_columns import FrameColumns, frame_columns
from data_mcp.manifest import Manifest
from dfo_frame_grammar.grammar import FrameError, normalize_role_keys
from pydantic import ValidationError


class ArtifactError(ValueError):
    """The structured result does not carry the value its capability declares."""


def extract_frame_ref(
    extract: Extract, structured_content: Mapping[str, object] | None
) -> FrameRef:
    """Read a frame reference from `structured_content` the way `extract` says to.

    `get_data`'s result is a definition manifest with the frame's reference beside
    it as `frame_ref`. The manifest's own pair names dfo-apis' materialisation, and
    a frame minted with extra columns is held under a `content_version` of its own
    (`data_mcp.frame_materialisation.frame_ref_for`), so the reference is read from
    `frame_ref`.

    Raises:
        ArtifactError: If the content is missing, is not what the extractor
            reads, or names a pair that is not a valid `FrameRef`.
    """
    if structured_content is None:
        raise ArtifactError("the tool returned no structured content")
    try:
        if extract is Extract.FRAME_REF_FROM_MANIFEST:
            manifest = parse_manifest(structured_content)
            if manifest.definition is None:
                raise ArtifactError("the manifest describes a collection, not one definition")
            return FrameRef.model_validate(structured_content.get("frame_ref"))
        if extract is Extract.FRAME_REF_PAIR:
            return FrameRef.model_validate(
                {
                    key: structured_content.get(key)
                    for key in ("spec_id", "content_version")
                    if key in structured_content
                }
            )
    except ValidationError as exc:
        raise ArtifactError(f"the result does not carry a valid frame reference: {exc}") from exc
    raise ArtifactError(f"{extract.value!r} does not yield a frame reference")


def declared_shape(
    structured_content: Mapping[str, object] | None,
) -> tuple[int | None, tuple[str, ...] | None, int | None]:
    """`(rows, role_keys, stat_var_count)` as a successful call declared them
    in its own `structured_content` (`analysis_mcp.result.VerbResult`'s wire
    fields), or `None` per field when the content carries none. `get_data`'s
    structured result is a `Manifest`, which declares no `rows`, with the
    frame's `role_keys` and `stat_var_count` beside it whenever data-mcp read
    the frame; and any of the three can be individually absent from a
    malformed payload.

    Read positionally rather than through pydantic validation: a step whose
    capability declares no output schema (`get_data`) is expected to carry
    at most `role_keys` and `stat_var_count`, so this is a read, not a
    contract check. The contract check is `_surface_as_compiled`'s
    schema-hash comparison.

    The role keys come back as declared, in the declared order. They are the
    column names every later step binds to, so a list the frame grammar
    refuses (`normalize_role_keys`: a reserved column name, or two keys
    equal after NFC normalisation) is not a shape to read.

    Raises:
        ArtifactError: If the declared `role_keys` are not valid role columns.
    """
    if structured_content is None:
        return None, None, None
    rows = structured_content.get("rows")
    role_keys = structured_content.get("role_keys")
    stat_var_count = structured_content.get("stat_var_count")
    keys = tuple(str(key) for key in role_keys) if isinstance(role_keys, list | tuple) else None
    if keys is not None:
        try:
            normalize_role_keys(keys)
        except FrameError as exc:
            raise ArtifactError(
                f"the result declares role keys {list(keys)!r} that are not valid role "
                f"columns: {exc}"
            ) from exc
    return (
        rows if isinstance(rows, int) else None,
        keys,
        stat_var_count if isinstance(stat_var_count, int) else None,
    )


def retrieved_frame_columns(
    structured_content: Mapping[str, object], *, compiled: Iterable[str]
) -> FrameColumns | None:
    """Bind `compiled` to the columns of the frame `get_data`'s result names.

    Reads the frame's `role_keys`, which `get_data` reports beside the
    manifest whenever it read or parsed the frame, and the flow `axis` of
    each of the manifest's coverage dimensions. `None` when the result
    carries no `role_keys`: the frame's columns are unknown, and the plan's
    own spelling stands.

    Raises:
        ArtifactError: If the content is not a manifest, or its `role_keys`
            are not valid role columns (`declared_shape`).
    """
    _rows, role_keys, _stat_var_count = declared_shape(structured_content)
    if role_keys is None:
        return None
    manifest = parse_manifest(structured_content)
    return frame_columns(
        role_keys=role_keys,
        axis_by_column={
            dimension.label: dimension.axis for dimension in manifest.coverage.dimensions
        },
        compiled=compiled,
    )


def parse_manifest(structured_content: Mapping[str, object]) -> Manifest:
    """Read `get_data`'s structured result as the `Manifest` it is.

    Raises:
        ArtifactError: If the content is not a manifest.
    """
    try:
        return Manifest.model_validate(dict(structured_content))
    except ValidationError as exc:
        raise ArtifactError(f"the result is not a manifest: {exc}") from exc
