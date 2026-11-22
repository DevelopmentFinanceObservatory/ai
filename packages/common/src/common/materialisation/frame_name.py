"""The pair a frame of a rendering is held under, derived from the dataset pair apis
names and the mint options that shape the frame.

`frame_ref_for` is the one rule: data-mcp registers a minted frame under it, and
presentation-mcp names the frame a refresh moved to (`presentation_mcp.client`). apis'
`analysis_roots.frame_ref_for` names root frames by the same preimage with apis' render
format in place of the artifact digest, so the two name a rendering's frame apart.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import rfc8785

from common.materialisation.frame_ref import FrameRef
from common.materialisation.mint_record import groupby_axes, normal_groupby

# The bare tokens apis expands to a `{column}_{token}` column for every party column a flow
# table has: the party's kind, class and whole-population mark (`common.place_type`).
KIND_COLUMNS: tuple[str, ...] = ("party_kind", "party_class", "party_whole")

# The mint option that asks apis for a flow frame without the group totals of the
# party roles its identity leaves open (apis' `MintOptions.without_party_totals`). A
# mint sends it only when true, and apis records it in the artifact's `rendering` only
# then. A cut that leaves rows out is a version of its own, which the manifest's pair names.
WITHOUT_PARTY_TOTALS = "without_party_totals"


def requests_kinds(columns: Iterable[str]) -> bool:
    """Whether `columns` asks apis for any of a flow party's kind, class or whole cells."""
    return not set(KIND_COLUMNS).isdisjoint(columns)


def frame_ref_for(
    *,
    spec_id: str,
    content_version: str,
    columns: tuple[str, ...] = (),
    groupby: Mapping[str, Any] | None = None,
    without_party_totals: bool = False,
    enrichment_stamp: str | None = None,
    artifact_digest: str | None = None,
    keep_by_order: bool = False,
) -> FrameRef:
    """The frame reference of the frame a mint names as `{spec_id, content_version}`,
    minted with `options.columns` set to `columns`, `options.groupby` to `groupby` and
    `options.without_party_totals` to `without_party_totals`.

    apis keeps one `content_version` for every rendering of a materialisation,
    so a frame minted with extra columns (`get_data(place_types=True)`), a sum that
    keeps the version, and one minted with the default columns come back under one
    pair. The default rendering, with no columns, no groupby and no cut, keeps the
    mint's own pair and holds the version's canonical frame. Any other names its options in its
    `content_version`: `cv_` and the first 32 hex digits of the SHA-256 of the
    RFC 8785 form of `{"columns": [...], "content_version": ...}`, with `columns`
    sorted, a `"groupby"` member holding `groupby` with its `by` sorted when it is set,
    and `"without_party_totals": true` when the cut is asked for. Two renderings
    of one pair are then two frames, and the store's insert-only entry for one is
    served for that one only. A groupby sum apis mints as a version of its own
    is still named by its groupby, since data-mcp cannot tell it from a sum that
    keeps the version. The versions route that says so needs a first-party
    principal this service does not hold.

    A `columns` that asks for `KIND_COLUMNS` also names `enrichment_stamp`, the digest
    of the code-index facts behind the kind cells of the artifact the frame is read
    from (`data_mcp.manifest.ArtifactEntry.enrichment_stamp`), in the preimage as an
    `"enrichment_stamp"` member. apis keeps one
    `content_version` across a change to a party's kind, so without it the store
    would serve the frame holding the old kinds for as long as it keeps the entry.
    With no stamp, or with columns that ask for no kinds, the ref is the unsalted one.

    `artifact_digest` is the digest of the bytes of the file the frame is read from
    (`data_mcp.manifest.ArtifactEntry.artifact_digest`), named in the preimage as an
    `"artifact_digest"` member when set. apis links each rendering of a version to one
    file and links a new one when it renders that rendering's cells another way, such as
    a change to how it decodes a breakdown code, keeping the `content_version`. Naming
    the frame by its file's digest makes such a change a new frame, so the store serves
    no frame read from the replaced file. A frame registered before data-mcp named
    frames by the digest is named without it. The version's own pair ignores it: the
    canonical frame is checked against its `content_version`.

    `options.without_party_totals` names the frame although apis mints a frame without
    party totals a version of its own: a cut that leaves nothing out serves its base's
    frame under its base's version, and a recipe replaying the frame read with the cut
    replays the cut, so the frame is named apart from the one read without it.

    `keep_by_order` names `groupby` with `by` in the order it holds, the pair of a frame
    registered before this function sorted `by`
    (`data_mcp.frame_materialisation.named_groupbys`).

    Raises:
        ValueError: When `spec_id` and `content_version` do not fit `FrameRef`'s pair
            shape (`pydantic.ValidationError`), or `groupby`'s `by` is no list of axis
            names (`groupby_axes`).
    """
    ref = FrameRef(spec_id=spec_id, content_version=content_version)
    if not columns and groupby is None and not without_party_totals:
        return ref
    preimage: dict[str, Any] = {"columns": sorted(columns), "content_version": content_version}
    if groupby is not None:
        preimage["groupby"] = (
            {**groupby, "by": groupby_axes(groupby)} if keep_by_order else normal_groupby(groupby)
        )
    if without_party_totals:
        preimage[WITHOUT_PARTY_TOTALS] = True
    if enrichment_stamp is not None and requests_kinds(columns):
        preimage["enrichment_stamp"] = enrichment_stamp
    if artifact_digest is not None:
        preimage["artifact_digest"] = artifact_digest
    rendering = rfc8785.dumps(preimage)
    return FrameRef(
        spec_id=spec_id, content_version=f"cv_{hashlib.sha256(rendering).hexdigest()[:32]}"
    )
