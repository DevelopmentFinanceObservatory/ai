"""Reconstructs the mint record of a root frame the store holds none for.

`materialise_and_register` writes a mint record (`common.materialisation.MintRecord`)
on every mint. A root registered before it did has none until it is minted again,
and `recover_mint_record` rebuilds one from the frame's pair, the frame the store
holds under it, and the dataset's versions in dfo-apis.

Recoverable:

- The dfo-apis pair. A frame minted with the default columns is held under the
  dfo-apis pair itself. A frame minted with `options.columns` is held under a
  `content_version` that is a one-way hash of the dfo-apis `content_version` and the
  columns (`frame_ref_for`), so the routine recomputes that hash for each stored
  version of the dataset and each column selection `get_data` can send
  (`RECOVERABLE_COLUMNS`) and takes the one that matches.
- `options.columns`, by the same match.
- `options.groupby` of a summed frame with a `content_version` of its own. dfo-apis
  records the version it was summed from as `base_content_version`, and its
  versions route returns the `groupby` the mint sent whenever a file stored under
  the version records one.
- `scope`, as empty: the dataset's `spec_id` is the identity after any scope was
  applied, so minting it with no scope names the same dataset.
- `stat_var_count`, read off the frame the store holds under the pair.

Not recoverable:

- A summed frame whose `groupby` dfo-apis recorded on no stored file. Its version
  lists as `not_recorded`, and it fails with `groupby_unrecorded`.
- Whether a frame under an unsummed version was a `groupby` sum. dfo-apis serves a
  sum over a dataset one StatVar contributes to under the unsummed
  `content_version`, so a frame there holding one StatVar may have been minted
  either way. It is recovered with no `groupby` and a `stat_var_count` of 1, which
  names the same rows while one StatVar contributes. Once a second contributes,
  `harness.recipe_replay` refuses the root `root_rendering_ambiguous`. A frame
  there holding several StatVars was minted with no `groupby`, since a sum over
  several has a version of its own.
- A frame under an unsummed version that the store no longer holds. Its StatVar
  count cannot be read, so the case above cannot be told apart, and it fails with
  `frame_not_held`.
- A frame whose dataset version dfo-apis no longer lists, or one minted with a
  column selection outside `RECOVERABLE_COLUMNS`. Both fail with
  `no_version_matches`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol

from common.materialisation import (
    FrameExpired,
    FrameRef,
    MaterialisationStore,
    MintRecord,
    MintRecordUnrecoverableError,
)

from data_mcp.frame_materialisation import (
    DEFAULT_MINT_OPTIONS,
    PLACE_TYPE_COLUMNS,
    frame_ref_for,
    observed_shape,
)

__all__ = [
    "RECOVERABLE_COLUMNS",
    "DatasetVersion",
    "DatasetVersions",
    "recover_mint_record",
]


class DatasetVersion(Protocol):
    """One stored version of a dataset, as dfo-apis'
    `GET /api/datasets/{spec_id}/versions` lists it. `base_content_version` is set on
    a version summed from another by a `groupby`, and `groupby` is that mint's
    `options.groupby` when dfo-apis recorded it. `common.live_apis.DatasetVersion`
    is one."""

    @property
    def content_version(self) -> str: ...

    @property
    def base_content_version(self) -> str | None: ...

    @property
    def groupby(self) -> Mapping[str, Any] | None: ...


class DatasetVersions(Protocol):
    """Reads the stored versions of the dataset `spec_id` names, newest first.
    `common.live_apis.LiveApisClient.list_versions` is one."""

    def __call__(self, *, spec_id: str) -> Awaitable[Sequence[DatasetVersion]]: ...


# Every `options.columns` `get_data` sends: none, and the place-type column.
RECOVERABLE_COLUMNS: tuple[tuple[str, ...], ...] = ((), PLACE_TYPE_COLUMNS)


async def recover_mint_record(
    ref: FrameRef, *, dataset_versions: DatasetVersions, store: MaterialisationStore
) -> MintRecord:
    """The mint record of the root frame held under `ref`, rebuilt from the dataset's
    stored versions and the frame `store` holds. See the module docstring for what can
    and cannot be rebuilt.

    The frame is read in full with the store's own row ceiling and no claim on the
    process row budget, once the matching version is found.

    Args:
        ref: A root frame's pair, as the materialisation store holds it.
        dataset_versions: Reads the stored versions of a dataset from dfo-apis.
        store: The materialisation store holding the frame.

    Raises:
        MintRecordUnrecoverableError: With `reason` `no_version_matches` when no stored
            version reproduces `ref` under any recoverable column selection,
            `groupby_unrecorded` when the matching version is a `groupby` sum whose
            `groupby` dfo-apis did not record, and `frame_not_held` when the matching
            version is unsummed and `store` no longer holds the frame.
    """
    for version in await dataset_versions(spec_id=ref.spec_id):
        for columns in RECOVERABLE_COLUMNS:
            candidate = frame_ref_for(
                spec_id=ref.spec_id, content_version=version.content_version, columns=columns
            )
            if candidate != ref:
                continue
            options: dict[str, Any] = dict(DEFAULT_MINT_OPTIONS)
            if version.base_content_version is not None:
                if version.groupby is None:
                    raise MintRecordUnrecoverableError(ref, reason="groupby_unrecorded")
                options["groupby"] = dict(version.groupby)
            if columns:
                options["columns"] = list(columns)
            stat_var_count = await _held_stat_var_count(store, ref)
            if stat_var_count is None and version.base_content_version is None:
                raise MintRecordUnrecoverableError(ref, reason="frame_not_held")
            return MintRecord(
                frame=ref,
                dataset=FrameRef(spec_id=ref.spec_id, content_version=version.content_version),
                options=options,
                stat_var_count=stat_var_count,
            )
    raise MintRecordUnrecoverableError(ref, reason="no_version_matches")


async def _held_stat_var_count(store: MaterialisationStore, ref: FrameRef) -> int | None:
    """The distinct StatVars of the frame `store` holds under `ref`, or `None` when it
    holds none."""
    try:
        held = await store.get(ref)
    except FrameExpired:
        return None
    if held is None:
        return None
    _role_keys, stat_var_count = observed_shape(held[0])
    return stat_var_count
