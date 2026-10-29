"""Reconstructs the mint record of a root frame the store holds none for.

`materialise_and_register` writes a mint record (`common.materialisation.MintRecord`)
on every mint. A root registered before it did has none until it is minted again,
and `recover_mint_record` rebuilds one from the frame's pair, the frame the store
holds under it, and the dataset's versions in apis.

Recoverable:

- The apis pair. A frame minted with the default options is held under the
  apis pair itself. A frame minted with `options.columns`, `options.groupby` or
  `options.without_party_totals` is held under a `content_version` that is a one-way
  hash of the apis `content_version`, the columns, the groupby and the cut
  (`frame_ref_for`), so the routine recomputes that hash for each stored version of
  the dataset, each column selection `get_data` can send (`RECOVERABLE_COLUMNS`), and
  no groupby or the one apis recorded for the version, and takes the one that matches.
  A groupby is tried with its `by` in each order `named_groupbys` lists, since a frame
  registered before `frame_ref_for` sorted `by` is named with the order get_data sent,
  and the record keeps the order that matched.
- `options.columns`, by the same match.
- `enrichment_stamp` of a frame minted with `KIND_COLUMNS`: the stamp its pair is named
  under. Each candidate above is tried under the stamp the versions answer lists on its
  version and under none, the pair of a frame minted before apis sent a stamp or of one
  whose frame holds no party role, which recovers with none. A frame named under
  another stamp is matched from its held trace, which keeps the stamp
  (`MaterialisationTrace.enrichment_stamp`), as in the next entry. A record under a
  stamp other than the one its version lists marks the
  root as moved to `harness.live_refresh`, which replays it with the kinds served now.
- A sum that kept its version, held under the pair that names its groupby. apis
  records no groupby on an unsummed version, so no candidate above reproduces the
  pair. The frame's own trace keeps the options it was minted with
  (`MaterialisationTrace.options`) and the stamp its pair is named under, and the
  pair `frame_ref_for` derives from them and a listed version is the match. The
  record holds those options and that stamp. A trace that keeps no stamp is tried under
  the stamp each version lists and under none.
- `options.groupby` of a summed frame with a `content_version` of its own. apis
  records the version it was summed from as `base_content_version`, and its
  versions route returns the `groupby` the mint sent whenever a file stored under
  the version records one. A frame registered before `frame_ref_for` named a
  groupby is held under the summed version's own pair, and one registered since
  under the pair that names it; both are recovered.
- `options.without_party_totals` of a frame without the group totals of its open
  party roles. apis mints such a frame a version of its own, with the version it
  was cut from as `base_content_version`, and its versions route says so beside any
  `groupby` (`DatasetVersion.without_party_totals`). A cut that left nothing out
  shares its base's version, which lists as plain. Its pair names the cut
  (`frame_ref_for`), so each candidate above is also tried with the cut, and a match
  recovers with the option. A frame registered before the pair named the cut is held
  under the pair without it and recovers with the option its version records, or
  without it on a plain version, which names the same rows while no total is left
  out.
- `scope`, as empty: the dataset's `spec_id` is the identity after any scope was
  applied, so minting it with no scope names the same dataset.
- `stat_var_count`, read off the frame the store holds under the pair.

Not recoverable:

- A summed or cut frame whose options apis recorded on no stored file. Its
  version lists as `not_recorded`, and it fails with `groupby_unrecorded`, as does a
  summed version whose recorded `groupby.by` is no list of axis names.
- Whether a frame under an unsummed version was a `groupby` sum. apis serves a
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
- A sum that kept its version whose frame the store no longer holds, or holds with a
  trace written before it kept the mint's options. It fails with
  `no_version_matches`.
- A frame named under another stamp whose frame the store no longer holds, or holds
  with a trace that keeps no stamp. It fails with `no_version_matches`.
- A frame whose dataset version apis no longer lists, or one minted with a
  column selection outside `RECOVERABLE_COLUMNS`. Both fail with
  `no_version_matches`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from itertools import combinations, product
from typing import Any, Protocol

from common.materialisation import (
    FrameExpired,
    FrameRef,
    MaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    MintRecordUnrecoverableError,
    groupby_axes,
)
from dfo_frame_grammar import Frame

from data_mcp.frame_materialisation import (
    DEFAULT_MINT_OPTIONS,
    FLOW_ENRICHMENT_COLUMNS,
    KIND_COLUMNS,
    TYPE_COLUMN_SELECTIONS,
    WITHOUT_PARTY_TOTALS,
    frame_ref_for,
    named_groupbys,
    observed_shape,
    recorded_stamp,
)

__all__ = [
    "RECOVERABLE_COLUMNS",
    "DatasetVersion",
    "DatasetVersions",
    "recover_mint_record",
]


class DatasetVersion(Protocol):
    """One stored version of a dataset, as apis'
    `GET /api/datasets/{spec_id}/versions` lists it. `base_content_version` is set on
    a version summed from another by a `groupby` or cut from it by
    `without_party_totals`. `groupby` is that mint's `options.groupby` when apis
    recorded it, and `without_party_totals` is true when apis recorded the cut.
    `enrichment_stamp` is the stamp a party-kind file of the version carries, the
    answering code index's digest, or `None` when its frame holds no party role or apis
    lists none. `common.live_apis.DatasetVersion` is one."""

    @property
    def content_version(self) -> str: ...

    @property
    def base_content_version(self) -> str | None: ...

    @property
    def groupby(self) -> Mapping[str, Any] | None: ...

    @property
    def without_party_totals(self) -> bool: ...

    @property
    def enrichment_stamp(self) -> str | None: ...


class DatasetVersions(Protocol):
    """Reads the stored versions of the dataset `spec_id` names, newest first.
    `common.live_apis.LiveApisClient.list_versions` is one."""

    def __call__(self, *, spec_id: str) -> Awaitable[Sequence[DatasetVersion]]: ...


# The kind tokens of a flow table's mint before `KIND_COLUMNS`: `party_kind` before the
# class and whole tokens joined it. A root minted with it stays in the frame store with no
# age rule, and `replayed_mint_options` upgrades the recovered selection on replay.
_EARLIER_KIND_SELECTIONS: tuple[tuple[str, ...], ...] = (("party_kind",),)

# The party type columns a flow mint may send beside its kinds: none, both, or the one left
# after a column refusal drops a role the table does not have, in `FLOW_ENRICHMENT_COLUMNS`
# order.
_PARTY_TYPE_COLUMNS = tuple(
    column for column in FLOW_ENRICHMENT_COLUMNS if column not in KIND_COLUMNS
)
_PARTY_TYPE_SELECTIONS: tuple[tuple[str, ...], ...] = tuple(
    selection
    for size in range(len(_PARTY_TYPE_COLUMNS) + 1)
    for selection in combinations(_PARTY_TYPE_COLUMNS, size)
)

# Every `options.columns` a recorded mint may hold: none, some of the type columns, a flow
# table's kinds under the current or the earlier tokens with any of its party types before
# them, or the earliest token `kind` alone.
RECOVERABLE_COLUMNS: tuple[tuple[str, ...], ...] = (
    *TYPE_COLUMN_SELECTIONS,
    *(
        (*types, *kinds)
        for kinds in (KIND_COLUMNS, *_EARLIER_KIND_SELECTIONS)
        for types in _PARTY_TYPE_SELECTIONS
    ),
    ("kind",),
)


async def recover_mint_record(
    ref: FrameRef,
    *,
    dataset_versions: DatasetVersions,
    store: MaterialisationStore,
) -> MintRecord:
    """The mint record of the root frame held under `ref`, rebuilt from the dataset's
    stored versions and the frame `store` holds. See the module docstring for what can
    and cannot be rebuilt.

    The frame is read in full with the store's own row ceiling and no claim on the
    process row budget, once the matching version is found, or when no candidate
    matches and the held frame's trace is the last source of its options.

    Args:
        ref: A root frame's pair, as the materialisation store holds it.
        dataset_versions: Reads the stored versions of a dataset from apis.
        store: The materialisation store holding the frame.

    Raises:
        MintRecordUnrecoverableError: With `reason` `no_version_matches` when no stored
            version reproduces `ref` under any recoverable column selection,
            `groupby_unrecorded` when the matching version is a `groupby` sum or a
            party-totals cut whose options apis did not record, and `frame_not_held` when the matching
            version is unsummed and `store` no longer holds the frame.
    """
    versions = await dataset_versions(spec_id=ref.spec_id)
    for version in versions:
        recorded = _recorded_groupby(version)
        groupbys = (None,) if recorded is None else (None, *named_groupbys(recorded))
        for groupby, cut, columns, stamp in product(
            groupbys, (False, True), RECOVERABLE_COLUMNS, _listed_or_none(version)
        ):
            candidate = frame_ref_for(
                spec_id=ref.spec_id,
                content_version=version.content_version,
                columns=columns,
                groupby=groupby,
                without_party_totals=cut,
                enrichment_stamp=stamp,
                keep_by_order=True,
            )
            if candidate != ref:
                continue
            options: dict[str, Any] = dict(DEFAULT_MINT_OPTIONS)
            if groupby is not None:
                options["groupby"] = groupby
            elif version.base_content_version is not None:
                if recorded is None and not version.without_party_totals:
                    raise MintRecordUnrecoverableError(ref, reason="groupby_unrecorded")
                if recorded is not None:
                    options["groupby"] = dict(recorded)
            if cut or version.without_party_totals:
                options[WITHOUT_PARTY_TOTALS] = True
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
                enrichment_stamp=recorded_stamp(columns, stamp),
            )
    return await _recover_from_trace(ref, versions=versions, store=store)


def _recorded_groupby(version: DatasetVersion) -> Mapping[str, Any] | None:
    """The `groupby` apis recorded for `version`, or `None` when it recorded none or one
    whose `by` is no list of axis names, which names no pair."""
    if version.groupby is None:
        return None
    try:
        groupby_axes(version.groupby)
    except ValueError:
        return None
    return version.groupby


def _named_cuts(options: Mapping[str, Any]) -> tuple[bool, ...]:
    """Whether a frame minted with `options` may be named by its party-totals cut: by the cut
    it asks for, and without it, the pair of a frame registered before `frame_ref_for` named
    the cut."""
    return tuple(dict.fromkeys((options.get(WITHOUT_PARTY_TOTALS) is True, False)))


def _listed_or_none(version: DatasetVersion) -> tuple[str | None, ...]:
    """The stamps a frame of `version` may be named under when nothing records its own:
    the one the version lists, and none, for a frame minted before apis sent one."""
    return tuple(dict.fromkeys((version.enrichment_stamp, None)))


async def _recover_from_trace(
    ref: FrameRef, *, versions: Sequence[DatasetVersion], store: MaterialisationStore
) -> MintRecord:
    """The mint record of `ref` rebuilt from the options and stamp its held frame's
    trace keeps, matched against `versions` by `frame_ref_for`. A trace that keeps no
    stamp is tried under each version's `_listed_or_none`."""
    held = await _held(store, ref)
    trace = None if held is None else held[1]
    if held is None or not isinstance(trace, MaterialisationTrace) or trace.options is None:
        raise MintRecordUnrecoverableError(ref, reason="no_version_matches")
    options = trace.options
    try:
        groupbys = (None,) if options.get("groupby") is None else named_groupbys(options["groupby"])
    except ValueError as exc:
        raise MintRecordUnrecoverableError(ref, reason="no_version_matches") from exc
    for version in versions:
        stamps = (
            (trace.enrichment_stamp,)
            if trace.enrichment_stamp is not None
            else _listed_or_none(version)
        )
        for stamp, cut, groupby in product(stamps, _named_cuts(options), groupbys):
            candidate = frame_ref_for(
                spec_id=ref.spec_id,
                content_version=version.content_version,
                columns=tuple(options.get("columns") or ()),
                groupby=groupby,
                without_party_totals=cut,
                enrichment_stamp=stamp,
                keep_by_order=True,
            )
            if candidate == ref:
                return MintRecord(
                    frame=ref,
                    dataset=FrameRef(spec_id=ref.spec_id, content_version=version.content_version),
                    options=dict(options),
                    stat_var_count=observed_shape(held[0])[1],
                    enrichment_stamp=recorded_stamp(options.get("columns") or (), stamp),
                )
    raise MintRecordUnrecoverableError(ref, reason="no_version_matches")


async def _held(store: MaterialisationStore, ref: FrameRef) -> tuple[Frame, Any] | None:
    """The frame and trace `store` holds under `ref`, or `None` when it holds none."""
    try:
        return await store.get(ref)
    except FrameExpired:
        return None


async def _held_stat_var_count(store: MaterialisationStore, ref: FrameRef) -> int | None:
    """The distinct StatVars of the frame `store` holds under `ref`, or `None` when it
    holds none."""
    held = await _held(store, ref)
    if held is None:
        return None
    _role_keys, stat_var_count = observed_shape(held[0])
    return stat_var_count
