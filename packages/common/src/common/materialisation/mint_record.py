"""The mint record: how a root frame was minted, kept beside the frame in the
materialisation store as an object of its own.

A root frame is one data-mcp's `get_data` registered from an apis service mint. Its
`MaterialisationTrace` names the pair the frame is held under and nothing about
the request that produced it, and that trace is written inside the frame's
insert-only object, so it can gain no field for a frame the store already holds.
The mint record is written separately, under the same pair, on every `get_data`
call that registers or finds the frame.

A record holds:

- `frame`: the pair the store holds the frame under. For a mint with
  `options.columns`, `options.groupby` or `options.without_party_totals` set this is
  the pair `data_mcp.frame_materialisation.frame_ref_for` derives, whose
  `content_version` is a one-way hash of the apis `content_version`, the
  columns, the groupby and the cut. For any other mint it is `dataset`, and the frame
  is that version's canonical frame.
- `dataset`: the pair apis names the materialisation by, which every
  rendering of it shares.
- `options`: the mint request's `options` as `get_data` sent them. `get_data` sends
  `groupby.by` sorted (`normal_groupby`), as apis records it, and a record decodes
  with the options as written, so a stored recipe root keeps the normal form it was
  published under. A record written before `get_data` sorted `by` keeps its own
  order, and its frame is held under the pair named with that order, which
  `data_mcp.frame_materialisation.named_groupbys` lists. A mint sent with them
  again registers the same rendering, since `frame_ref_for` sorts `by` before it
  names the pair.
- `scope`: the mint request's `scope` as `get_data` sent it. `dataset.spec_id`
  is the identity apis served after applying it, and each scope member
  replaces a whole member of the identity, so sending the same scope with
  `dataset.spec_id` leaves that identity as it is.
- `stat_var_count`: how many distinct StatVars the frame's rows hold, read off
  the frame (`data_mcp.frame_materialisation.observed_shape`). `None` on a record
  written before the count was kept, on one rebuilt (`data_mcp.mint_recovery`)
  for a `groupby` sum with a version of its own whose frame the store no longer
  holds, and on one a bare read wrote (`rendering.record_bare_read`).
- `enrichment_stamp`: the `enrichment_stamp` of the artifact the frame was read from
  (`data_mcp.manifest.ArtifactEntry.enrichment_stamp`) when `options.columns` asks for
  a `{role}_party_kind` column, else `None`. It is also `None` on a record written
  before apis sent one and on a frame read from an artifact without one.
  `frame_ref_for` takes it back to reproduce `frame`.

`encode_mint_record` is the RFC 8785 form of the record, so two writers
recording the same request write the same bytes. `frame`, `dataset`,
`options.columns` and `stat_var_count` are functions of the pair the record is
stored under, the count because a pair names one frame's bytes. Two
requests can still reach one pair with different records: a scope sent with a
base definition and no scope sent with the definition that scope produced.
Either record mints the frame the pair names, and the store keeps the first.

A record written before `frame_ref_for` named a groupby may share its pair with
another: a `groupby` sum and no `groupby` over a dataset one StatVar contributes
to were both held under the version's own pair. That is why the count is kept.
While one StatVar contributes, the summed and the unsummed frame are the same
bytes, and the record says only which request reached the store first. Once a
second StatVar contributes the two differ, so a later mint of a root recorded
with a count of 1 cannot tell which one its frame was. `harness.recipe_replay`
refuses such a root unless its frame is held under the pair that names its
groupby.

`holder.dataset_pair` maps a frame's pair to the dataset pair its record names, the
pair apis holds, for a caller that sends a frame to apis. `rendering.frame_rendering`
answers what a record says about the frame's rows, for a caller that recovers it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.materialisation.frame_ref import FrameRef

MINT_RECORD_SCHEMA_VERSION = 1

# Why a root's mint record could not be rebuilt: no stored version of its dataset
# reproduces the frame's pair, the version that does is a `groupby` sum whose
# `groupby` apis did not record, or the version is unsummed and the store no
# longer holds the frame its StatVar count is read from.
UnrecoverableReason = Literal["no_version_matches", "groupby_unrecorded", "frame_not_held"]


class MintRecordUnrecoverableError(Exception):
    """The store holds no mint record for the root frame `ref`, and none could be
    rebuilt for it. `reason` says why."""

    def __init__(self, ref: FrameRef, *, reason: UnrecoverableReason) -> None:
        self.ref = ref
        self.reason: UnrecoverableReason = reason
        super().__init__(
            f"no mint record is recoverable for {ref.spec_id}, {ref.content_version}: {reason}"
        )


class MintRecord(BaseModel):
    """How the root frame held under `frame` was minted. See the module docstring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = MINT_RECORD_SCHEMA_VERSION
    frame: FrameRef
    dataset: FrameRef
    options: dict[str, Any]
    scope: dict[str, Any] = Field(default_factory=dict)
    stat_var_count: int | None = Field(default=None, ge=0)
    enrichment_stamp: str | None = Field(default=None, exclude_if=lambda stamp: stamp is None)

    @field_validator("options")
    @classmethod
    def _groupby_has_axes(cls, options: dict[str, Any]) -> dict[str, Any]:
        """Refuse a `groupby` whose `by` is no list of axis names. The options are kept as
        written."""
        if (groupby := options.get("groupby")) is not None:
            groupby_axes(groupby)
        return options


def groupby_axes(groupby: object) -> list[str]:
    """The `by` axes of the `options.groupby` `groupby`, in the order it holds them.

    Raises:
        ValueError: When `groupby` is no mapping or its `by` is no list of strings.
    """
    by = groupby.get("by") if isinstance(groupby, Mapping) else None
    if isinstance(by, list):
        axes = [axis for axis in by if isinstance(axis, str)]
        if len(axes) == len(by):
            return axes
    raise ValueError(f"a groupby's `by` must be a list of axis names, not {by!r}")


def normal_groupby(groupby: Mapping[str, Any]) -> dict[str, Any]:
    """`groupby` with its `by` sorted, the form apis records and names a pair by.

    Raises:
        ValueError: When `by` is no list of strings (`groupby_axes`).
    """
    return {**groupby, "by": sorted(groupby_axes(groupby))}


class _LenientMintRecord(MintRecord):
    """The decode shape of a stored record: a member a later deployment added is
    dropped, as `trace_codec` does for a stored trace."""

    model_config = ConfigDict(extra="ignore", frozen=True)


def encode_mint_record(record: MintRecord) -> bytes:
    """The bytes a store writes for `record`: its RFC 8785 form."""
    return rfc8785.dumps(record.model_dump(mode="json"))


def decode_mint_record(data: bytes) -> MintRecord:
    """The record `encode_mint_record` wrote as `data`.

    Raises:
        ValueError: When `data` is not JSON or does not fit the record's shape
            (`pydantic.ValidationError` is a `ValueError`).
    """
    lenient = _LenientMintRecord.model_validate(json.loads(data))
    return MintRecord(**lenient.model_dump())
