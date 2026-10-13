"""GCS-backed `MaterialisationStore`: a `(Frame, StoredFrameTrace)` pair
survives a process restart in one shared bucket, keyed by the reference
pair a `FrameRef` names.

One object per reference: `frames/v4/{spec_id}/{content_version}.parquet`,
uploaded with `if_generation_match=0` -- insert-only, first-writer-wins. A
pair is content-addressed by construction (the reference pair names the
exact frame it names), so a `PreconditionFailed` (412) means the bytes are
already held under this exact name, by whichever writer's write landed
first -- treated as success, not an error. `put` embeds an envelope (the
encoded trace, plus the reference it's stored under) in the object's own
file-level key-value metadata before writing it
(`common.materialisation.frame_codec.frame_to_parquet_bytes`'s own
`metadata` parameter), so the frame and its trace are one write: a `put`
either lands both halves together, under one name, or it lands nothing --
there is no representable state where a frame exists without the trace it
belongs to, or the reverse.

`put` refuses a frame over `max_frame_rows` before writing anything --
insert-only writes mean no later `put` could ever replace an object already
landed under its content-addressed name, so a frame that would fail this
same ceiling on every future `get` is refused up front instead of minting a
reference no read can ever return.

`put` also records the frame's row count in the object's GCS custom
metadata, beside its sha256 and writer, so a reader learns how many rows an
object holds before it downloads a byte of it.

A read verifies, in order:

1. `blob.size` against an encoded-bytes ceiling, and the recorded row count
   against the row ceiling, before `download_as_bytes` runs at all --
   `reload()` already fetches both for free.
2. The downloaded bytes' own sha256 against the digest `put` recorded in
   the object's GCS custom metadata.
3. The Parquet footer's `num_rows` against the recorded row count, then its
   `num_rows`, `num_columns`, and total row-group uncompressed size against
   their own ceilings, and its `num_rows` against the rows the caller claimed
   -- footer-resident, so none of this decodes a column.
   The uncompressed-size ceiling exists because the encoded-bytes ceiling
   alone can be defeated by a small number of highly compressible,
   oversized cells: a single very long string can compress to a small
   on-disk footprint while still fully materialising to its real size the
   moment a column is decoded.
4. The footer's own envelope against the requested `FrameRef` -- a label
   comparison, not a derivation: the envelope names the reference `put` was
   called with, and this checks it against the reference `get` was asked
   for.

A ceiling in steps 1 and 3 raises `FrameTooLarge`, which
`analysis_mcp.frame_ref_resolution.resolve_input_frame` maps to the
retryable `frame_too_large` refusal -- the stored bytes are exactly what was
written, just too large for this store's contract to serve back, and a
narrower `frame_ref` can succeed. A recorded row count that is not a count
or that the footer contradicts, and every failure in steps 2 and 4, raise
`FrameIntegrityError`, mapped to `frame_integrity_failed`'s `do_not_retry`
-- the object itself is wrong, and retrying the identical reference reaches
the identical wrong object.

`open` runs step 1 alone and returns a `HeldFrame` whose `num_rows` is the
recorded row count, so a caller claims the rows (`common.row_budget`)
before any byte is downloaded. An object stored without a recorded count
has `num_rows` `None`, and its caller claims its own row ceiling instead.
The `HeldFrame`'s `read(max_rows=...)` downloads the object, runs steps 2
to 4 with `max_rows` as the claim, and decodes it. `get` is `open` then
`read` with this store's row ceiling as the claim.

Every object this store keeps sits under the layout version `_LAYOUT`
(`v4`): frames under `frames/v4/`, mint records under `mints/v4/` and anchor
records under `anchors/v4/`. A pair can name other bytes from one layout to
the next (under `v4` a dfo-apis pair holds that version's canonical frame,
where it held a rendering before), and a mint or anchor record describes the
frames of the pairs it names, so the records move with the frames.

`frames/v4/` (`_FRAMES_PREFIX`) is the one frame prefix this store writes and
reads. It holds `frame_to_parquet_bytes`'s canonical Arrow layout, whose
`unit` struct is `{ref, request, composed}` with `composed` a composed
unit's JCS text. `frame_from_verified_parquet_bytes` reads it with
`pq.read_table` and wraps the table with `Frame.from_table`, which checks
the schema against the canonical frame schema and trusts the cells.

A flow table minted before dfo-apis renamed its party-kind column holds
`{role}_kind` where a current mint holds `{role}_party_kind`. The bucket keeps
such a frame with no age rule, and every reader looks the column up by its
current name, so a read renames the legacy column once, after the
content-version check (`_with_party_kind_columns`).

An object that passes all four checks and still has another schema raises
`FrameIntegrityError`.

An object under an older prefix (`frames/v3/`, `frames/v2/`, `frames/v1/`,
`mints/v1/`, `anchors/v1/`) is a plain miss for `open`, `get`, `get_trace`,
`get_mint_record` and `get_anchor_record`, and `list_frames` leaves it out, as
it would an object the lifecycle rule has deleted; the bucket's lifecycle rule
deletes it on the schedule it deletes any object. A root whose mint record is
held only under `mints/v1/` therefore has none, as one whose record was lost,
and `data_mcp.mint_recovery` rebuilds it. A
deployment reads only its own prefix, so during a rollout a pair one
deployment stores is a miss for a deployment on another prefix, never
bytes it would decode with the wrong layout, and the caller re-runs the
tool that made it.

`verify_content_version_on_read` (off by default, since it costs a full
recompute on every read) additionally re-derives the frame's own
frame-grammar content_version via the injected `derive_content_version`
callable. For a root reference (`ref.content_version` starts with `cv_`),
that value *is* the frame's own content version, so the recompute is
checked against `ref` itself -- the one comparison a bucket-side tamperer
cannot forge, since `ref` is the caller's own request, never a stored
field. A toolkit result's own `ocv_` reference hashes the op spec and its
inputs, never the frame's bytes, so no such comparison is available for it;
the recompute is checked against the stored trace's own claim instead.

Recorded digests and the envelope itself live in GCS custom metadata and
the object's own Parquet footer -- both are patchable, even on a
generation-locked object, by anyone holding bucket write permission. These
checks catch bit rot, truncated writes, and misfiling; they are not a
defense against an actor with write access to the bucket.

Retention is the bucket's own lifecycle rule alone: this store never
sweeps, and a pair the bucket's own lifecycle rule has deleted comes back
from `open` and `get` as a plain miss (`None`), indistinguishable from a
pair that was never registered. The one `FrameExpired` this store raises
comes from `HeldFrame.read`, for an object deleted after `open` found it.

`get_trace` downloads the object and runs steps 1 to 4, then returns the trace
from the footer's envelope. It decodes no column, so it claims no rows. The
encoded-bytes ceiling bounds what it holds.

A root frame's mint record (`common.materialisation.mint_record`) is one more
object per reference, `mints/v4/{spec_id}/{content_version}.json`, holding
`encode_mint_record`'s bytes. It is uploaded with `if_generation_match=0` like a
frame, and a 412 means a record is already held, which stands. It is written
apart from the frame's object, so a frame the bucket already holds gains one on
its next mint. A stored record that does not decode, or whose `frame` is not the
reference it was read under, raises `FrameIntegrityError`. The bucket's lifecycle
rule deletes it on the schedule it deletes any object.

An anchor record (`common.materialisation.anchor_record`) is one object per
`anchor_key`, `anchors/v4/{anchor_key}.json`, holding `encode_anchor_record`'s
bytes. It is uploaded with `if_generation_match=0`, and a 412 means a record is
already held, which stands. A stored record that does not decode, or whose
populations, roots and rule hash to another key, raises
`AnchorRecordIntegrityError`. The bucket's lifecycle rule deletes it on the
schedule it deletes any object.

`list_frames` lists the object names under `frames/v4/` with each object's creation
time and downloads nothing. A name that is not a frame's is skipped.

`asyncio.to_thread` crosses the synchronous `google-cloud-storage` client
into this async protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import re
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, cast

import pyarrow.parquet as pq
from dfo_frame_grammar import Frame, FrameError
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage
from pydantic import ValidationError

from common.materialisation.anchor_record import (
    AnchorRecord,
    AnchorRecordIntegrityError,
    decode_anchor_record,
    encode_anchor_record,
)
from common.materialisation.frame_codec import (
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
)
from common.materialisation.frame_ref import FrameRef
from common.materialisation.mint_record import MintRecord, decode_mint_record, encode_mint_record
from common.materialisation.store import (
    FrameExpired,
    FrameIntegrityError,
    FrameListing,
    FrameTooLarge,
    HeldFrame,
)
from common.materialisation.trace_codec import decode_trace, encode_trace
from common.place_type import PARTY_ROLES, kind_column

if TYPE_CHECKING:
    from common.materialisation.trace import StoredFrameTrace

_PARQUET_CONTENT_TYPE = "application/octet-stream"
_RECORD_CONTENT_TYPE = "application/json"
_SHA256_METADATA_KEY = "frame_sha256"
_WRITER_METADATA_KEY = "writer"
_ROWS_METADATA_KEY = "frame_rows"
_ENVELOPE_METADATA_KEY = "envelope"

# Every ceiling below is a constructor default, overridable per instance
# (tests inject a small ceiling so a violation doesn't require building an
# object anywhere near real scale), never threaded from a service's own
# settings: more than one service builds a store over the same bucket, and
# per-service ceilings would let them diverge on what the bucket serves back.
# `resolve_input_frame`'s own `settings.max_frame_rows` check is a separate,
# service-level business ceiling on top of this one.

# The default of the `max_frame_rows` setting (`common.FrameServiceSettings`) that
# data-mcp, analysis-mcp and presentation-mcp enforce; a service may set its own
# lower, and claims and reads at that.
_DEFAULT_MAX_FRAME_ROWS = 250_000

# The frame's own column set is small and fixed-shape: the four reserved
# columns (`stat_var`, `time_period`, `value`, `unit`) plus one string
# column per declared role. A legitimate frame carries well under a dozen;
# this leaves generous headroom while still bounding a role set that could
# grow without limit.
_DEFAULT_MAX_COLUMNS = 64

# Bounds the raw bytes `get` downloads before it decodes anything, checked
# against `blob.size` before `download_as_bytes()` ever runs. At
# `_DEFAULT_MAX_FRAME_ROWS` rows, 64 MiB allows 268 bytes a row. A
# 250,000-row, two-role frame encodes to 2.6 MB (`scripts/bench_frames.py`),
# since Parquet dictionary-encodes the repeated text cells, and a row whose
# StatVar, three roles, period, value and unit were all unique text would
# still need about 180 bytes. The envelope (the encoded trace and its
# reference) rides inside this same ceiling rather than needing one of its own.
_DEFAULT_MAX_ENCODED_BYTES = 64 * 1024 * 1024

# Bounds the total uncompressed size Parquet's own footer promises across
# every row group, regardless of how well the object's on-disk bytes
# happen to compress -- see this module's own docstring for why the
# encoded-bytes ceiling alone cannot catch an outsized cell. The footer
# counts encoded pages before compression, 2.7 MB for the same 250,000-row
# frame, so the 268-bytes-a-row allowance above applies here too.
_DEFAULT_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


class _Blob(Protocol):
    metadata: dict[str, str] | None
    size: int | None

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> object: ...
    def download_as_bytes(self) -> bytes: ...
    def reload(self) -> None: ...


class _Bucket(Protocol):
    def blob(self, name: str) -> _Blob: ...


class _ListedBlob(Protocol):
    name: str
    time_created: datetime | None


class _StorageClient(Protocol):
    """The narrow slice of `google.cloud.storage.Client` this module calls.

    `google-cloud-storage`'s own annotations are uneven; a test's fake is
    checked against this Protocol instead of the real client, so calls into
    it don't need scattered ignore comments.
    """

    def bucket(self, name: str) -> _Bucket: ...
    def list_blobs(self, bucket_or_name: str, *, prefix: str) -> Iterable[_ListedBlob]: ...
    def close(self) -> None: ...


# The layout version of every object this store keeps: the canonical frame layout
# `frame_from_verified_parquet_bytes` decodes. Frames and the records that describe
# them share it, so a move to a new frame layout moves the records too.
_LAYOUT = "v4"
_FRAMES_PREFIX = f"frames/{_LAYOUT}"
_MINT_RECORD_PREFIX = f"mints/{_LAYOUT}"
_ANCHOR_RECORD_PREFIX = f"anchors/{_LAYOUT}"


def _with_party_kind_columns(frame: Frame) -> Frame:
    """`frame` with each flow party's `{role}_kind` column renamed `{role}_party_kind`.

    Serves frames minted before the rename, which the bucket keeps with no age rule: the
    old column is an unknown role to `publish_body` and invisible to the party-kind
    checks of the analysis verbs. A frame holding no such column is returned as is. The
    rename keeps the column order derived from the role keys and every row where it
    was, since a party's kind follows from the party.

    Raises:
        FrameError: `schema_violation` when the renamed columns are not a derived frame
            column order.
    """
    renames = {
        f"{role}_kind": kind_column(role) for role in PARTY_ROLES if f"{role}_kind" in frame.columns
    }
    if not renames:
        return frame
    columns = tuple(renames.get(column, column) for column in frame.columns)
    return Frame.from_table(columns=columns, table=frame.table.rename_columns(list(columns)))


# The name of a frame's object. `FrameRef` decides whether the two ids are a frame's.
_FRAME_OBJECT_NAME = re.compile(
    re.escape(_FRAMES_PREFIX) + r"/(?P<spec_id>[^/]+)/(?P<content_version>[^/]+)\.parquet"
)


def _frame_object_name(ref: FrameRef) -> str:
    return f"{_FRAMES_PREFIX}/{ref.spec_id}/{ref.content_version}.parquet"


def _mint_record_object_name(ref: FrameRef) -> str:
    return f"{_MINT_RECORD_PREFIX}/{ref.spec_id}/{ref.content_version}.json"


def _anchor_record_object_name(anchor_key: str) -> str:
    return f"{_ANCHOR_RECORD_PREFIX}/{anchor_key}.json"


class GcsMaterialisationStore:
    """`google-cloud-storage`-backed `MaterialisationStore`.

    `client` is injected, never constructed here, so a test can run without
    ever reaching a real bucket -- a bare `storage.Client()` reaches the
    metadata server for Application Default Credentials. `bucket_name` is
    read from config by the caller constructing this, never hardcoded.
    """

    def __init__(
        self,
        *,
        bucket_name: str,
        client: storage.Client | None = None,
        max_frame_rows: int = _DEFAULT_MAX_FRAME_ROWS,
        max_columns: int = _DEFAULT_MAX_COLUMNS,
        max_encoded_bytes: int = _DEFAULT_MAX_ENCODED_BYTES,
        max_uncompressed_bytes: int = _DEFAULT_MAX_UNCOMPRESSED_BYTES,
        verify_content_version_on_read: bool = False,
        derive_content_version: Callable[[Frame], str] | None = None,
    ) -> None:
        if verify_content_version_on_read and derive_content_version is None:
            raise ValueError(
                "verify_content_version_on_read=True requires a derive_content_version callable"
            )
        self._bucket_name = bucket_name
        self._max_frame_rows = max_frame_rows
        self._max_columns = max_columns
        self._max_encoded_bytes = max_encoded_bytes
        self._max_uncompressed_bytes = max_uncompressed_bytes
        # `storage.Client`'s own methods carry no annotations, so `ty` can't
        # verify it satisfies `_StorageClient` structurally -- it does at
        # runtime, exercised by every test in this module that injects a
        # fake in its place. One cast here beats an ignore comment on every
        # call below.
        self._client = cast("_StorageClient", client if client is not None else storage.Client())
        self._verify_content_version_on_read = verify_content_version_on_read
        self._derive_content_version = derive_content_version

    async def put(
        self, ref: FrameRef, *, frame: Frame, trace: StoredFrameTrace, writer: str
    ) -> None:
        await asyncio.to_thread(self._put_one, ref, frame, trace, writer)

    def _put_one(self, ref: FrameRef, frame: Frame, trace: StoredFrameTrace, writer: str) -> None:
        # Checked before anything is written, not just on read: a frame over
        # this ceiling would otherwise land under a content-addressed name
        # that a later `get` can only ever refuse -- insert-only writes mean
        # no later `put` could replace it, so the reference would be
        # unreadable forever. Only the row ceiling is checked here (not
        # columns/encoded/uncompressed bytes) -- those bound the *encoded
        # object*, cheap to check only once it exists; a frame's row count is
        # known before encoding, so it's the one ceiling worth refusing on
        # up front.
        if frame.num_rows > self._max_frame_rows:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"frame has {frame.num_rows} rows, over this store's "
                    f"{self._max_frame_rows}-row ceiling"
                ),
            )
        bucket = self._client.bucket(self._bucket_name)

        envelope = {
            "ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
            "trace": encode_trace(trace),
        }
        frame_bytes = frame_to_parquet_bytes(
            frame, metadata={_ENVELOPE_METADATA_KEY: json.dumps(envelope).encode("utf-8")}
        )
        blob = bucket.blob(_frame_object_name(ref))
        blob.metadata = {
            _SHA256_METADATA_KEY: hashlib.sha256(frame_bytes).hexdigest(),
            _WRITER_METADATA_KEY: writer,
            _ROWS_METADATA_KEY: str(frame.num_rows),
        }
        # The object already existing under this content-addressed name
        # means someone else's write won the race -- success, not an error.
        # One insert-only upload carries the frame and its trace together,
        # so there is no partial state a second writer's failed upload
        # could leave: this object either exists complete, or not at all.
        with contextlib.suppress(PreconditionFailed):
            blob.upload_from_string(
                frame_bytes, content_type=_PARQUET_CONTENT_TYPE, if_generation_match=0
            )

    async def get(self, ref: FrameRef) -> tuple[Frame, StoredFrameTrace] | None:
        held = await self.open(ref)
        return None if held is None else await held.read(max_rows=self._max_frame_rows)

    async def open(self, ref: FrameRef) -> HeldFrame | None:
        return await asyncio.to_thread(self._open_one, ref)

    def _open_one(self, ref: FrameRef) -> HeldFrame | None:
        blob = self._client.bucket(self._bucket_name).blob(_frame_object_name(ref))
        try:
            # `reload()` fetches the object's resource -- size and custom
            # metadata -- from the bucket; neither is populated on a freshly
            # constructed `Blob`. Fetching it first is what lets step 1 run
            # before a single byte of the object itself is downloaded.
            blob.reload()
        except NotFound:
            return None
        recorded_rows = self._admit(ref, blob)
        return self._held(ref, blob, recorded_rows)

    def _admit(self, ref: FrameRef, blob: _Blob) -> int | None:
        """Step 1 of this module's docstring: the row count `put` recorded, or `None`
        for an object stored without one."""
        if blob.size is not None and blob.size > self._max_encoded_bytes:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"stored object is {blob.size} bytes, over the "
                    f"{self._max_encoded_bytes}-byte ceiling"
                ),
            )
        recorded = (blob.metadata or {}).get(_ROWS_METADATA_KEY)
        if recorded is None:
            return None
        if not re.fullmatch(r"[0-9]+", recorded):
            raise FrameIntegrityError(
                ref, reason=f"stored object's recorded row count {recorded!r} is not a count"
            )
        recorded_rows = int(recorded)
        if recorded_rows > self._max_frame_rows:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"stored frame has {recorded_rows} rows, over this "
                    f"store's {self._max_frame_rows}-row ceiling"
                ),
            )
        return recorded_rows

    def _held(self, ref: FrameRef, blob: _Blob, recorded_rows: int | None) -> HeldFrame:
        async def read_in_thread(*, max_rows: int) -> tuple[Frame, StoredFrameTrace]:
            return await asyncio.to_thread(self._read, ref, blob, recorded_rows, max_rows)

        return HeldFrame(num_rows=recorded_rows, read=read_in_thread)

    def _read(
        self,
        ref: FrameRef,
        blob: _Blob,
        recorded_rows: int | None,
        max_rows: int,
    ) -> tuple[Frame, StoredFrameTrace]:
        """Steps 2 to 4 of this module's docstring on the downloaded object, with
        `max_rows` as the caller's claim, then the decode."""
        try:
            frame_bytes = blob.download_as_bytes()
        except NotFound as exc:
            raise FrameExpired(ref) from exc
        trace = self._verify(ref, blob, frame_bytes, recorded_rows, max_rows)
        return self._decode(ref, frame_bytes, trace), trace

    def _verify(
        self,
        ref: FrameRef,
        blob: _Blob,
        frame_bytes: bytes,
        recorded_rows: int | None,
        max_rows: int,
    ) -> StoredFrameTrace:
        """Steps 2 to 4 of this module's docstring: the stored frame's trace, read from
        its footer without decoding a column."""
        recorded_digest = (blob.metadata or {}).get(_SHA256_METADATA_KEY)
        if recorded_digest != hashlib.sha256(frame_bytes).hexdigest():
            raise FrameIntegrityError(
                ref, reason="stored object bytes do not match the recorded sha256"
            )

        # The digest check above only ever hashes raw bytes, so it still
        # catches a corrupted or tampered object even when it isn't valid
        # Parquet at all; only past that point is `frame_bytes` trusted
        # enough to open as a `ParquetFile` for its footer metadata -- row
        # count, column count, and per-row-group uncompressed size, none of
        # which requires decoding a single column.
        parquet_file = pq.ParquetFile(io.BytesIO(frame_bytes))
        file_metadata = parquet_file.metadata
        if recorded_rows is not None and file_metadata.num_rows != recorded_rows:
            raise FrameIntegrityError(
                ref,
                reason=(
                    f"stored frame has {file_metadata.num_rows} rows, not the "
                    f"{recorded_rows} its metadata records"
                ),
            )
        if file_metadata.num_rows > self._max_frame_rows:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"stored frame has {file_metadata.num_rows} rows, over this "
                    f"store's {self._max_frame_rows}-row ceiling"
                ),
            )
        if file_metadata.num_rows > max_rows:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"stored frame has {file_metadata.num_rows} rows, over this "
                    f"read's {max_rows}-row limit"
                ),
            )
        if file_metadata.num_columns > self._max_columns:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"stored frame has {file_metadata.num_columns} columns, over "
                    f"this store's {self._max_columns}-column ceiling"
                ),
            )
        uncompressed_bytes = sum(
            file_metadata.row_group(group).column(column).total_uncompressed_size
            for group in range(file_metadata.num_row_groups)
            for column in range(file_metadata.row_group(group).num_columns)
        )
        if uncompressed_bytes > self._max_uncompressed_bytes:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"stored frame decodes to {uncompressed_bytes} uncompressed bytes, "
                    f"over this store's {self._max_uncompressed_bytes}-byte ceiling -- a "
                    "small on-disk footprint does not bound this"
                ),
            )

        # File-level key-value metadata lives in the same footer the checks
        # above already read -- `put` attached it to the table's own schema
        # before writing it (`frame_to_parquet_bytes`'s own `metadata`
        # parameter), so it's available here without decoding a column
        # either. Its own length is checked against the same ceiling the
        # whole object was already checked against above, since it can
        # never legitimately exceed it.
        schema_metadata = parquet_file.schema_arrow.metadata or {}
        envelope_bytes = schema_metadata.get(_ENVELOPE_METADATA_KEY.encode("utf-8"))
        if envelope_bytes is None or len(envelope_bytes) > self._max_encoded_bytes:
            raise FrameIntegrityError(ref, reason="stored object carries no valid envelope")
        envelope = json.loads(envelope_bytes)
        envelope_ref = envelope.get("ref", {})
        if (
            envelope_ref.get("spec_id") != ref.spec_id
            or envelope_ref.get("content_version") != ref.content_version
        ):
            raise FrameIntegrityError(
                ref, reason="stored object's envelope names a different reference"
            )
        return decode_trace(envelope["trace"])

    def _decode(self, ref: FrameRef, frame_bytes: bytes, trace: StoredFrameTrace) -> Frame:
        """The frame `frame_bytes` holds, which `_verify` has already checked. The
        decoder trusts the rows, so with `verify_content_version_on_read` on, rows out
        of order or held twice refuse when the recompute hashes them."""
        try:
            frame = frame_from_verified_parquet_bytes(frame_bytes)
            recomputed = None
            if self._verify_content_version_on_read:
                assert self._derive_content_version is not None  # enforced in __init__
                recomputed = self._derive_content_version(frame)
        except FrameError as exc:
            raise FrameIntegrityError(
                ref, reason=f"stored object does not decode to a frame: {exc}"
            ) from exc
        if self._verify_content_version_on_read:
            # For a root reference, `ref.content_version` IS the frame's own content
            # version, so the recompute is checked against `ref`. An `ocv_` reference
            # has no such comparison available -- see this module's own docstring.
            expected = (
                ref.content_version
                if ref.content_version.startswith("cv_")
                else trace.frame_content_version
            )
            if recomputed != expected:
                raise FrameIntegrityError(
                    ref, reason="recomputed content_version does not match the expected value"
                )
        try:
            return _with_party_kind_columns(frame)
        except FrameError as exc:
            raise FrameIntegrityError(
                ref, reason=f"stored object's legacy party-kind column does not rename: {exc}"
            ) from exc

    async def get_trace(self, ref: FrameRef) -> StoredFrameTrace | None:
        return await asyncio.to_thread(self._trace_of, ref)

    def _trace_of(self, ref: FrameRef) -> StoredFrameTrace | None:
        blob = self._client.bucket(self._bucket_name).blob(_frame_object_name(ref))
        try:
            blob.reload()
        except NotFound:
            return None
        recorded_rows = self._admit(ref, blob)
        try:
            frame_bytes = blob.download_as_bytes()
        except NotFound as exc:
            raise FrameExpired(ref) from exc
        return self._verify(ref, blob, frame_bytes, recorded_rows, self._max_frame_rows)

    async def list_frames(self) -> list[FrameListing]:
        return await asyncio.to_thread(self._list_frames)

    def _list_frames(self) -> list[FrameListing]:
        listings: list[FrameListing] = []
        for blob in self._client.list_blobs(self._bucket_name, prefix=f"{_FRAMES_PREFIX}/"):
            named = _FRAME_OBJECT_NAME.fullmatch(blob.name)
            if named is None or blob.time_created is None:
                continue
            try:
                ref = FrameRef(**named.groupdict())
            except ValidationError:
                continue
            listings.append(FrameListing(ref=ref, created_at=blob.time_created))
        return sorted(
            listings,
            key=lambda listing: (
                listing.created_at,
                listing.ref.spec_id,
                listing.ref.content_version,
            ),
        )

    async def put_mint_record(self, record: MintRecord) -> None:
        await asyncio.to_thread(self._put_mint_record, record)

    def _put_mint_record(self, record: MintRecord) -> None:
        blob = self._client.bucket(self._bucket_name).blob(_mint_record_object_name(record.frame))
        # A record already under this name stands (see the module docstring).
        with contextlib.suppress(PreconditionFailed):
            blob.upload_from_string(
                encode_mint_record(record),
                content_type=_RECORD_CONTENT_TYPE,
                if_generation_match=0,
            )

    async def get_mint_record(self, ref: FrameRef) -> MintRecord | None:
        return await asyncio.to_thread(self._mint_record_of, ref)

    def _mint_record_of(self, ref: FrameRef) -> MintRecord | None:
        blob = self._client.bucket(self._bucket_name).blob(_mint_record_object_name(ref))
        try:
            data = blob.download_as_bytes()
        except NotFound:
            return None
        try:
            record = decode_mint_record(data)
        except ValueError as exc:
            raise FrameIntegrityError(
                ref, reason=f"stored mint record does not decode: {exc}"
            ) from exc
        if record.frame != ref:
            raise FrameIntegrityError(ref, reason="stored mint record names a different reference")
        return record

    async def put_anchor_record(self, record: AnchorRecord) -> None:
        await asyncio.to_thread(self._put_anchor_record, record)

    def _put_anchor_record(self, record: AnchorRecord) -> None:
        blob = self._client.bucket(self._bucket_name).blob(
            _anchor_record_object_name(record.anchor_key)
        )
        # A record already under this name stands (see the module docstring).
        with contextlib.suppress(PreconditionFailed):
            blob.upload_from_string(
                encode_anchor_record(record),
                content_type=_RECORD_CONTENT_TYPE,
                if_generation_match=0,
            )

    async def get_anchor_record(self, anchor_key: str) -> AnchorRecord | None:
        return await asyncio.to_thread(self._anchor_record_of, anchor_key)

    def _anchor_record_of(self, anchor_key: str) -> AnchorRecord | None:
        blob = self._client.bucket(self._bucket_name).blob(_anchor_record_object_name(anchor_key))
        try:
            data = blob.download_as_bytes()
        except NotFound:
            return None
        try:
            record = decode_anchor_record(data)
        except ValueError as exc:
            raise AnchorRecordIntegrityError(
                anchor_key, reason=f"stored anchor record does not decode: {exc}"
            ) from exc
        if record.anchor_key != anchor_key:
            raise AnchorRecordIntegrityError(
                anchor_key, reason="stored anchor record hashes to a different key"
            )
        return record

    async def aclose(self) -> None:
        await asyncio.to_thread(self._client.close)
