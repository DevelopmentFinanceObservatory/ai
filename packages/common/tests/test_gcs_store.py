"""Tests for `GcsMaterialisationStore` against an injected fake client --
never the real `google.cloud.storage.Client()`, whose construction can
reach the metadata server for Application Default Credentials.

Hand-written fakes, structurally typed against this module's own narrow
Protocols, backed by plain dicts.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.materialisation import (
    AnchorRecord,
    AnchorRecordIntegrityError,
    FrameExpired,
    FrameIntegrityError,
    FrameListing,
    FrameRef,
    FrameTooLarge,
    GcsMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    encode_anchor_record,
    encode_mint_record,
    encode_trace,
    frame_to_parquet_bytes,
    gcs_store,
)
from dfo_frame_grammar import (
    Frame,
    canonical_frame_bytes,
    canonical_frame_schema,
    frame_sha256,
    unit_cell_to_json,
)
from google.api_core.exceptions import NotFound, PreconditionFailed

_BUCKET = "dev-fin-observatory-frames"
_LEGACY_DIR = Path(__file__).parent / "fixtures" / "legacy_parquet"
_WRITTEN_PREFIX = "frames/v3"


class _FakeBlob:
    def __init__(
        self, objects: dict[str, bytes], metadata_store: dict[str, dict[str, str]], name: str
    ) -> None:
        self._objects = objects
        self._metadata_store = metadata_store
        self._name = name
        self.metadata: dict[str, str] | None = None
        self.size: int | None = None

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert if_generation_match == 0
        if self._name in self._objects:
            raise PreconditionFailed("already exists")
        self._objects[self._name] = data
        self._metadata_store[self._name] = dict(self.metadata or {})

    def download_as_bytes(self) -> bytes:
        if self._name not in self._objects:
            raise NotFound(f"no such object: {self._name}")
        return self._objects[self._name]

    def reload(self) -> None:
        if self._name not in self._objects:
            raise NotFound(f"no such object: {self._name}")
        self.metadata = dict(self._metadata_store.get(self._name, {}))
        self.size = len(self._objects[self._name])


@dataclass(frozen=True)
class _FakeListedBlob:
    name: str
    time_created: datetime | None


class _FakeBucket:
    def __init__(
        self, objects: dict[str, bytes], metadata_store: dict[str, dict[str, str]]
    ) -> None:
        self._objects = objects
        self._metadata_store = metadata_store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._objects, self._metadata_store, name)


class _FakeStorageClient:
    """Fakes the narrow `_StorageClient` seam `gcs_store.py` calls through.

    `objects`/`metadata` are plain dicts, exposed so a test can pass the
    same pair to a second, independently constructed fake client -- standing
    in for a second store instance reading what a first instance, before a
    restart, wrote.
    """

    def __init__(
        self,
        objects: dict[str, bytes] | None = None,
        metadata: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.objects: dict[str, bytes] = objects if objects is not None else {}
        self.metadata: dict[str, dict[str, str]] = metadata if metadata is not None else {}
        self.created: dict[str, datetime] = {}
        self.closed = False

    def bucket(self, name: str) -> _FakeBucket:
        assert name == _BUCKET
        return _FakeBucket(self.objects, self.metadata)

    def list_blobs(self, bucket_or_name: str, *, prefix: str) -> list[_FakeListedBlob]:
        """Every object under `prefix`, in name order as a bucket lists them, created
        at the instant `created` gives its name."""
        assert bucket_or_name == _BUCKET
        return [
            _FakeListedBlob(name=name, time_created=self.created.get(name))
            for name in sorted(self.objects)
            if name.startswith(prefix)
        ]

    def close(self) -> None:
        self.closed = True


def _ref() -> FrameRef:
    return FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)


def _frame() -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 100.5,
                "unit": "dcid:USDollar",
            }
        ],
    )


def _two_row_frame() -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "time_period": str(2020 + offset),
                "value": 100.5,
                "unit": "dcid:USDollar",
            }
            for offset in range(2)
        ],
    )


def _trace(ref: FrameRef) -> MaterialisationTrace:
    return MaterialisationTrace(
        upstream_trace={"graph_revision": "gr_1"},
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )


async def test_put_then_get_round_trips_the_frame_and_trace() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    frame = _frame()
    trace = _trace(ref)

    await store.put(ref, frame=frame, trace=trace, writer="writer-1")
    result = await store.get(ref)

    assert result is not None
    round_tripped_frame, round_tripped_trace = result
    assert canonical_frame_bytes(round_tripped_frame) == canonical_frame_bytes(frame)
    assert round_tripped_trace == trace


async def test_open_reads_the_recorded_row_count_without_downloading_and_read_returns_the_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller claims the rows `open` reports before any byte of the frame is downloaded."""
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=_FakeStorageClient())
    ref = _ref()
    frame = _frame()
    await store.put(ref, frame=frame, trace=_trace(ref), writer="writer-1")
    downloads: list[str] = []
    download = _FakeBlob.download_as_bytes

    def _counting_download(self: _FakeBlob) -> bytes:
        downloads.append(self._name)
        return download(self)

    monkeypatch.setattr(_FakeBlob, "download_as_bytes", _counting_download)

    held = await store.open(ref)

    assert held is not None
    assert (held.num_rows, downloads) == (1, [])
    read_frame, read_trace = await held.read(max_rows=1)
    assert canonical_frame_bytes(read_frame) == canonical_frame_bytes(frame)
    assert read_trace == _trace(ref)
    assert downloads == [_object_name(ref, prefix=_WRITTEN_PREFIX)]


async def test_put_records_the_frames_row_count_beside_its_digest_and_writer() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()

    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    assert client.metadata[_object_name(ref, prefix=_WRITTEN_PREFIX)]["frame_rows"] == "1"


async def test_open_reports_no_row_count_for_an_object_stored_without_one() -> None:
    """An object written before `put` recorded a row count reads back under any claim
    its footer fits."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client, max_frame_rows=7)
    ref = _ref()
    _store_table(client, _object_name(ref, prefix=_WRITTEN_PREFIX), _frame().table, ref)

    held = await store.open(ref)

    assert held is not None
    assert held.num_rows is None
    read_frame, _ = await held.read(max_rows=1)
    assert read_frame.num_rows == 1


async def test_read_refuses_a_footer_over_the_claim_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that claimed fewer rows than an object without a recorded count holds
    is refused with the footer's count, and nothing decodes the object."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    _store_table(client, _object_name(ref, prefix=_WRITTEN_PREFIX), _two_row_frame().table, ref)

    def _fail_if_called(data: bytes) -> Frame:
        raise AssertionError("the decode must not run past the claim")

    monkeypatch.setattr(gcs_store, "frame_from_verified_parquet_bytes", _fail_if_called)
    held = await store.open(ref)
    assert held is not None

    with pytest.raises(FrameTooLarge, match=r"has 2 rows, over this read's 1-row limit"):
        await held.read(max_rows=1)


async def test_open_refuses_a_recorded_row_count_over_the_ceiling_before_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects: dict[str, bytes] = {}
    metadata: dict[str, dict[str, str]] = {}
    ref = _ref()
    writer_store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    await writer_store.put(ref, frame=_two_row_frame(), trace=_trace(ref), writer="writer-1")
    store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata), max_frame_rows=1
    )

    def _fail_if_called(self: _FakeBlob) -> bytes:
        raise AssertionError("download_as_bytes must not run past the recorded row count")

    monkeypatch.setattr(_FakeBlob, "download_as_bytes", _fail_if_called)

    with pytest.raises(FrameTooLarge, match=r"2 rows.*1-row ceiling"):
        await store.open(ref)


async def test_read_refuses_an_object_with_no_recorded_count_over_the_row_ceiling() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client, max_frame_rows=1)
    ref = _ref()
    _store_table(client, _object_name(ref, prefix=_WRITTEN_PREFIX), _two_row_frame().table, ref)
    held = await store.open(ref)
    assert held is not None

    with pytest.raises(FrameTooLarge, match=r"2 rows.*1-row ceiling"):
        await held.read(max_rows=5)


async def test_read_refuses_an_object_whose_footer_contradicts_its_recorded_row_count() -> None:
    """The claim a caller took covers the recorded count, so a footer holding more
    rows than that is refused before anything decodes it."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    await store.put(ref, frame=_two_row_frame(), trace=_trace(ref), writer="writer-1")
    client.metadata[_object_name(ref, prefix=_WRITTEN_PREFIX)]["frame_rows"] = "1"
    held = await store.open(ref)
    assert held is not None
    assert held.num_rows == 1

    with pytest.raises(FrameIntegrityError, match="has 2 rows, not the 1"):
        await held.read(max_rows=1)


async def test_open_refuses_a_recorded_row_count_that_is_not_a_count() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")
    client.metadata[_object_name(ref, prefix=_WRITTEN_PREFIX)]["frame_rows"] = "-1"

    with pytest.raises(FrameIntegrityError, match="not a count"):
        await store.open(ref)


async def test_read_raises_frame_expired_for_an_object_deleted_after_open() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")
    held = await store.open(ref)
    assert held is not None
    del client.objects[_object_name(ref, prefix=_WRITTEN_PREFIX)]

    with pytest.raises(FrameExpired):
        await held.read(max_rows=1)


async def test_open_returns_none_for_a_pair_never_registered() -> None:
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=_FakeStorageClient())

    assert await store.open(_ref()) is None


async def test_put_is_idempotent_on_a_412() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()

    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")
    # A second put under the identical ref must not raise, and must not
    # disturb what the first write recorded.
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-2")

    result = await store.get(ref)
    assert result is not None


async def test_get_returns_none_for_a_pair_never_registered() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)

    result = await store.get(_ref())

    assert result is None


@pytest.mark.parametrize("prefix", ["frames/v3", "frames/v2", "frames/v1"])
async def test_get_raises_frame_integrity_error_on_a_digest_mismatch(prefix: str) -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    frame_object_name = _object_name(ref, prefix=prefix)
    _store_table(client, frame_object_name, _LAYOUT_TABLES[prefix](_frame()), ref)
    # Tamper with the stored bytes directly, standing in for bit rot or a
    # tampered object -- the recorded sha256 in the blob's metadata no
    # longer matches.
    client.objects[frame_object_name] = client.objects[frame_object_name] + b"\x00"

    with pytest.raises(FrameIntegrityError, match="recorded sha256"):
        await store.get(ref)


async def test_verify_content_version_on_read_off_by_default_ignores_a_bad_derivation() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    # No derive_content_version was even given -- the flag defaults off, so
    # a `get` never calls it.
    result = await store.get(ref)

    assert result is not None


async def test_verify_content_version_on_read_raises_on_a_mismatch() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(
        bucket_name=_BUCKET,
        client=client,
        verify_content_version_on_read=True,
        derive_content_version=lambda frame, upstream_trace: "cv_" + "z" * 32,
    )
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    with pytest.raises(FrameIntegrityError):
        await store.get(ref)


async def test_verify_content_version_on_read_passes_on_a_match() -> None:
    client = _FakeStorageClient()
    ref = _ref()
    store = GcsMaterialisationStore(
        bucket_name=_BUCKET,
        client=client,
        verify_content_version_on_read=True,
        derive_content_version=lambda frame, upstream_trace: ref.content_version,
    )
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    result = await store.get(ref)

    assert result is not None


def test_constructor_requires_derive_content_version_when_verification_is_on() -> None:
    with pytest.raises(ValueError, match="derive_content_version"):
        GcsMaterialisationStore(
            bucket_name=_BUCKET, client=_FakeStorageClient(), verify_content_version_on_read=True
        )


async def test_a_second_store_instance_over_the_same_bucket_reads_what_the_first_wrote() -> None:
    """Stands in for a process restart: a fresh `GcsMaterialisationStore`,
    constructed over a fresh fake client sharing the first client's own
    backing dicts, reads the pair back with no knowledge of the first
    store instance."""
    objects: dict[str, bytes] = {}
    metadata: dict[str, dict[str, str]] = {}
    ref = _ref()
    frame = _frame()
    trace = _trace(ref)

    first_store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    await first_store.put(ref, frame=frame, trace=trace, writer="writer-1")

    second_store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    result = await second_store.get(ref)

    assert result is not None
    round_tripped_frame, round_tripped_trace = result
    assert canonical_frame_bytes(round_tripped_frame) == canonical_frame_bytes(frame)
    assert round_tripped_trace == trace


async def test_aclose_closes_the_injected_client() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)

    await store.aclose()

    assert client.closed is True


async def test_put_refuses_an_over_row_ceiling_frame_and_writes_nothing() -> None:
    """A frame over `max_frame_rows` is refused before any object is
    created -- unlike the read-time ceilings below, this doesn't need a
    separate fixture producing the oversized bytes; it fires the moment
    `put` sees the frame's own row count. Insert-only writes mean a frame
    that landed anyway would be unreadable forever: no later `put` could
    replace it, and every later `get` would raise this same `FrameTooLarge`.
    """
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client, max_frame_rows=1)
    ref = _ref()
    over_ceiling_frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "time_period": str(2020 + offset),
                "value": 100.5,
                "unit": "dcid:USDollar",
            }
            for offset in range(2)
        ],
    )

    with pytest.raises(FrameTooLarge, match=r"2 rows.*1-row ceiling"):
        await store.put(ref, frame=over_ceiling_frame, trace=_trace(ref), writer="writer-1")

    assert client.objects == {}
    assert client.metadata == {}


async def test_get_refuses_an_over_row_ceiling_frame_without_decoding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row ceiling is now also enforced at `put` (see
    `test_put_refuses_an_over_row_ceiling_frame_and_writes_nothing` above),
    so the over-ceiling object this test needs already stored is written
    through a second, permissive-ceiling store sharing the same backing
    dicts -- standing in for an object that landed under a looser ceiling
    (or before this store's own ceiling was lowered) and is now read back
    through a store configured with a stricter one.
    """
    objects: dict[str, bytes] = {}
    metadata: dict[str, dict[str, str]] = {}
    ref = _ref()
    over_ceiling_frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "time_period": str(2020 + offset),
                "value": 100.5,
                "unit": "dcid:USDollar",
            }
            for offset in range(2)
        ],
    )
    writer_store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    await writer_store.put(ref, frame=over_ceiling_frame, trace=_trace(ref), writer="writer-1")

    store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata), max_frame_rows=1
    )

    def _fail_if_called(data: bytes) -> Frame:
        raise AssertionError("no decoder may run past the row ceiling check")

    monkeypatch.setattr(
        "common.materialisation.gcs_store.frame_from_verified_parquet_bytes", _fail_if_called
    )
    monkeypatch.setattr(
        "common.materialisation.gcs_store.frame_from_parquet_bytes", _fail_if_called
    )

    with pytest.raises(FrameTooLarge, match=r"2 rows.*1-row ceiling"):
        await store.get(ref)


async def test_get_refuses_an_over_column_ceiling_frame() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client, max_columns=3)
    ref = _ref()
    # `stat_var`, `time_period`, `value`, `unit`, `reporter`, `recipient` --
    # six columns, over a three-column ceiling.
    over_ceiling_frame = Frame.build(
        role_keys=["reporter", "recipient"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "recipient": "dcid:BGD",
                "time_period": "2020",
                "value": 100.5,
                "unit": "dcid:USDollar",
            }
        ],
    )
    await store.put(ref, frame=over_ceiling_frame, trace=_trace(ref), writer="writer-1")

    with pytest.raises(FrameTooLarge, match=r"columns.*3-column ceiling"):
        await store.get(ref)


async def test_get_refuses_an_oversized_object_before_downloading_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client, max_encoded_bytes=10)
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    def _fail_if_called(self: _FakeBlob) -> bytes:
        raise AssertionError("download_as_bytes must not run past the size ceiling check")

    monkeypatch.setattr(_FakeBlob, "download_as_bytes", _fail_if_called)

    with pytest.raises(FrameTooLarge, match="byte ceiling"):
        await store.get(ref)


async def test_get_refuses_a_one_row_object_over_the_uncompressed_size_ceiling() -> None:
    """A single outsized cell can compress to a small on-disk footprint
    (Parquet's own dictionary/RLE encoding collapses a repeated character
    string well) while still fully materialising to its real size once
    decoded -- the row-count ceiling alone never sees this, since the frame
    is one row."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=client, max_uncompressed_bytes=1_000_000
    )
    ref = _ref()
    huge_cell_frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "x" * (20 * 1024 * 1024),
                "time_period": "2020",
                "value": 100.5,
                "unit": "dcid:USDollar",
            }
        ],
    )
    await store.put(ref, frame=huge_cell_frame, trace=_trace(ref), writer="writer-1")

    with pytest.raises(FrameTooLarge, match="uncompressed"):
        await store.get(ref)


async def test_get_raises_frame_integrity_error_for_a_mismatched_envelope_reference() -> None:
    """Stands in for a stored object whose envelope was written for a
    different pair than the one it's filed under -- bucket-side
    tampering, or misfiling under the wrong object name."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    other_ref = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "d" * 32)
    mismatched_envelope = {
        "ref": {"spec_id": other_ref.spec_id, "content_version": other_ref.content_version},
        "trace": encode_trace(_trace(other_ref)),
    }
    frame_bytes = frame_to_parquet_bytes(
        _frame(),
        metadata={"envelope": json.dumps(mismatched_envelope).encode("utf-8")},
    )
    frame_object_name = _object_name(ref, prefix=_WRITTEN_PREFIX)
    client.objects[frame_object_name] = frame_bytes
    client.metadata[frame_object_name] = {"frame_sha256": hashlib.sha256(frame_bytes).hexdigest()}

    with pytest.raises(FrameIntegrityError, match="different reference"):
        await store.get(ref)


async def test_put_propagates_a_non_precondition_upload_failure() -> None:
    """An upload failure that isn't a 412 (already-exists) race must
    propagate out of `put`, not be swallowed alongside the idempotent case
    `contextlib.suppress(PreconditionFailed)` handles."""

    class _FailingBlob(_FakeBlob):
        def upload_from_string(
            self, data: bytes, *, content_type: str, if_generation_match: int
        ) -> None:
            raise RuntimeError("upload failed")

    class _FailingBucket(_FakeBucket):
        def blob(self, name: str) -> _FakeBlob:
            return _FailingBlob(self._objects, self._metadata_store, name)

    class _FailingClient(_FakeStorageClient):
        def bucket(self, name: str) -> _FakeBucket:
            assert name == _BUCKET
            return _FailingBucket(self.objects, self.metadata)

    client = _FailingClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()

    with pytest.raises(RuntimeError, match="upload failed"):
        await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    # Nothing landed -- a single insert-only write either lands the whole
    # object or nothing, so `get` sees a plain miss.
    result = await store.get(ref)
    assert result is None


def _object_name(ref: FrameRef, *, prefix: str) -> str:
    return f"{prefix}/{ref.spec_id}/{ref.content_version}.parquet"


def _store_table(client: _FakeStorageClient, name: str, table: pa.Table, ref: FrameRef) -> None:
    """Writes `table` under `name` with a valid envelope and digest, as `put`
    would, so every check before decoding passes."""
    envelope = {
        "ref": {"spec_id": ref.spec_id, "content_version": ref.content_version},
        "trace": encode_trace(_trace(ref)),
    }
    sink = io.BytesIO()
    pq.write_table(
        table.replace_schema_metadata({"envelope": json.dumps(envelope).encode("utf-8")}), sink
    )
    client.objects[name] = sink.getvalue()
    client.metadata[name] = {"frame_sha256": hashlib.sha256(sink.getvalue()).hexdigest()}


def _every_unit_frame(units: int = 4) -> Frame:
    """One row for each of the first `units` forms a `unit` cell takes: none, a ref,
    a converted unit and a composed unit."""
    forms: list[object] = [
        None,
        "dcid:USDollar",
        {"currency": "USD", "prices": "constant", "base_year": 2015},
        {"numerator": "dcid:USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
    ]
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "time_period": str(2020 + offset),
                "value": 100.5 - offset,
                "unit": unit,
            }
            for offset, unit in enumerate(forms[:units])
        ],
    )


def _two_member_unit_table(frame: Frame) -> pa.Table:
    """`frame`'s table in the `frames/v2/` layout: `unit` as `struct<ref, request>`."""
    position = frame.table.schema.get_field_index("unit")
    unit = frame.table.column(position).combine_chunks()
    assert isinstance(unit, pa.StructArray)
    ref, request, composed = unit.flatten()
    assert composed.null_count == len(unit)
    two_member = pa.StructArray.from_arrays(
        [ref, request], names=["ref", "request"], mask=unit.is_null()
    )
    return frame.table.set_column(position, "unit", two_member)


def _text_unit_table(frame: Frame) -> pa.Table:
    """`frame`'s rows in the `frames/v1/` layout: `unit` as JSON text, no `status`."""
    position = frame.columns.index("unit")
    texts = [
        None if row[position] is None else json.dumps(unit_cell_to_json(row[position]))
        for row in frame.rows
    ]
    table = frame.table.drop_columns("status")
    return table.set_column(
        table.schema.get_field_index("unit"), "unit", pa.array(texts, type=pa.string())
    )


# Each prefix the store reads, with a frame's table as that layout holds it.
_LAYOUT_TABLES: dict[str, Callable[[Frame], pa.Table]] = {
    "frames/v3": lambda frame: frame.table,
    "frames/v2": _two_member_unit_table,
    "frames/v1": _text_unit_table,
}


@pytest.mark.parametrize(
    ("prefix", "units"), [("frames/v3", 4), ("frames/v2", 3), ("frames/v1", 4)]
)
async def test_get_reads_the_frame_its_rows_build_from_every_layout(
    prefix: str, units: int
) -> None:
    """Each layout holds a null, a ref and a converted unit; the two-member struct
    alone holds no composed unit."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    frame = _every_unit_frame(units)
    _store_table(client, _object_name(ref, prefix=prefix), _LAYOUT_TABLES[prefix](frame), ref)

    result = await store.get(ref)

    assert result is not None
    assert result[0] == frame
    assert canonical_frame_bytes(result[0]) == canonical_frame_bytes(frame)
    assert result[1] == _trace(ref)


async def test_get_reads_the_first_layout_that_holds_the_reference() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    _store_table(client, _object_name(ref, prefix="frames/v3"), _frame().table, ref)
    two_member = _two_member_unit_table(_two_row_frame())
    _store_table(client, _object_name(ref, prefix="frames/v2"), two_member, ref)

    result = await store.get(ref)

    assert result is not None
    assert result[0] == _frame()


async def test_put_writes_the_canonical_table_under_the_v3_prefix_and_get_reads_it_back() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    frame = _every_unit_frame()

    await store.put(ref, frame=frame, trace=_trace(ref), writer="writer-1")
    name = _object_name(ref, prefix="frames/v3")
    stored_schema = pq.read_schema(io.BytesIO(client.objects[name]))
    result = await store.get(ref)

    assert list(client.objects) == [name]
    assert stored_schema == canonical_frame_schema(frame.columns)
    assert result is not None
    assert result[0] == frame
    assert result[0].table.schema.metadata is None
    assert frame_sha256(result[0]) == frame_sha256(frame)


def _float32_value_table() -> pa.Table:
    table = _frame().table
    position = table.schema.get_field_index("value")
    return table.set_column(position, "value", table.column("value").cast(pa.float32()))


def _legacy_text_unit_table() -> pa.Table:
    return pq.read_table(_LEGACY_DIR / "codec-004.parquet")


@pytest.mark.parametrize(
    ("prefix", "table"),
    [
        pytest.param("frames/v3", _legacy_text_unit_table, id="v3-text-unit"),
        pytest.param("frames/v3", _float32_value_table, id="v3-float32-value"),
        pytest.param(
            "frames/v3", lambda: _two_member_unit_table(_frame()), id="v3-two-member-unit"
        ),
        pytest.param("frames/v2", _legacy_text_unit_table, id="v2-text-unit"),
        pytest.param("frames/v2", _float32_value_table, id="v2-float32-value"),
    ],
)
async def test_get_refuses_an_object_whose_schema_is_not_its_layouts_own(
    prefix: str, table: Callable[[], pa.Table]
) -> None:
    """Each object passes every byte check, and each holds cells a v1 read
    would validate into a frame: the committed text-layout object
    (JSON-text `unit`, no `status`), the canonical columns with a
    `float32` value, and the two-member `unit` struct. The v3 and v2
    decoders wrap only their own schemas, so each refuses."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    _store_table(client, _object_name(ref, prefix=prefix), table(), ref)

    with pytest.raises(FrameIntegrityError, match="does not decode to a frame"):
        await store.get(ref)


async def test_get_reads_an_earlier_layout_object_under_the_v1_prefix() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    manifest = json.loads((_LEGACY_DIR / "manifest.json").read_text())
    legacy_table = pq.read_table(_LEGACY_DIR / "codec-004.parquet")
    _store_table(client, _object_name(ref, prefix="frames/v1"), legacy_table, ref)

    held = await store.open(ref)
    result = await store.get(ref)

    assert held is not None
    assert held.num_rows is None
    assert result is not None
    assert frame_sha256(result[0]) == manifest["codec-004.parquet"]
    assert result[1] == _trace(ref)


async def test_get_trace_reads_the_trace_without_decoding_a_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=_FakeStorageClient())
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    def _fail_if_called(data: bytes) -> Frame:
        raise AssertionError("reading a trace decodes no frame")

    monkeypatch.setattr(gcs_store, "frame_from_verified_parquet_bytes", _fail_if_called)
    monkeypatch.setattr(gcs_store, "frame_from_parquet_bytes", _fail_if_called)

    assert await store.get_trace(ref) == _trace(ref)


async def test_get_trace_returns_none_for_a_miss_and_checks_the_stored_bytes() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()

    assert await store.get_trace(ref) is None

    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")
    client.objects[_object_name(ref, prefix=_WRITTEN_PREFIX)] += b"tampered"
    with pytest.raises(FrameIntegrityError, match="sha256"):
        await store.get_trace(ref)


def _mint_record(ref: FrameRef, *, scope: dict[str, object] | None = None) -> MintRecord:
    return MintRecord(
        frame=ref,
        dataset=FrameRef(spec_id=ref.spec_id, content_version="cv_" + "d" * 32),
        options={"format": "parquet", "bundle": False, "layout": "frame"},
        scope=scope or {},
    )


async def test_a_mint_record_is_written_beside_a_frame_the_bucket_already_holds() -> None:
    """The frame's object is insert-only, so the record is an object of its own, and a
    second store over the same bucket reads it back."""
    objects: dict[str, bytes] = {}
    metadata: dict[str, dict[str, str]] = {}
    store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")
    frame_bytes = objects[_object_name(ref, prefix=_WRITTEN_PREFIX)]

    assert await store.get_mint_record(ref) is None
    await store.put_mint_record(_mint_record(ref))

    record_name = f"mints/v1/{ref.spec_id}/{ref.content_version}.json"
    assert objects == {
        _object_name(ref, prefix=_WRITTEN_PREFIX): frame_bytes,
        record_name: encode_mint_record(_mint_record(ref)),
    }
    restarted = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    assert await restarted.get_mint_record(ref) == _mint_record(ref)


async def test_the_first_mint_record_written_under_a_pair_stands() -> None:
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=_FakeStorageClient())
    ref = _ref()

    await store.put_mint_record(_mint_record(ref))
    await store.put_mint_record(_mint_record(ref, scope={"window": {"kind": "all"}}))

    assert await store.get_mint_record(ref) == _mint_record(ref)


@pytest.mark.parametrize(
    ("stored", "reason"),
    [
        (b"not json", "does not decode"),
        (
            encode_mint_record(
                _mint_record(FrameRef(spec_id="spec_" + "f" * 32, content_version="cv_" + "b" * 32))
            ),
            "names a different reference",
        ),
    ],
    ids=["undecodable", "misfiled"],
)
async def test_a_stored_mint_record_that_is_wrong_raises_frame_integrity_error(
    stored: bytes, reason: str
) -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    client.objects[f"mints/v1/{ref.spec_id}/{ref.content_version}.json"] = stored

    with pytest.raises(FrameIntegrityError, match=reason) as raised:
        await store.get_mint_record(ref)

    assert raised.value.ref == ref


async def test_list_frames_names_every_frame_object_oldest_first_and_downloads_none() -> None:
    """A bucket lists by name, so the order comes from each object's creation time. A
    pair under two layout prefixes is one frame, as old as its older object, and a
    mint record or a name that is no frame's is left out."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    start = datetime(2026, 9, 14, tzinfo=UTC)
    root = _ref()
    derived = FrameRef(spec_id="ops_" + "1" * 32, content_version="ocv_" + "2" * 32)
    both = FrameRef(spec_id="ops_" + "0" * 32, content_version="ocv_" + "3" * 32)
    written = {
        _object_name(derived, prefix="frames/v3"): 2,
        _object_name(root, prefix="frames/v2"): 0,
        _object_name(both, prefix="frames/v3"): 5,
        _object_name(both, prefix="frames/v1"): 1,
        f"mints/v1/{root.spec_id}/{root.content_version}.json": 0,
        "frames/v3/not-a-frame/ocv_x.parquet": 0,
        "frames/v3/stray.txt": 0,
    }
    for name, day in written.items():
        client.objects[name] = b"never read"
        client.created[name] = start + timedelta(days=day)

    assert await store.list_frames() == [
        FrameListing(ref=root, created_at=start),
        FrameListing(ref=both, created_at=start + timedelta(days=1)),
        FrameListing(ref=derived, created_at=start + timedelta(days=2)),
    ]


def _anchor_record(*, period: str = "2023", rule: str = "half1") -> AnchorRecord:
    return AnchorRecord(
        populations=[{"spec_id": _ref().spec_id, "bindings": {"reporter": ["dcid:USA"]}}],
        roots=[_ref()],
        rule=rule,
        period=period,
        left_out=[{"root": 0, "coordinates": {"reporter": "dcid:USA"}}],
    )


async def test_an_anchor_record_is_one_object_under_its_key_and_the_first_stands() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    record = _anchor_record()

    assert await store.get_anchor_record(record.anchor_key) is None
    await store.put_anchor_record(record)
    await store.put_anchor_record(_anchor_record(period="2022"))

    assert client.objects == {f"anchors/v1/{record.anchor_key}.json": encode_anchor_record(record)}
    restarted = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(client.objects, client.metadata)
    )
    assert await restarted.get_anchor_record(record.anchor_key) == record
    assert await restarted.list_frames() == []


@pytest.mark.parametrize(
    ("stored", "reason"),
    [
        (b"not json", "does not decode"),
        (encode_anchor_record(_anchor_record(rule="half2")), "hashes to a different key"),
    ],
    ids=["undecodable", "misfiled"],
)
async def test_a_stored_anchor_record_that_is_wrong_raises_its_integrity_error(
    stored: bytes, reason: str
) -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    key = _anchor_record().anchor_key
    client.objects[f"anchors/v1/{key}.json"] = stored

    with pytest.raises(AnchorRecordIntegrityError, match=reason) as raised:
        await store.get_anchor_record(key)

    assert raised.value.anchor_key == key
