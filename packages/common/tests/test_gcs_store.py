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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    GcsMaterialisationStore,
    MaterialisationTrace,
    encode_trace,
    frame_to_parquet_bytes,
    gcs_store,
)
from dfo_frame_grammar import Frame, canonical_frame_bytes, canonical_frame_schema, frame_sha256
from google.api_core.exceptions import NotFound, PreconditionFailed

_BUCKET = "dev-fin-observatory-frames"
_LEGACY_DIR = Path(__file__).parent / "fixtures" / "legacy_parquet"


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
        self.closed = False

    def bucket(self, name: str) -> _FakeBucket:
        assert name == _BUCKET
        return _FakeBucket(self.objects, self.metadata)

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
    assert downloads == [_object_name(ref, prefix="frames/v2")]


async def test_put_records_the_frames_row_count_beside_its_digest_and_writer() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()

    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")

    assert client.metadata[_object_name(ref, prefix="frames/v2")]["frame_rows"] == "1"


async def test_open_reports_no_row_count_for_an_object_stored_without_one() -> None:
    """An object written before `put` recorded a row count reads back under any claim
    its footer fits."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client, max_frame_rows=7)
    ref = _ref()
    _store_table(client, _object_name(ref, prefix="frames/v2"), _frame().table, ref)

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
    _store_table(client, _object_name(ref, prefix="frames/v2"), _two_row_frame().table, ref)

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
    _store_table(client, _object_name(ref, prefix="frames/v2"), _two_row_frame().table, ref)
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
    client.metadata[_object_name(ref, prefix="frames/v2")]["frame_rows"] = "1"
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
    client.metadata[_object_name(ref, prefix="frames/v2")]["frame_rows"] = "-1"

    with pytest.raises(FrameIntegrityError, match="not a count"):
        await store.open(ref)


async def test_read_raises_frame_expired_for_an_object_deleted_after_open() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")
    held = await store.open(ref)
    assert held is not None
    del client.objects[_object_name(ref, prefix="frames/v2")]

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


async def test_get_raises_frame_integrity_error_on_a_digest_mismatch() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    await store.put(ref, frame=_frame(), trace=_trace(ref), writer="writer-1")
    # Tamper with the stored bytes directly, standing in for bit rot or a
    # tampered object -- the recorded sha256 in the blob's metadata no
    # longer matches.
    frame_object_name = f"frames/v2/{ref.spec_id}/{ref.content_version}.parquet"
    client.objects[frame_object_name] = client.objects[frame_object_name] + b"\x00"

    with pytest.raises(FrameIntegrityError):
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
    frame_object_name = f"frames/v2/{ref.spec_id}/{ref.content_version}.parquet"
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


async def test_put_writes_the_canonical_table_under_the_v2_prefix_and_get_reads_it_back() -> None:
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 100.5,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "reporter": "dcid:GBR",
                "time_period": "2020",
                "value": -0.1,
                "unit": {"currency": "USD", "prices": "constant", "base_year": 2015},
                "provenance": "dcid:P1",
            },
        ],
    )

    await store.put(ref, frame=frame, trace=_trace(ref), writer="writer-1")
    name = _object_name(ref, prefix="frames/v2")
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


@pytest.mark.parametrize(
    "table",
    [
        pytest.param(lambda: pq.read_table(_LEGACY_DIR / "codec-004.parquet"), id="v1-layout"),
        pytest.param(_float32_value_table, id="float32-value"),
    ],
)
async def test_get_refuses_a_v2_object_whose_schema_is_not_canonical(
    table: Callable[[], pa.Table],
) -> None:
    """Each object passes every byte check, and each holds cells a v1 read
    would validate into a frame: the committed earlier-layout object
    (JSON-text `unit`, no `status`), and the canonical columns with a
    `float32` value. The v2 decoder wraps only the canonical schema, so
    both refuse."""
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    ref = _ref()
    _store_table(client, _object_name(ref, prefix="frames/v2"), table(), ref)

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
