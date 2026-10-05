"""Tests for `data_mcp.frame_materialisation`: fetch, parse, and register."""

from __future__ import annotations

import asyncio
import decimal
import hashlib
import io
import threading
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.materialisation import (
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    GcsMaterialisationStore,
    HeldFrame,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    gcs_store,
)
from common.materialisation import get_store as get_frame_store
from common.materialisation import set_store as set_frame_store
from common.materialisation.memory_store import FRAME_TTL
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S, RowBudget, row_budget, set_row_budget
from data_mcp import client_holder, frame_materialisation
from data_mcp.deadline import CallDeadline
from data_mcp.frame_materialisation import (
    _ARTIFACT_GET_ATTEMPTS,
    MaterialisationOutcome,
    frame_ref_for,
    materialise_and_register,
)
from data_mcp.manifest import Manifest, strip_manifest
from data_mcp.settings import Settings
from data_mcp.tool_errors import RetryMode, ToolErrorCode
from dfo_frame_grammar import Frame, FrameError
from dfo_frame_grammar.grammar import UnitRequest
from google.api_core.exceptions import NotFound

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client
from .test_operations import _FakeClock

_SPEC_ID = "spec_" + "1" * 32
_CONTENT_VERSION = "cv_" + "2" * 32
_FRAME_REF = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
_ARTIFACT_DIGEST = "sha256:" + "3" * 64
_SIGNED_URL = "https://storage.example/fresh-signed-url"


def _digest_for(data: bytes) -> str:
    """The manifest's own `sha256:<hex>` digest shape, computed from whichever bytes a
    test's fake transport actually serves -- so the digest gate compares against the
    truth rather than a placeholder that never matches.
    """
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"artifact_fetch_budget_s": 5.0, "max_frame_rows": 50_000}
    defaults.update(overrides)
    return Settings(**defaults)


def _manifest(
    *,
    spec_id: str = _SPEC_ID,
    content_version: str = _CONTENT_VERSION,
    artifact_digest: str = _ARTIFACT_DIGEST,
    trace: dict[str, Any] | None = None,
    as_collection: bool = False,
) -> Manifest:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "content_version": content_version,
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {"text": "OECD DAC2a", "sources": []},
        "trace": trace if trace is not None else {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": artifact_digest,
                "rendering": {"bundle": False, "format": "parquet"},
                "bytes": 100,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
            }
        ],
    }
    if as_collection:
        payload["collection"] = {"collection_id": "col_" + "0" * 32, "members": []}
        return strip_manifest(payload)
    payload["definition"] = {"schema_version": 1, "spec_id": spec_id, "identity": {}}
    return strip_manifest(payload)


def _parquet_bytes(
    *,
    role_values: list[str] | None = None,
    time_periods: list[str] | None = None,
    values: list[float] | None = None,
    stat_vars: list[str] | None = None,
) -> bytes:
    """A tiny Parquet blob shaped like a real fetched artifact:
    `stat_var`, one role column (`observationAbout`), `time_period`, and `value`, all
    already in the types `build_frame_from_arrow` reads directly -- no conversion needed.
    Every row carries `dcid:Test` unless `stat_vars` names one per row.
    """
    role_values = role_values if role_values is not None else ["country/FRA"]
    time_periods = time_periods if time_periods is not None else ["2020"]
    values = values if values is not None else [1.5]
    n = len(role_values)
    table = pa.table(
        {
            "stat_var": pa.array(stat_vars or ["dcid:Test"] * n, type=pa.string()),
            "observationAbout": pa.array(role_values, type=pa.string()),
            "time_period": pa.array(time_periods, type=pa.string()),
            "value": pa.array(values, type=pa.float64()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _snapshot_payload(
    *,
    subject_id: str = _SPEC_ID,
    content_version: str = _CONTENT_VERSION,
    artifact_digest: str = _ARTIFACT_DIGEST,
    url: str = _SIGNED_URL,
) -> dict[str, Any]:
    """The snapshot route's own response shape `read_artifact` re-validates and signs from."""
    url_expires_at = (datetime.now(UTC) + timedelta(seconds=900)).isoformat()
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": subject_id, "identity": {}},
        "content_version": content_version,
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {"text": "OECD DAC2a", "sources": []},
        "trace": {},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": artifact_digest,
                "rendering": {"bundle": False, "format": "parquet"},
                "bytes": 100,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": url,
                "url_expires_at": url_expires_at,
            }
        ],
    }


class _RecordingMaterialisationStore:
    """A hand-written fake mirroring `test_get_data_tool.py`'s own `_RecordingStore`
    pattern, but for `common.materialisation.MaterialisationStore`: pre-seedable, and
    recording every `open`, `HeldFrame.read` and `put` call so a test can assert on the
    already-held no-op path.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.open_calls: list[FrameRef] = []
        self.reads = 0
        self.put_calls: list[FrameRef] = []

    def seed(self, ref: FrameRef, *, frame: Any = None, trace: Any = None) -> None:
        self._entries[(ref.spec_id, ref.content_version)] = (frame, trace)

    async def put(self, ref: FrameRef, *, frame: Any, trace: Any, writer: str) -> None:
        del writer  # unused by this store, matching InMemoryMaterialisationStore
        self.put_calls.append(ref)
        self._entries[(ref.spec_id, ref.content_version)] = (frame, trace)

    async def open(self, ref: FrameRef) -> HeldFrame | None:
        self.open_calls.append(ref)
        entry = self._entries.get((ref.spec_id, ref.content_version))
        if entry is None:
            return None
        frame, trace = entry

        async def read(*, max_rows: int) -> Any:
            self.reads += 1
            return frame, trace

        return HeldFrame(num_rows=0 if frame is None else frame.num_rows, read=read)

    async def aclose(self) -> None:
        pass


class _FailingGetMaterialisationStore:
    """A fake whose `open` always raises `error`, standing in for a durable store that
    already holds an object under the requested pair but can't serve it back -- the
    cache-check branches this exercises never reach a real `GcsMaterialisationStore`."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.put_calls: list[FrameRef] = []

    async def open(self, _ref: FrameRef) -> HeldFrame | None:
        raise self._error

    async def put(self, ref: FrameRef, *, frame: Any, trace: Any, writer: str) -> None:
        del frame, trace, writer
        self.put_calls.append(ref)

    async def aclose(self) -> None:
        pass


class _SlowMaterialisationStore:
    """A fake whose `open` misses and whose `open` and `put` each take a set real delay,
    standing in for a GCS bucket slower than the call deadline allows."""

    def __init__(self, *, get_delay_s: float = 0.0, put_delay_s: float = 0.0) -> None:
        self._get_delay_s = get_delay_s
        self._put_delay_s = put_delay_s
        self.put_calls: list[FrameRef] = []

    async def open(self, _ref: FrameRef) -> HeldFrame | None:
        await asyncio.sleep(self._get_delay_s)
        return None

    async def put(self, ref: FrameRef, *, frame: Any, trace: Any, writer: str) -> None:
        del frame, trace, writer
        await asyncio.sleep(self._put_delay_s)
        self.put_calls.append(ref)

    async def aclose(self) -> None:
        pass


class _RowCeilingMaterialisationStore:
    """A fake `MaterialisationStore` with its own row ceiling, independent
    of `settings.max_frame_rows` -- standing in for `GcsMaterialisationStore`'s
    own `_max_frame_rows`, which `frame_materialisation.py`'s admission
    check above never sees or coordinates with (see `gcs_store.py`'s own
    module docstring for why the two constants are never coupled).
    """

    def __init__(self, *, max_frame_rows: int) -> None:
        self._max_frame_rows = max_frame_rows
        self.put_calls: list[FrameRef] = []

    async def open(self, _ref: FrameRef) -> HeldFrame | None:
        return None

    async def put(self, ref: FrameRef, *, frame: Any, trace: Any, writer: str) -> None:
        del writer
        if frame.num_rows > self._max_frame_rows:
            raise FrameTooLarge(
                ref,
                reason=(
                    f"frame has {frame.num_rows} rows, over this store's "
                    f"{self._max_frame_rows}-row ceiling"
                ),
            )
        self.put_calls.append(ref)

    async def aclose(self) -> None:
        pass


async def test_materialise_and_register_registers_a_real_frame() -> None:
    data = _parquet_bytes()
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome == MaterialisationOutcome(
        status="registered",
        attempts=1,
        refusal=None,
        role_keys=("observationAbout",),
        stat_var_count=1,
        frame_ref=_FRAME_REF,
    )
    stored = await get_frame_store().get(
        FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    )
    assert stored is not None
    frame, trace = stored
    assert frame.num_rows == 1
    assert trace.upstream_trace == manifest.trace


def _family_parquet_bytes() -> bytes:
    """Four rows over three StatVars, as a family retrieval's artifact holds them."""
    return _parquet_bytes(
        role_values=["country/FRA"] * 4,
        time_periods=["2020", "2021", "2020", "2020"],
        values=[1.0, 2.0, 3.0, 4.0],
        stat_vars=["dcid:Grants", "dcid:Grants", "dcid:Loans", "dcid:Equity"],
    )


async def test_a_fetched_frames_stat_var_count_is_read_off_its_rows() -> None:
    data = _family_parquet_bytes()
    digest = _digest_for(data)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=_manifest(artifact_digest=digest),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "registered"
    assert outcome.stat_var_count == 3


async def test_a_held_frames_stat_var_count_is_read_off_its_rows() -> None:
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = _RecordingMaterialisationStore()
    store.seed(ref, frame=frame_materialisation._frame_from_parquet_bytes(_family_parquet_bytes()))
    set_frame_store(store)

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on the already-held path")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=_manifest(),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable),
        )

    assert outcome.attempts == 0
    assert outcome.stat_var_count == 3


async def test_already_held_pair_is_a_no_op_fetch() -> None:
    manifest = _manifest()
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = _RecordingMaterialisationStore()
    store.seed(ref, frame=frame_materialisation._frame_from_parquet_bytes(_parquet_bytes()))
    set_frame_store(store)

    def unreachable_get(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no GET is expected on the already-held path")

    def unreachable_snapshot(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no read_artifact call is expected on the already-held path")

    async with _client(unreachable_snapshot) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable_get),
        )

    assert outcome == MaterialisationOutcome(
        status="registered",
        attempts=0,
        refusal=None,
        role_keys=("observationAbout",),
        stat_var_count=1,
        frame_ref=_FRAME_REF,
    )
    # No fetch and no `put` on this path. The held pair is read once, so it passes every
    # check the store's read makes.
    assert store.put_calls == []
    assert store.reads == 1


async def test_cache_check_integrity_failure_refuses_without_refetching() -> None:
    """The store already holds an object under this pair and it fails an integrity
    check on read: the store is content-addressed and insert-only, so falling through
    to re-fetch and re-`put` could never heal it -- refuses instead, and never
    attempts the fetch or the `put`."""
    manifest = _manifest()
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = _FailingGetMaterialisationStore(FrameIntegrityError(ref, reason="digest mismatch"))
    set_frame_store(store)

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on the integrity-failure path")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable),
        )

    assert outcome.status == "refused"
    assert outcome.attempts == 0
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.FRAME_INTEGRITY_FAILED
    assert outcome.refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert store.put_calls == []


async def test_cache_check_too_large_failure_refuses_without_refetching() -> None:
    """Same shape as the integrity-failure case, for the store's own pre-decode size
    ceiling: the held bytes are exactly what was written, just too large for this
    store's contract to serve back, so this is retryable -- with a narrower scope, not
    with the identical call."""
    manifest = _manifest()
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = _FailingGetMaterialisationStore(FrameTooLarge(ref, reason="stored object is huge"))
    set_frame_store(store)

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on the too-large path")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable),
        )

    assert outcome.status == "refused"
    assert outcome.attempts == 0
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.OVER_CEILING
    assert outcome.refusal.retry_mode == RetryMode.CORRECT_AND_RETRY
    assert store.put_calls == []


class _FakeGcsBlob:
    """`GcsMaterialisationStore`'s narrow blob seam over plain dicts. Duplicates
    `packages/common/tests/test_gcs_store.py`'s own fake, since a service's tests don't
    import another package's own test module."""

    def __init__(
        self, objects: dict[str, bytes], metadata: dict[str, dict[str, str]], name: str
    ) -> None:
        self._objects = objects
        self._metadata = metadata
        self._name = name
        self.metadata: dict[str, str] | None = None
        self.size: int | None = None

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert if_generation_match == 0
        self._objects[self._name] = data
        self._metadata[self._name] = dict(self.metadata or {})

    def download_as_bytes(self) -> bytes:
        return self._objects[self._name]

    def reload(self) -> None:
        if self._name not in self._objects:
            raise NotFound(f"no such object: {self._name}")
        self.metadata = dict(self._metadata[self._name])
        self.size = len(self._objects[self._name])


class _FakeGcsClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}

    def bucket(self, name: str) -> _FakeGcsClient:
        del name
        return self

    def blob(self, name: str) -> _FakeGcsBlob:
        return _FakeGcsBlob(self.objects, self.metadata, name)

    def close(self) -> None:
        pass


async def test_a_cache_hit_whose_recomputed_content_version_mismatches_refuses() -> None:
    """With the store's content-version check on, a held pair is read in full on a
    cache hit, so a frame whose recomputed content version disagrees with its reference
    refuses `frame_integrity_failed` rather than reporting the pair registered."""
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = GcsMaterialisationStore(
        bucket_name="frames",
        client=_FakeGcsClient(),
        verify_content_version_on_read=True,
        derive_content_version=lambda _frame, _upstream_trace: "cv_" + "9" * 32,
    )
    await store.put(
        ref,
        frame=frame_materialisation._frame_from_parquet_bytes(_parquet_bytes()),
        trace=MaterialisationTrace(
            upstream_trace={}, spec_id=_SPEC_ID, content_version=_CONTENT_VERSION
        ),
        writer="tester",
    )
    set_frame_store(store)

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on a held pair")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=_manifest(),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable),
        )

    assert outcome.status == "refused"
    assert outcome.attempts == 0
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.FRAME_INTEGRITY_FAILED
    assert row_budget().in_flight == 0


# A lowered `MAX_FRAME_ROWS`, under the store's own 250,000-row ceiling, with the
# smallest frame row budget the startup check allows beside it.
_LOWERED_MAX_FRAME_ROWS = 100_000


async def _gcs_store_holding(rows: int, *, record_rows: bool) -> GcsMaterialisationStore:
    """A store holding `rows` copies of `_parquet_bytes`' one row under the manifest's
    pair, with or without the row count `put` records."""
    one_row = frame_materialisation._frame_from_parquet_bytes(_parquet_bytes())
    frame = Frame.from_table(columns=one_row.columns, table=one_row.table.take([0] * rows))
    client = _FakeGcsClient()
    store = GcsMaterialisationStore(bucket_name="frames", client=client)
    await store.put(
        FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION),
        frame=frame,
        trace=MaterialisationTrace(
            upstream_trace={}, spec_id=_SPEC_ID, content_version=_CONTENT_VERSION
        ),
        writer="tester",
    )
    if not record_rows:
        (metadata,) = client.metadata.values()
        del metadata["frame_rows"]
    return store


async def _materialise_held() -> MaterialisationOutcome:
    """`materialise_and_register` for the manifest's pair under the lowered ceiling and
    budget, failing on any upstream call."""
    set_row_budget(RowBudget(max_rows=2 * _LOWERED_MAX_FRAME_ROWS))

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on a held pair")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        return await materialise_and_register(
            manifest=_manifest(),
            settings=_settings(
                max_frame_rows=_LOWERED_MAX_FRAME_ROWS,
                frame_rows_in_flight_max=2 * _LOWERED_MAX_FRAME_ROWS,
            ),
            writer="tester",
            transport=httpx2.MockTransport(unreachable),
        )


async def test_a_cache_hit_with_no_recorded_row_count_resolves_under_a_lowered_ceiling() -> None:
    set_frame_store(await _gcs_store_holding(1, record_rows=False))

    outcome = await _materialise_held()

    assert (outcome.status, outcome.attempts) == ("registered", 0)
    assert row_budget().high_water == _LOWERED_MAX_FRAME_ROWS
    assert row_budget().in_flight == 0


async def test_a_cache_hit_with_no_recorded_row_count_over_the_ceiling_refuses_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_frame_store(await _gcs_store_holding(_LOWERED_MAX_FRAME_ROWS + 1, record_rows=False))

    def _fail_if_called(data: bytes) -> Frame:
        raise AssertionError("the decode must not run past the claim")

    monkeypatch.setattr(gcs_store, "frame_from_verified_parquet_bytes", _fail_if_called)

    outcome = await _materialise_held()

    assert (outcome.status, outcome.attempts) == ("refused", 0)
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.OVER_CEILING
    assert outcome.refusal.fix["detail"] == (
        f"stored frame has {_LOWERED_MAX_FRAME_ROWS + 1} rows, over this read's "
        f"{_LOWERED_MAX_FRAME_ROWS}-row limit"
    )
    assert row_budget().in_flight == 0


async def test_a_cache_hit_with_a_recorded_row_count_over_the_ceiling_refuses_before_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_frame_store(await _gcs_store_holding(_LOWERED_MAX_FRAME_ROWS + 1, record_rows=True))

    def _fail_if_called(self: _FakeGcsBlob) -> bytes:
        raise AssertionError("nothing is downloaded past the recorded row count")

    monkeypatch.setattr(_FakeGcsBlob, "download_as_bytes", _fail_if_called)

    outcome = await _materialise_held()

    assert (outcome.status, outcome.attempts) == ("refused", 0)
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.OVER_CEILING
    assert outcome.refusal.fix["detail"] == (
        f"stored frame has {_LOWERED_MAX_FRAME_ROWS + 1} rows, over this service's "
        f"{_LOWERED_MAX_FRAME_ROWS}-row ceiling"
    )
    assert row_budget().high_water == 0


async def test_a_cache_hit_is_refused_while_the_row_budget_is_full() -> None:
    """A held pair's rows are claimed before it is read, so a full budget refuses the
    retryable `service_unavailable` the fetch path refuses, with nothing read."""
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = _RecordingMaterialisationStore()
    store.seed(ref, frame=frame_materialisation._frame_from_parquet_bytes(_parquet_bytes()))
    set_frame_store(store)
    set_row_budget(RowBudget(max_rows=1))

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on a held pair")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        with row_budget().claim(1):
            outcome = await materialise_and_register(
                manifest=_manifest(),
                settings=_settings(),
                writer="tester",
                transport=httpx2.MockTransport(unreachable),
            )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.SERVICE_UNAVAILABLE
    assert outcome.refusal.retry_mode == RetryMode.RETRY_AFTER_DELAY
    assert store.reads == 0


class _ThreadedSlowReadStore:
    """A store holding one three-row pair whose `HeldFrame.read` waits in a worker
    thread until `finish` is set."""

    def __init__(self) -> None:
        self.finish = threading.Event()
        self.reading = threading.Event()

    async def open(self, _ref: FrameRef) -> HeldFrame | None:
        async def read(*, max_rows: int) -> Any:
            await asyncio.to_thread(self._read)

        return HeldFrame(num_rows=3, read=read)

    def _read(self) -> None:
        self.reading.set()
        self.finish.wait()

    async def put(self, ref: FrameRef, *, frame: Any, trace: Any, writer: str) -> None:
        raise AssertionError("no write is expected")

    async def aclose(self) -> None:
        pass


async def test_a_store_read_that_outruns_the_deadline_keeps_its_rows_claimed_until_it_ends() -> (
    None
):
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no fetch is expected with under the fetch floor left")

    store = _ThreadedSlowReadStore()
    set_frame_store(store)
    try:
        async with _client(handler) as upstream, asyncio.timeout(5.0):
            client_holder.set_client(upstream)
            outcome = await materialise_and_register(
                manifest=_manifest(),
                settings=_settings(),
                writer="tester",
                transport=httpx2.MockTransport(handler),
                deadline=CallDeadline.after(14.05, clock=_FakeClock()),
            )

        assert outcome.degraded_reason == "store_unchecked"
        assert store.reading.is_set()
        assert row_budget().in_flight == 3
    finally:
        store.finish.set()
    async with asyncio.timeout(5.0):
        while row_budget().in_flight:
            await asyncio.sleep(0.01)


async def test_a_cache_hit_by_a_second_caller_is_still_a_no_op_fetch() -> None:
    """A second `materialise_and_register` call for the same manifest, from a different
    `writer`, hits the already-held path -- a no-op fetch, no re-registration."""
    data = _parquet_bytes()
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)
    get_calls = 0

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal get_calls
        get_calls += 1
        return httpx2.Response(200, content=data)

    store = _RecordingMaterialisationStore()
    set_frame_store(store)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        first = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="writer-one",
            transport=httpx2.MockTransport(get_handler),
        )
        second = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="writer-two",
            transport=httpx2.MockTransport(get_handler),
        )

    assert first.status == "registered"
    assert second == MaterialisationOutcome(
        status="registered",
        attempts=0,
        refusal=None,
        role_keys=("observationAbout",),
        stat_var_count=1,
        frame_ref=_FRAME_REF,
    )
    assert get_calls == 1
    assert store.put_calls == [FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)]


async def test_manifest_with_no_definition_refuses_upstream_response_unusable() -> None:
    manifest = _manifest(as_collection=True)

    outcome = await materialise_and_register(
        manifest=manifest, settings=_settings(), writer="tester"
    )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "upstream_error"


async def test_a_malformed_manifest_spec_id_refuses_rather_than_raising() -> None:
    manifest = _manifest(content_version="not-a-valid-content-version")

    outcome = await materialise_and_register(
        manifest=manifest, settings=_settings(), writer="tester"
    )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "upstream_error"


async def test_fetch_failure_after_the_retry_budget_returns_degraded_not_refused() -> None:
    manifest = _manifest()

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload())

    def failing_get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, content=b"boom")

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(failing_get_handler),
        )

    assert outcome.status == "degraded"
    assert outcome.attempts == _ARTIFACT_GET_ATTEMPTS
    assert outcome.refusal is None
    assert outcome.degraded_reason == "fetch"


async def test_fetch_budget_timeout_returns_degraded() -> None:
    manifest = _manifest()

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload())

    async def slow_get_handler(_request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.2)
        return httpx2.Response(200, content=_parquet_bytes())

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(artifact_fetch_budget_s=0.02),
            writer="tester",
            transport=httpx2.MockTransport(slow_get_handler),
        )

    assert outcome.status == "degraded"
    assert outcome.refusal is None
    # The budget crosses mid-GET, before the first attempt's response ever lands --
    # honestly zero, never assumed to be `_ARTIFACT_GET_ATTEMPTS`.
    assert outcome.attempts == 0


async def test_a_sign_failure_before_any_get_returns_degraded_with_zero_attempts() -> None:
    manifest = _manifest()

    def failing_snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"not valid json")

    def unreachable_get(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no GET is expected when signing itself already failed")

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(failing_snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable_get),
        )

    assert outcome.status == "degraded"
    assert outcome.attempts == 0
    assert outcome.refusal is None


async def test_a_signed_url_expiry_response_re_signs_once_and_succeeds() -> None:
    data = _parquet_bytes()
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)
    snapshot_calls = 0
    get_attempts = 0

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal get_attempts
        get_attempts += 1
        if get_attempts == 1:
            return httpx2.Response(403, content=b"expired")
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "registered"
    assert snapshot_calls == 2
    assert get_attempts == 2


async def test_bytes_not_matching_the_artifact_digest_refuse_with_no_store_write() -> None:
    data = _parquet_bytes()
    advertised_digest = _ARTIFACT_DIGEST  # deliberately not `_digest_for(data)`
    manifest = _manifest(artifact_digest=advertised_digest)
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=advertised_digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "upstream_error"
    assert advertised_digest in outcome.refusal.message
    assert _digest_for(data) in outcome.refusal.message
    assert await get_frame_store().get(ref) is None


async def test_row_count_over_the_ceiling_refuses_before_any_row_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _parquet_bytes(
        role_values=["country/A", "country/B", "country/C"],
        time_periods=["2020", "2021", "2022"],
        values=[1.0, 2.0, 3.0],
    )
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    def _unreachable(_data: bytes) -> Any:
        raise AssertionError("_frame_from_parquet_bytes must not be reached over the ceiling")

    monkeypatch.setattr(frame_materialisation, "_frame_from_parquet_bytes", _unreachable)
    set_frame_store(InMemoryMaterialisationStore())

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(max_frame_rows=1),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "over_ceiling"


async def test_a_store_row_ceiling_below_the_service_ceiling_refuses_rather_than_raising() -> None:
    """The row_count admission check above only guards against this
    service's own `settings.max_frame_rows`; the store's own row ceiling is
    a second, independently-owned constant that can be lower (see
    `_RowCeilingMaterialisationStore`'s own docstring), so a frame that
    already passed the check above can still be refused inside `store.put`.
    Proves that escapes this function as a clean `over_ceiling` refusal,
    never a raised `FrameTooLarge`.
    """
    data = _parquet_bytes(
        role_values=["country/A", "country/B", "country/C"],
        time_periods=["2020", "2021", "2022"],
        values=[1.0, 2.0, 3.0],
    )
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    store = _RowCeilingMaterialisationStore(max_frame_rows=1)
    set_frame_store(store)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            # Well above the store's own 1-row ceiling, so the service-level
            # admission check above passes and `store.put` is what refuses.
            settings=_settings(max_frame_rows=50_000),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "refused"
    assert outcome.attempts == 1
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "over_ceiling"
    assert outcome.refusal.fix["ceiling_kind"] == "rows"
    assert store.put_calls == []


class _BudgetRecordingStore(InMemoryMaterialisationStore):
    """An in-memory store that records the process's frame rows in flight at each write."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight_at_put: list[int] = []

    async def put(self, ref: FrameRef, *, frame: Any, trace: Any, writer: str) -> None:
        self.in_flight_at_put.append(row_budget().in_flight)
        await super().put(ref, frame=frame, trace=trace, writer=writer)


async def test_a_fetched_artifact_is_refused_while_the_row_budget_is_full() -> None:
    """Refused `service_unavailable` while another call holds the budget, and
    registered once that call releases, holding its rows through the store write."""
    data = _parquet_bytes(
        role_values=["country/A", "country/B", "country/C"],
        time_periods=["2020", "2021", "2022"],
        values=[1.0, 2.0, 3.0],
    )
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_row_budget(RowBudget(max_rows=4))
    store = _BudgetRecordingStore()
    set_frame_store(store)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        with row_budget().claim(2):
            refused = await materialise_and_register(
                manifest=manifest,
                settings=_settings(),
                writer="tester",
                transport=httpx2.MockTransport(get_handler),
            )
        registered = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert refused.status == "refused"
    assert refused.refusal is not None
    assert refused.refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE
    assert refused.refusal.retry_mode is RetryMode.RETRY_AFTER_DELAY
    assert refused.refusal.retry_after_s == ROW_BUDGET_RETRY_AFTER_S
    assert refused.refusal.fix == {
        "service": "data-mcp",
        "rows_requested": 3,
        "rows_free": 2,
        "rows_in_flight_max": 4,
    }
    assert registered.status == "registered"
    assert store.in_flight_at_put == [3]
    assert row_budget().in_flight == 0


async def test_corrupt_parquet_bytes_on_a_200_response_refuses_rather_than_raising() -> None:
    """A 2xx GET response can still carry a body that isn't a valid or complete Parquet
    file at all -- `pyarrow.ArrowException`, not a `FrameError`, and not caught by the
    existing `except FrameError` guard unless this module maps it to a refusal itself.
    """
    data = b"not a parquet file at all"
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "upstream_error"


async def test_a_parquet_file_missing_a_reserved_column_refuses_rather_than_raising() -> None:
    """A valid Parquet file that never carried `time_period` raises `KeyError` while
    decoding that column -- not a `FrameError`, so this module maps it separately.
    """
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:Test"], type=pa.string()),
            "observationAbout": pa.array(["country/FRA"], type=pa.string()),
            "value": pa.array([1.5], type=pa.float64()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    data = buf.getvalue()
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "upstream_error"


async def test_a_mismatched_recompute_still_registers_under_the_manifest_content_version() -> None:
    """No cv_ integrity gate is wired at all (see the module's own comment at the gate
    site): `manifest.trace`'s facets here carry a `basis` value (`deployment_fallback`,
    observed on a real live mint) outside `mint_content_version`'s own closed vocabulary
    (`bound`/`declared_default`) -- attempting a recompute over this trace would raise
    `ContentVersionError` rather than merely disagree, so this proves no such attempt is
    made: the frame still registers under the manifest's own `content_version`.
    """
    data = _parquet_bytes()
    digest = _digest_for(data)
    manifest = _manifest(
        artifact_digest=digest,
        trace={
            "graph_revision": "g1",
            "facets": [
                {
                    "axis": "unit",
                    "ref": "SDG_CON_USD_M",
                    "basis": "deployment_fallback",
                    "default_policy": "dfo/facet-policy/most-observations",
                    "selection_reason": "sole_candidate",
                }
            ],
        },
    )

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "registered"
    stored = await get_frame_store().get(
        FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    )
    assert stored is not None
    _frame, trace = stored
    assert trace.upstream_trace == manifest.trace


async def test_an_expired_held_entry_is_treated_as_not_held_and_heals_on_put() -> None:
    """`store.get()` on an entry past `FRAME_TTL` raises `FrameExpired`; the
    already-held check treats that the same as a genuine miss and falls through to a
    fresh fetch, whose `put` heals the pair -- the store's existing-key branch keeps the
    stored frame and only refreshes `created_at`, sound here because the seeded frame is
    built from the same bytes a fresh fetch would parse (see the module's own `except
    FrameExpired` comment at the already-held check).
    """
    data = _parquet_bytes()
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    get_calls = 0

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal get_calls
        get_calls += 1
        return httpx2.Response(200, content=data)

    seed_time = datetime(2026, 1, 1, tzinfo=UTC)
    past_ttl = seed_time + FRAME_TTL + timedelta(seconds=1)
    # `get()` reads `_now()` twice on a successful read (verification, below) -- the
    # sequence below is: seed `put()`, the pipeline's own expired `get()` (one read,
    # raises without deleting), the pipeline's healing `put()`, then the
    # verification `get()`'s own two reads.
    clock = iter([seed_time, past_ttl, past_ttl, past_ttl, past_ttl])
    store = InMemoryMaterialisationStore(now=lambda: next(clock))
    seeded_frame = frame_materialisation._frame_from_parquet_bytes(data)
    seeded_trace = MaterialisationTrace(
        upstream_trace={}, spec_id=_SPEC_ID, content_version=_CONTENT_VERSION
    )
    await store.put(ref, frame=seeded_frame, trace=seeded_trace, writer="stale-writer")
    set_frame_store(store)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "registered"
    assert outcome.attempts == 1
    assert get_calls == 1
    stored = await store.get(ref)
    assert stored is not None
    frame, _trace = stored
    assert frame.num_rows == 1


async def _fetch_budget_under(
    monkeypatch: pytest.MonkeyPatch, *, deadline: CallDeadline | None
) -> float | None:
    """The `budget_s` `materialise_and_register` hands its fetch under `deadline`, or
    `None` when it degrades without starting one."""
    budgets: list[float] = []

    async def fake_fetch(*, budget_s: float, **_: Any) -> bytes | None:
        budgets.append(budget_s)
        return None

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)
    set_frame_store(InMemoryMaterialisationStore())
    outcome = await materialise_and_register(
        manifest=_manifest(),
        settings=_settings(artifact_fetch_budget_s=60.0),
        writer="tester",
        deadline=deadline,
    )
    assert outcome.status == "degraded"
    return budgets[0] if budgets else None


async def test_the_fetch_budget_is_cut_to_the_deadline_less_the_store_write_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()

    cut = await _fetch_budget_under(monkeypatch, deadline=CallDeadline.after(35.0, clock=clock))
    capped = await _fetch_budget_under(monkeypatch, deadline=CallDeadline.after(165.0, clock=clock))
    floor = await _fetch_budget_under(monkeypatch, deadline=CallDeadline.after(15.0, clock=clock))
    spent = await _fetch_budget_under(monkeypatch, deadline=CallDeadline.after(14.0, clock=clock))
    unbounded = await _fetch_budget_under(monkeypatch, deadline=None)

    assert (cut, capped, floor, spent, unbounded) == (25.0, 60.0, 5.0, None, 60.0)


async def test_a_store_read_that_outruns_the_deadline_counts_as_a_miss() -> None:
    """15.05s left: the read must end with the 5s fetch floor plus the 10s write
    reserve left, so it gets 0.05s, runs out, and the fetch still runs."""
    data = _parquet_bytes()
    digest = _digest_for(data)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    store = _SlowMaterialisationStore(get_delay_s=60.0)
    set_frame_store(store)
    async with _client(snapshot_handler) as upstream, asyncio.timeout(5.0):
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=_manifest(artifact_digest=digest),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
            deadline=CallDeadline.after(15.05, clock=_FakeClock()),
        )

    assert outcome == MaterialisationOutcome(
        status="registered",
        attempts=1,
        role_keys=("observationAbout",),
        stat_var_count=1,
        frame_ref=_FRAME_REF,
    )
    assert store.put_calls == [FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)]


async def test_a_store_read_that_outruns_the_deadline_degrades_with_no_time_to_fetch() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no fetch is expected with under the fetch floor left")

    set_frame_store(_SlowMaterialisationStore(get_delay_s=60.0))
    async with _client(handler) as upstream, asyncio.timeout(5.0):
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=_manifest(),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(handler),
            deadline=CallDeadline.after(14.05, clock=_FakeClock()),
        )

    assert outcome == MaterialisationOutcome(
        status="degraded", attempts=0, degraded_reason="store_unchecked", frame_ref=_FRAME_REF
    )


async def test_a_store_write_that_outruns_the_deadline_degrades_unconfirmed() -> None:
    """The GET leaves 3.05s, and the write must end with 3s left, so it gets 0.05s."""
    data = _parquet_bytes()
    digest = _digest_for(data)
    clock = _FakeClock()

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        clock.now += 16.95
        return httpx2.Response(200, content=data)

    store = _SlowMaterialisationStore(put_delay_s=60.0)
    set_frame_store(store)
    async with _client(snapshot_handler) as upstream, asyncio.timeout(5.0):
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            manifest=_manifest(artifact_digest=digest),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
            deadline=CallDeadline.after(20.0, clock=clock),
        )

    assert outcome == MaterialisationOutcome(
        status="degraded",
        attempts=1,
        degraded_reason="store_write",
        role_keys=("observationAbout",),
        stat_var_count=1,
        frame_ref=_FRAME_REF,
    )
    assert store.put_calls == []


class _ThreadedSlowPutStore(InMemoryMaterialisationStore):
    """An in-memory store whose write waits in a worker thread until `finish` is set, as
    `GcsMaterialisationStore.put` encodes and uploads in one."""

    def __init__(self) -> None:
        super().__init__()
        self.finish = threading.Event()

    async def put(self, ref: FrameRef, *, frame: Any, trace: Any, writer: str) -> None:
        await asyncio.to_thread(self.finish.wait)
        await super().put(ref, frame=frame, trace=trace, writer=writer)


async def test_a_store_write_that_outruns_the_deadline_keeps_its_rows_claimed_until_it_ends() -> (
    None
):
    data = _parquet_bytes(
        role_values=["country/A", "country/B", "country/C"],
        time_periods=["2020", "2021", "2022"],
        values=[1.0, 2.0, 3.0],
    )
    digest = _digest_for(data)
    clock = _FakeClock()

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        clock.now += 16.95
        return httpx2.Response(200, content=data)

    store = _ThreadedSlowPutStore()
    set_frame_store(store)
    try:
        async with _client(snapshot_handler) as upstream, asyncio.timeout(5.0):
            client_holder.set_client(upstream)
            outcome = await materialise_and_register(
                manifest=_manifest(artifact_digest=digest),
                settings=_settings(),
                writer="tester",
                transport=httpx2.MockTransport(get_handler),
                deadline=CallDeadline.after(20.0, clock=clock),
            )

        assert outcome.degraded_reason == "store_write"
        assert row_budget().in_flight == 3
    finally:
        store.finish.set()
    async with asyncio.timeout(5.0):
        while row_budget().in_flight:
            await asyncio.sleep(0.01)
    assert await store.get(FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION))


_UNIT_STRUCT = pa.struct(
    [("currency", pa.string()), ("prices", pa.string()), ("base_year", pa.int64())]
)


def _unit_struct_parquet(unit: dict[str, object]) -> bytes:
    """A one-row artifact whose `unit` column is the struct dfo-apis writes for a converted member."""
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:Test"], type=pa.string()),
            "observationAbout": pa.array(["country/FRA"], type=pa.string()),
            "time_period": pa.array(["2020"], type=pa.string()),
            "value": pa.array([1.5], type=pa.float64()),
            "unit": pa.array([unit], type=_UNIT_STRUCT),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _decoded_unit(unit: dict[str, object]) -> object:
    frame = frame_materialisation._frame_from_parquet_bytes(_unit_struct_parquet(unit))
    return frame.rows[0][frame.columns.index("unit")]


def test_a_current_price_unit_struct_drops_its_null_base_year() -> None:
    unit = _decoded_unit({"currency": "USD", "prices": "current", "base_year": None})

    assert unit == UnitRequest(currency="USD", prices="current", base_year=None)


def test_a_constant_price_unit_struct_keeps_its_base_year() -> None:
    unit = _decoded_unit({"currency": "USD", "prices": "constant", "base_year": 2021})

    assert unit == UnitRequest(currency="USD", prices="constant", base_year=2021)


def test_a_current_price_unit_struct_with_a_base_year_still_refuses_invalid_unit() -> None:
    with pytest.raises(FrameError) as excinfo:
        _decoded_unit({"currency": "USD", "prices": "current", "base_year": 2021})

    assert excinfo.value.category == "invalid_unit"


def test_a_unit_struct_with_no_currency_still_refuses_invalid_unit() -> None:
    with pytest.raises(FrameError) as excinfo:
        _decoded_unit({"currency": None, "prices": "current", "base_year": None})

    assert excinfo.value.category == "invalid_unit"


def _decoded(*, row_group_size: int | None = None, **columns: pa.Array | None) -> Any:
    """`_frame_from_parquet_bytes` over a four-row artifact, each column in `columns`
    replacing (or, as `None`, dropping) the string-typed default of that name."""
    table_columns: dict[str, pa.Array] = {
        "stat_var": pa.array(["dcid:Test"] * 4),
        "observationAbout": pa.array([f"country/C{index}" for index in range(4)]),
        "time_period": pa.array(["2020"] * 4),
        "value": pa.array([1.5, -2.0, 0.0, 3.25]),
        "unit": pa.array(["dcid:USDollar"] * 4),
        "status": pa.nulls(4, type=pa.string()),
    }
    for name, column in columns.items():
        if column is None:
            del table_columns[name]
        else:
            table_columns[name] = column
    buf = io.BytesIO()
    pq.write_table(pa.table(table_columns), buf, row_group_size=row_group_size)
    return frame_materialisation._frame_from_parquet_bytes(buf.getvalue())


def _cells(frame: Any, column: str) -> dict[str, object]:
    """`column`'s cell for each `observationAbout` entity."""
    return dict(
        zip(
            frame.table.column("observationAbout").to_pylist(),
            frame.table.column(column).to_pylist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("row_group_size", [None, 1])
def test_decimal_values_convert_as_python_float_does(row_group_size: int | None) -> None:
    """Arrow's own decimal-to-double cast rounds these three wrongly; the decode
    matches `float(Decimal)` for each distinct value, across row groups too."""
    decimals = [
        decimal.Decimal("-68258643466660376553"),
        decimal.Decimal("51438419027618680893"),
        decimal.Decimal("41035029590811307848"),
        decimal.Decimal("51438419027618680893"),
    ]
    frame = _decoded(
        row_group_size=row_group_size, value=pa.array(decimals, type=pa.decimal128(38, 0))
    )

    assert _cells(frame, "value") == {
        f"country/C{index}": float(cell) for index, cell in enumerate(decimals)
    }


def test_decimal256_values_convert_to_float() -> None:
    decimals = [decimal.Decimal(f"{index}.125") for index in range(4)]
    frame = _decoded(value=pa.array(decimals, type=pa.decimal256(60, 3)))

    assert list(_cells(frame, "value").values()) == [0.125, 1.125, 2.125, 3.125]


@pytest.mark.parametrize(
    "value",
    [
        pa.array([decimal.Decimal("1"), None, decimal.Decimal("2"), decimal.Decimal("3")]),
        pa.array([None, 1.0, 2.0, 3.0], type=pa.float64()),
    ],
    ids=["decimal", "float64"],
)
def test_a_null_value_refuses_as_a_schema_violation(value: pa.Array) -> None:
    with pytest.raises(FrameError) as raised:
        _decoded(value=value)

    assert raised.value.category == "schema_violation"


def test_a_zoned_timestamp_period_takes_the_date_in_its_own_zone() -> None:
    """02:00 UTC on 1 January is still 31 December in New York."""
    instant = datetime(2020, 1, 1, 2, tzinfo=UTC)
    frame = _decoded(
        time_period=pa.array([instant] * 4, type=pa.timestamp("us", tz="America/New_York"))
    )

    assert set(_cells(frame, "time_period").values()) == {"2019-12-31"}


def test_a_nanosecond_timestamp_period_keeps_its_date() -> None:
    frame = _decoded(time_period=pa.array([1_600_000_000_000_000_001] * 4, type=pa.timestamp("ns")))

    assert set(_cells(frame, "time_period").values()) == {"2020-09-13"}


@pytest.mark.parametrize(
    ("time_period", "category"),
    [
        (
            pa.array([date(2020, 1, 1), None, date(2020, 1, 2), date(2020, 1, 3)]),
            "schema_violation",
        ),
        (pa.array([253402300800] * 4, type=pa.timestamp("s")), "invalid_time_period"),
        (pa.array([371085174374400] * 4, type=pa.timestamp("s")), "invalid_time_period"),
        (pa.array([-719163] * 4, type=pa.date32()), "invalid_time_period"),
    ],
    ids=["null-date", "year-10000", "2-to-the-32-days", "year-0"],
)
def test_a_malformed_temporal_period_refuses(time_period: pa.Array, category: str) -> None:
    with pytest.raises(FrameError) as raised:
        _decoded(time_period=time_period)

    assert raised.value.category == category


def test_a_zoned_timestamp_period_after_2037_keeps_daylight_saving() -> None:
    """04:30 UTC on 1 July 2040 is 00:30 EDT; Arrow's own zone rules drop daylight
    saving after 2037 and read it as 23:30 EST the day before."""
    instant = datetime(2040, 7, 1, 4, 30, tzinfo=UTC)
    frame = _decoded(
        time_period=pa.array([instant] * 4, type=pa.timestamp("s", tz="America/New_York"))
    )

    assert set(_cells(frame, "time_period").values()) == {"2040-07-01"}


_FIRST_DAY_S = -62_135_596_800  # 0001-01-01T00:00:00Z
_AFTER_LAST_DAY_S = 253_402_300_800  # 10000-01-01T00:00:00Z


@pytest.mark.parametrize(
    ("data_type", "tick", "expected"),
    [
        (pa.timestamp("s"), 371_085_174_374_400, None),
        (pa.timestamp("s"), 185_542_587_187_200, None),
        (pa.timestamp("s"), _AFTER_LAST_DAY_S - 1, "9999-12-31"),
        (pa.timestamp("s"), _AFTER_LAST_DAY_S, None),
        (pa.timestamp("s"), _FIRST_DAY_S, "0001-01-01"),
        (pa.timestamp("s"), _FIRST_DAY_S - 1, None),
        (pa.timestamp("s"), 2**63 - 1, None),
        (pa.timestamp("ms"), 371_085_174_374_400_000, None),
        (pa.timestamp("ms"), _AFTER_LAST_DAY_S * 10**3 - 1, "9999-12-31"),
        (pa.timestamp("ms"), _FIRST_DAY_S * 10**3 - 1, None),
        (pa.timestamp("us"), _AFTER_LAST_DAY_S * 10**6 - 1, "9999-12-31"),
        (pa.timestamp("us"), _AFTER_LAST_DAY_S * 10**6, None),
        (pa.timestamp("us"), _FIRST_DAY_S * 10**6, "0001-01-01"),
        (pa.timestamp("us"), -(2**63), None),
        (pa.timestamp("ns"), 2**63 - 1, "2262-04-11"),
        (pa.timestamp("ns"), -(2**63), "1677-09-21"),
        (pa.timestamp("ns"), -1, "1969-12-31"),
        (pa.timestamp("s", tz="-05:00"), _FIRST_DAY_S + 5 * 3600, "0001-01-01"),
        (pa.timestamp("s", tz="-05:00"), _FIRST_DAY_S + 5 * 3600 - 1, None),
        (pa.timestamp("s", tz="-05:00"), _AFTER_LAST_DAY_S + 3600, None),
        (pa.timestamp("s", tz="+09:00"), _AFTER_LAST_DAY_S - 9 * 3600 - 1, "9999-12-31"),
        (pa.timestamp("s", tz="+09:00"), _AFTER_LAST_DAY_S - 9 * 3600, None),
        (pa.timestamp("ms", tz="America/New_York"), 371_085_174_374_400_000, None),
        (pa.timestamp("us", tz="Asia/Tokyo"), 2**63 - 1, None),
        (pa.timestamp("ns", tz="UTC"), 2**63 - 1, "2262-04-11"),
        (pa.date32(), -719_162, "0001-01-01"),
        (pa.date32(), -719_163, None),
        (pa.date32(), 2_932_896, "9999-12-31"),
        (pa.date32(), 2_932_897, None),
        (pa.date32(), 2**31 - 1, None),
        (pa.date32(), -(2**31), None),
        (pa.date64(), -1, "1969-12-31"),
        (pa.date64(), -719_163 * 86_400_000, None),
        (pa.date64(), 2_932_897 * 86_400_000, None),
        (pa.date64(), 371_085_174_374_400_000, None),
        (pa.date64(), 2**63 - 1, None),
    ],
    ids=[
        "s-2-to-the-32-days",
        "s-2-to-the-31-days",
        "s-last-second",
        "s-year-10000",
        "s-first-second",
        "s-year-0",
        "s-max",
        "ms-2-to-the-32-days",
        "ms-last-millisecond",
        "ms-year-0",
        "us-last-microsecond",
        "us-year-10000",
        "us-first-microsecond",
        "us-min",
        "ns-max",
        "ns-min",
        "ns-before-epoch",
        "minus-5h-first-local-day",
        "minus-5h-year-0",
        "minus-5h-utc-year-10000",
        "plus-9h-last-local-second",
        "plus-9h-year-10000",
        "new-york-ms-2-to-the-32-days",
        "tokyo-us-max",
        "utc-ns-max",
        "date32-first-day",
        "date32-year-0",
        "date32-last-day",
        "date32-year-10000",
        "date32-max",
        "date32-min",
        "date64-before-epoch",
        "date64-year-0",
        "date64-year-10000",
        "date64-2-to-the-32-days",
        "date64-max",
    ],
)
def test_a_temporal_period_decodes_to_its_exact_date_or_refuses(
    data_type: pa.DataType, tick: int, expected: str | None
) -> None:
    """Each stored integer reads as the date Python's `datetime` gives it, in the
    column's zone, and one Python cannot represent refuses. Parquet stores `s` as
    `ms` and `date64` as `date32`, so these call the time_period decode directly."""
    column = pa.chunked_array([pa.array([tick, tick], type=data_type)])

    if expected is None:
        with pytest.raises(FrameError) as raised:
            frame_materialisation._time_period_text(column)
        assert raised.value.category == "invalid_time_period"
    else:
        decoded = frame_materialisation._time_period_text(column)
        assert decoded.to_pylist() == [expected, expected]


def test_a_timestamp_period_in_an_unknown_zone_refuses() -> None:
    column = pa.chunked_array([pa.array([0], type=pa.timestamp("s", tz="Mars/Olympus"))])

    with pytest.raises(FrameError) as raised:
        frame_materialisation._time_period_text(column)

    assert raised.value.category == "invalid_time_period"


def test_a_unit_struct_without_a_base_year_field_decodes_current_prices() -> None:
    frame = _decoded(unit=pa.array([{"currency": "USD", "prices": "current"}] * 4))

    request = {"currency": "USD", "prices": "current", "base_year": None}
    assert list(_cells(frame, "unit").values()) == [{"ref": None, "request": request}] * 4


def test_a_unit_struct_in_the_frame_tagged_form_refuses_as_an_invalid_unit() -> None:
    """An artifact's struct `unit` holds requests; the frame's own `ref`/`request`
    tagged form is two unknown request members."""
    unit = pa.array([{"ref": "dcid:USDollar", "request": None}] * 4)

    with pytest.raises(FrameError) as raised:
        _decoded(unit=unit)

    assert raised.value.category == "invalid_unit"


def test_a_text_unit_column_decodes_refs_and_converted_units_as_dfo_apis_writes_them() -> None:
    """dfo-apis writes a ref as its own text and a converted unit as JCS object text."""
    unit = pa.array(
        [
            "dcid:USDollar",
            '{"currency":"EUR","prices":"current"}',
            '{"base_year":2024,"currency":"USD","prices":"constant"}',
            None,
        ]
    )

    frame = _decoded(unit=unit)

    assert _cells(frame, "unit") == {
        "country/C0": {"ref": "dcid:USDollar", "request": None},
        "country/C1": {
            "ref": None,
            "request": {"currency": "EUR", "prices": "current", "base_year": None},
        },
        "country/C2": {
            "ref": None,
            "request": {"currency": "USD", "prices": "constant", "base_year": 2024},
        },
        "country/C3": None,
    }


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ('{"currency":"EUR"', "schema_violation"),
        ("{1}", "schema_violation"),
        ('{"currency":"EUR","prices":"current","base_year":2024}', "invalid_unit"),
        ('{"currency":"EUR","prices":"current","scale":"millions"}', "invalid_unit"),
    ],
    ids=["truncated", "invalid-json", "current-with-base-year", "unknown-member"],
)
def test_a_text_unit_that_opens_an_object_but_is_no_converted_unit_refuses(
    text: str, category: str
) -> None:
    with pytest.raises(FrameError) as raised:
        _decoded(unit=pa.array([text] * 4))

    assert raised.value.category == category


def test_a_parquet_file_missing_value_raises_key_error() -> None:
    with pytest.raises(KeyError):
        _decoded(value=None)


def test_an_empty_parquet_file_missing_time_period_raises_key_error() -> None:
    table = pa.table(
        {"stat_var": pa.array([], type=pa.string()), "value": pa.array([], pa.float64())}
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)

    with pytest.raises(KeyError):
        frame_materialisation._frame_from_parquet_bytes(buf.getvalue())


@pytest.mark.parametrize("column", ["unit", "measurementMethod", "provenance"])
@pytest.mark.parametrize(
    "arrow_type",
    [pa.float64(), pa.int64(), pa.list_(pa.string()), pa.null()],
    ids=["float64", "int64", "list", "null"],
)
def test_an_all_null_optional_column_of_any_type_decodes_as_none(
    column: str, arrow_type: pa.DataType
) -> None:
    frame = _decoded(**{column: pa.nulls(4, type=arrow_type)})

    cells = _cells(frame, column).values()
    if column == "unit":
        assert all(cell is None or cell == {"ref": None, "request": None} for cell in cells)
    else:
        assert set(cells) == {None}


@pytest.mark.parametrize("column", ["unit", "measurementMethod", "provenance"])
def test_a_non_null_non_text_optional_column_still_refuses(column: str) -> None:
    with pytest.raises(FrameError) as raised:
        _decoded(**{column: pa.array([1.5, None, None, None])})

    assert raised.value.category == "schema_violation"


_FACET_STRUCT = pa.struct([("label", pa.string())])


@pytest.mark.parametrize("column", ["measurementMethod", "provenance"])
def test_an_all_null_struct_facet_column_decodes_as_none(column: str) -> None:
    frame = _decoded(**{column: pa.nulls(4, type=_FACET_STRUCT)})

    assert set(_cells(frame, column).values()) == {None}


def test_an_all_null_struct_unit_column_keeps_its_type_and_decodes() -> None:
    frame = _decoded(unit=pa.nulls(4, type=_UNIT_STRUCT))

    assert all(
        cell is None or cell == {"ref": None, "request": None}
        for cell in _cells(frame, "unit").values()
    )


@pytest.mark.parametrize("column", ["measurementMethod", "provenance"])
def test_a_struct_facet_column_with_a_value_still_refuses(column: str) -> None:
    values = pa.array([{"label": "x"}, None, None, None], type=_FACET_STRUCT)

    with pytest.raises(FrameError) as raised:
        _decoded(**{column: values})

    assert raised.value.category == "schema_violation"


@pytest.mark.parametrize("column", ["stat_var", "time_period"])
def test_an_empty_parquet_file_missing_a_required_column_refuses(column: str) -> None:
    columns = {
        "stat_var": pa.array([], type=pa.string()),
        "observationAbout": pa.array([], type=pa.string()),
        "time_period": pa.array([], type=pa.string()),
        "value": pa.array([], pa.float64()),
    }
    del columns[column]
    buf = io.BytesIO()
    pq.write_table(pa.table(columns), buf)

    with pytest.raises((KeyError, FrameError)):
        frame_materialisation._frame_from_parquet_bytes(buf.getvalue())


@pytest.mark.parametrize("column", ["observationAbout", "time_period", "unit"])
def test_an_empty_parquet_file_with_an_int_typed_column_refuses(column: str) -> None:
    columns = {
        "stat_var": pa.array([], type=pa.string()),
        "observationAbout": pa.array([], type=pa.string()),
        "time_period": pa.array([], type=pa.string()),
        "value": pa.array([], pa.float64()),
        "unit": pa.array([], type=pa.string()),
    }
    columns[column] = pa.array([], type=pa.int64())
    buf = io.BytesIO()
    pq.write_table(pa.table(columns), buf)

    with pytest.raises(FrameError) as raised:
        frame_materialisation._frame_from_parquet_bytes(buf.getvalue())

    assert raised.value.category == "schema_violation"


# A frame asked for place types: GDP with no place named, as dfo-apis renders it with
# `observationAbout_type` (dfo-apis `materialise.enrichment`): every `typeOf` sorted and
# comma-joined, and an empty cell for a place its graph holds no type for.
_EVERY_PLACE_ROWS = [
    ("Earth", "Place"),
    ("country/HIC", "ProvisionalNode"),
    ("country/USA", "Country"),
    ("nuts/FI2", "AdministrativeArea1,Country,EurostatNUTS1,Region"),
    ("undata-geo/G00158000", ""),
]


def _every_place_parquet(
    types: pa.Array | None = None, *, decoded: dict[str, pa.Array] | None = None
) -> bytes:
    count = len(_EVERY_PLACE_ROWS)
    table = pa.table(
        {
            **(decoded or {}),
            "stat_var": pa.array(
                ["dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"] * count
            ),
            "observationAbout": pa.array([place for place, _types in _EVERY_PLACE_ROWS]),
            "time_period": pa.array(["2023"] * count),
            "value": pa.array([float(index + 1) for index in range(count)]),
            "unit": pa.array(["USDollar"] * count),
            "measurementMethod": pa.array([None] * count, type=pa.string()),
            "provenance": pa.array(["dc/base/WorldDevelopmentIndicators"] * count),
            "observationAbout_type": (
                pa.array([cell for _place, cell in _EVERY_PLACE_ROWS]) if types is None else types
            ),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def test_a_place_type_column_is_a_role_and_an_empty_cell_reads_untyped() -> None:
    frame = frame_materialisation._frame_from_parquet_bytes(_every_place_parquet())

    role_keys, stat_var_count = frame_materialisation.observed_shape(frame)
    types = dict(
        zip(
            frame.table.column("observationAbout").to_pylist(),
            frame.table.column("observationAbout_type").to_pylist(),
            strict=True,
        )
    )

    assert (role_keys, stat_var_count) == (("observationAbout", "observationAbout_type"), 1)
    assert types == {
        "Earth": "Place",
        "country/HIC": "ProvisionalNode",
        "country/USA": "Country",
        "nuts/FI2": "AdministrativeArea1,Country,EurostatNUTS1,Region",
        "undata-geo/G00158000": "untyped",
    }


def test_a_place_type_column_of_nulls_reads_untyped() -> None:
    nulls = pa.array([None] * len(_EVERY_PLACE_ROWS), type=pa.string())
    frame = frame_materialisation._frame_from_parquet_bytes(_every_place_parquet(nulls))

    assert set(frame.table.column("observationAbout_type").to_pylist()) == {"untyped"}


def test_a_decoded_code_system_column_survives_beside_the_place_type_column() -> None:
    # A mint that names only `observationAbout_type` in `options.columns` keeps every default
    # column, so the Parquet carries `stat_var_<code system>` beside the type.
    decoded = {"stat_var_sector": pa.array(["Basic health care"] * len(_EVERY_PLACE_ROWS))}
    frame = frame_materialisation._frame_from_parquet_bytes(_every_place_parquet(decoded=decoded))

    role_keys, stat_var_count = frame_materialisation.observed_shape(frame)

    assert set(role_keys) == {"observationAbout", "observationAbout_type", "stat_var_sector"}
    assert stat_var_count == 1
    assert set(frame.table.column("stat_var_sector").to_pylist()) == {"Basic health care"}
    assert "untyped" in frame.table.column("observationAbout_type").to_pylist()


def test_the_default_columns_keep_the_mints_own_pair() -> None:
    assert frame_ref_for(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION) == _FRAME_REF


def test_extra_columns_name_a_frame_of_their_own() -> None:
    """The `content_version` is recomputed from the form `frame_ref_for` documents,
    since a change to the derivation would orphan every stored typed frame."""
    typed = frame_ref_for(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION, columns=("observationAbout_type",)
    )
    two = frame_ref_for(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION, columns=("b_type", "a_type")
    )

    assert typed.spec_id == _SPEC_ID
    assert (
        typed.content_version
        == "cv_"
        + hashlib.sha256(
            b'{"columns":["observationAbout_type"],"content_version":"'
            + _CONTENT_VERSION.encode()
            + b'"}'
        ).hexdigest()[:32]
    )
    assert two == frame_ref_for(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION, columns=("a_type", "b_type")
    )
    assert len({_FRAME_REF.content_version, typed.content_version, two.content_version}) == 3
