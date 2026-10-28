"""Tests for `data_mcp.frame_materialisation`: fetch, parse, and register."""

from __future__ import annotations

import asyncio
import decimal
import hashlib
import io
import threading
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.apis_auth import TOKEN_HEADER
from common.materialisation import (
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    GcsMaterialisationStore,
    HeldFrame,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
    gcs_store,
)
from common.materialisation import get_store as get_frame_store
from common.materialisation import set_store as set_frame_store
from common.materialisation.memory_store import FRAME_TTL
from common.materialisation.mint_record import encode_mint_record
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S, RowBudget, row_budget, set_row_budget
from data_mcp import client_holder, frame_materialisation
from data_mcp.deadline import CallDeadline
from data_mcp.frame_materialisation import (
    _ARTIFACT_GET_ATTEMPTS,
    CANONICAL_FRAME_RENDERING,
    KIND_COLUMNS,
    MaterialisationOutcome,
    frame_ref_for,
    materialise_and_register,
)
from data_mcp.manifest import Manifest, strip_manifest
from data_mcp.settings import Settings
from data_mcp.tool_errors import RetryMode, ToolErrorCode
from dfo_frame_grammar import (
    ComposedUnit,
    Frame,
    FrameError,
    StatVarPart,
    UnitRequest,
    frame_sha256,
    mint_content_version,
    unit_cell_array,
)
from google.api_core.exceptions import NotFound

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import ID_TOKEN, _client, id_token_auth
from .test_client import _settings as _upstream_settings
from .test_operations import _FakeClock

_SPEC_ID = "spec_" + "1" * 32
_ARTIFACT_DIGEST = "sha256:" + "3" * 64
# The options `get_data` posts for the default rendering.
_OPTIONS: dict[str, object] = {"bundle": False, "format": "parquet", "layout": "frame"}


def _parquet_bytes(
    *,
    role_values: list[str] | None = None,
    time_periods: list[str] | None = None,
    values: list[float] | None = None,
    stat_vars: list[str] | None = None,
) -> bytes:
    """A tiny frame in the canonical layout apis writes a version's canonical
    frame in (`frame_to_parquet_bytes`): `stat_var`, one role column
    (`observationAbout`), `time_period`, `value`, and the null `unit`, facet and
    `status` columns. Every row carries `dcid:Test` unless `stat_vars` names one per
    row. A rendering's decoder (`frame_from_parquet_bytes`) reads it too.
    """
    role_values = role_values if role_values is not None else ["country/FRA"]
    time_periods = time_periods if time_periods is not None else ["2020"]
    values = values if values is not None else [1.5]
    stat_vars = stat_vars or ["dcid:Test"] * len(role_values)
    frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {"stat_var": stat_var, "observationAbout": role, "time_period": period, "value": value}
            for stat_var, role, period, value in zip(
                stat_vars, role_values, time_periods, values, strict=True
            )
        ],
    )
    return frame_to_parquet_bytes(frame)


def _content_version_of(data: bytes) -> str:
    """The `content_version` apis mints for the canonical frame `data` holds."""
    return mint_content_version(frame_sha256=frame_sha256(frame_from_verified_parquet_bytes(data)))


_CONTENT_VERSION = _content_version_of(_parquet_bytes())
_FRAME_REF = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
_LINK = f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}/files/{_ARTIFACT_DIGEST}"


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
    renderings: dict[str, dict[str, object]] | None = None,
    stamps: Mapping[str, str] | None = None,
) -> Manifest:
    """A manifest listing one default-rendering artifact under `artifact_digest`, or one
    artifact per `renderings` entry (digest to rendering), in its order, each carrying
    the `enrichment_stamp` `stamps` gives its digest, if any."""
    if renderings is None:
        renderings = {artifact_digest: {"bundle": False, "format": "parquet", "layout": "frame"}}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "content_version": content_version,
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {"text": "OECD DAC2a", "sources": []},
        "trace": trace if trace is not None else {"graph_revision": "g1", "facets": []},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": digest,
                "rendering": rendering,
                "bytes": 100,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                **({"enrichment_stamp": stamps[digest]} if digest in (stamps or {}) else {}),
            }
            for digest, rendering in renderings.items()
        ],
    }
    payload["definition"] = {"schema_version": 1, "spec_id": spec_id, "identity": {}}
    return strip_manifest(payload)


def _snapshot_payload(
    *,
    subject_id: str = _SPEC_ID,
    content_version: str = _CONTENT_VERSION,
    artifact_digest: str = _ARTIFACT_DIGEST,
    url: str = _LINK,
    rendering: dict[str, object] | None = None,
) -> dict[str, Any]:
    """The snapshot route's own response shape `read_artifact` re-validates and reads
    the link from, listing one artifact under `rendering`, by default the canonical
    frame a default mint reads."""
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
                "rendering": rendering or dict(CANONICAL_FRAME_RENDERING),
                "bytes": 100,
                "media_type": "application/vnd.apache.parquet",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": url,
            }
        ],
    }


class _RecordingMaterialisationStore:
    """A hand-written fake mirroring `test_get_data_tool.py`'s own `_RecordingStore`
    pattern, but for `common.materialisation.MaterialisationStore`: pre-seedable, and
    recording every `open`, `HeldFrame.read`, `put` and `put_mint_record` call so a
    test can assert on the already-held no-op path.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.open_calls: list[FrameRef] = []
        self.reads = 0
        self.put_calls: list[FrameRef] = []
        self.mint_records: list[MintRecord] = []

    async def put_mint_record(self, record: MintRecord) -> None:
        self.mint_records.append(record)

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

    async def put_mint_record(self, record: MintRecord) -> None:
        del record

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
            options=_OPTIONS,
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
        family_summed=False,
        frame_ref=_FRAME_REF,
    )
    stored = await get_frame_store().get(
        FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    )
    assert stored is not None
    frame, trace = stored
    assert frame.num_rows == 1
    assert trace.upstream_trace == manifest.trace
    assert await get_frame_store().get_mint_record(_FRAME_REF) == MintRecord(
        frame=_FRAME_REF, dataset=_FRAME_REF, options=_OPTIONS, stat_var_count=1
    )


def _family_parquet_bytes() -> bytes:
    """Four rows over three StatVars, as a family retrieval's artifact holds them."""
    return _parquet_bytes(
        role_values=["country/FRA"] * 4,
        time_periods=["2020", "2021", "2020", "2020"],
        values=[1.0, 2.0, 3.0, 4.0],
        stat_vars=["dcid:Grants", "dcid:Grants", "dcid:Loans", "dcid:Equity"],
    )


def _summed_parquet_bytes() -> bytes:
    """Two rows of one series under the family ref, as a summed retrieval's artifact
    holds them."""
    return _parquet_bytes(
        role_values=["country/FRA"] * 2,
        time_periods=["2020", "2021"],
        values=[8.0, 2.0],
        stat_vars=[f"dcid:{_SPEC_ID}"] * 2,
    )


_FRAME_STAT_VARS = pytest.mark.parametrize(
    ("data", "stat_var_count", "family_summed"),
    [(_family_parquet_bytes(), 3, False), (_summed_parquet_bytes(), 1, True)],
    ids=["family", "summed"],
)


@_FRAME_STAT_VARS
async def test_a_fetched_frames_stat_vars_are_read_off_its_rows(
    data: bytes, stat_var_count: int, family_summed: bool
) -> None:
    digest = _digest_for(data)

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
            manifest=_manifest(artifact_digest=digest, content_version=_content_version_of(data)),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "registered"
    assert (outcome.stat_var_count, outcome.family_summed) == (stat_var_count, family_summed)


@_FRAME_STAT_VARS
async def test_a_held_frames_stat_vars_are_read_off_its_rows(
    data: bytes, stat_var_count: int, family_summed: bool
) -> None:
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = _RecordingMaterialisationStore()
    store.seed(ref, frame=frame_materialisation.frame_from_parquet_bytes(data))
    set_frame_store(store)

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on the already-held path")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
            manifest=_manifest(),
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable),
        )

    assert outcome.attempts == 0
    assert (outcome.stat_var_count, outcome.family_summed) == (stat_var_count, family_summed)


async def test_already_held_pair_is_a_no_op_fetch() -> None:
    manifest = _manifest()
    ref = FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)
    store = _RecordingMaterialisationStore()
    store.seed(ref, frame=frame_materialisation.frame_from_parquet_bytes(_parquet_bytes()))
    set_frame_store(store)
    scope = {"entities": {"recipient": ["country/UGA"]}}

    def unreachable_get(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no GET is expected on the already-held path")

    def unreachable_snapshot(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no read_artifact call is expected on the already-held path")

    async with _client(unreachable_snapshot) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
            scope=scope,
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
        family_summed=False,
        frame_ref=_FRAME_REF,
    )
    # No fetch and no `put` on this path. The held pair is read once, so it passes every
    # check the store's read makes.
    assert store.put_calls == []
    assert store.reads == 1
    # The frame was held before this call and still gains its mint record.
    assert store.mint_records == [
        MintRecord(
            frame=_FRAME_REF, dataset=_FRAME_REF, options=_OPTIONS, scope=scope, stat_var_count=1
        )
    ]


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
            options=_OPTIONS,
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
            options=_OPTIONS,
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
        derive_content_version=lambda _frame: "cv_" + "9" * 32,
    )
    await store.put(
        ref,
        frame=frame_materialisation.frame_from_parquet_bytes(_parquet_bytes()),
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
            options=_OPTIONS,
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
    one_row = frame_materialisation.frame_from_parquet_bytes(_parquet_bytes())
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
            options=_OPTIONS,
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
    store.seed(ref, frame=frame_materialisation.frame_from_parquet_bytes(_parquet_bytes()))
    set_frame_store(store)
    set_row_budget(RowBudget(max_rows=1))

    def unreachable(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call is expected on a held pair")

    async with _client(unreachable) as upstream:
        client_holder.set_client(upstream)
        with row_budget().claim(1):
            outcome = await materialise_and_register(
                options=_OPTIONS,
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

    async def put_mint_record(self, record: MintRecord) -> None:
        del record

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
                options=_OPTIONS,
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
            options=_OPTIONS,
            manifest=manifest,
            settings=_settings(),
            writer="writer-one",
            transport=httpx2.MockTransport(get_handler),
        )
        second = await materialise_and_register(
            options=_OPTIONS,
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
        family_summed=False,
        frame_ref=_FRAME_REF,
    )
    assert get_calls == 1
    assert store.put_calls == [FrameRef(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION)]


async def test_a_malformed_manifest_spec_id_refuses_rather_than_raising() -> None:
    manifest = _manifest(content_version="not-a-valid-content-version")

    outcome = await materialise_and_register(
        options=_OPTIONS, manifest=manifest, settings=_settings(), writer="tester"
    )

    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code.value == "upstream_error"


# A done operation lists every rendering apis stores under the pair. Dev listed
# another caller's groupby sum first, then the plain rendering and its statistics.
_SUM_GROUPBY: dict[str, object] = {"by": ["time_period", "crsPurpose"], "op": "sum"}
_SUM_SIBLING_RENDERING: dict[str, object] = {
    "bundle": False,
    "format": "parquet",
    "layout": "frame",
    "groupby": _SUM_GROUPBY,
}
_TYPED_OPTIONS: dict[str, object] = _OPTIONS | {"columns": ["observationAbout_type"]}
_STATISTICS_RENDERING: dict[str, object] = {"statistics": "series"}


def _bytes_by_digest(count: int) -> dict[str, bytes]:
    """`count` distinct one-row Parquet blobs, keyed by their own digests."""
    blobs = [_parquet_bytes(values=[float(i)]) for i in range(count)]
    return {_digest_for(blob): blob for blob in blobs}


@pytest.mark.parametrize(
    ("listed", "options", "requested_at", "ref_content_version"),
    [
        (
            [_OPTIONS, _SUM_SIBLING_RENDERING, _STATISTICS_RENDERING],
            _SUM_SIBLING_RENDERING,
            1,
            frame_ref_for(
                spec_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                groupby=_SUM_GROUPBY,
            ).content_version,
        ),
        (
            [_OPTIONS, _TYPED_OPTIONS],
            _TYPED_OPTIONS,
            1,
            frame_ref_for(
                spec_id=_SPEC_ID,
                content_version=_CONTENT_VERSION,
                columns=("observationAbout_type",),
            ).content_version,
        ),
    ],
    ids=["sum-listed-after-the-plain-rendering", "place-types-listed-after-the-plain-rendering"],
)
async def test_the_fetched_artifact_is_the_one_whose_rendering_the_mint_requested(
    monkeypatch: pytest.MonkeyPatch,
    listed: list[dict[str, object]],
    options: dict[str, object],
    requested_at: int,
    ref_content_version: str,
) -> None:
    blobs = _bytes_by_digest(len(listed))
    digests = list(blobs)
    fetched: list[str] = []

    async def fake_fetch(*, artifact_digest: str, **_: Any) -> bytes:
        fetched.append(artifact_digest)
        return blobs[artifact_digest]

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)
    set_frame_store(InMemoryMaterialisationStore())
    outcome = await materialise_and_register(
        options=options,
        manifest=_manifest(renderings=dict(zip(digests, listed, strict=True))),
        settings=_settings(),
        writer="tester",
    )

    assert outcome.status == "registered"
    assert fetched == [digests[requested_at]]
    ref = FrameRef(spec_id=_SPEC_ID, content_version=ref_content_version)
    assert outcome.frame_ref == ref
    # The mint record sits under the frame's pair and names apis' own.
    assert await get_frame_store().get_mint_record(ref) == MintRecord(
        frame=ref, dataset=_FRAME_REF, options=options, stat_var_count=1
    )


@pytest.mark.parametrize("deadline_s", [None, 4.0], ids=["no-deadline", "under-the-fetch-floor"])
async def test_a_manifest_without_the_requested_rendering_refuses_without_fetching(
    monkeypatch: pytest.MonkeyPatch, deadline_s: float | None
) -> None:
    """The refusal holds whatever deadline is left, under the 5 s fetch floor too."""

    async def unreachable(**_: Any) -> bytes:
        raise AssertionError("no fetch is expected without the requested rendering")

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", unreachable)
    set_frame_store(InMemoryMaterialisationStore())
    blobs = _bytes_by_digest(2)
    outcome = await materialise_and_register(
        options=_TYPED_OPTIONS,
        manifest=_manifest(
            renderings=dict(
                zip(blobs, [_SUM_SIBLING_RENDERING, _STATISTICS_RENDERING], strict=True)
            )
        ),
        settings=_settings(),
        writer="tester",
        deadline=None if deadline_s is None else CallDeadline.after(deadline_s, clock=_FakeClock()),
    )

    assert outcome.status == "refused"
    assert outcome.attempts == 0
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert outcome.refusal.retry_mode == RetryMode.DO_NOT_RETRY


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
            options=_OPTIONS,
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(failing_get_handler),
        )

    assert outcome.status == "degraded"
    assert outcome.attempts == _ARTIFACT_GET_ATTEMPTS
    assert outcome.refusal is None
    assert outcome.degraded_reason == "fetch"
    # No frame was read, so no record is written: one without the frame's StatVar
    # count would stand for good under the store's first-writer rule.
    assert await get_frame_store().get_mint_record(_FRAME_REF) is None


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
            options=_OPTIONS,
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
            options=_OPTIONS,
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(unreachable_get),
        )

    assert outcome.status == "degraded"
    assert outcome.attempts == 0
    assert outcome.refusal is None


async def test_an_id_token_that_cannot_mint_fails_each_get_as_a_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot read takes the first token. Every mint for the link GET fails
    and is retried as a failed GET is, and the fetch degrades with nothing sent."""
    monkeypatch.setattr(frame_materialisation, "_ARTIFACT_GET_BACKOFF_S", 0.0)
    manifest = _manifest()
    seen: list[httpx2.Request] = []

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload())

    def get_handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(500)

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler, auth=id_token_auth(mints=1)) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "degraded"
    assert outcome.degraded_reason == "fetch"
    assert outcome.attempts == _ARTIFACT_GET_ATTEMPTS
    assert seen == []


async def test_the_link_is_followed_across_origins_carrying_no_credentials() -> None:
    data = _parquet_bytes()
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)
    seen: list[httpx2.Request] = []

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.host == "storage.example":
            return httpx2.Response(200, content=data)
        return httpx2.Response(
            302, headers={"Location": "https://storage.example/signed?X-Goog-Signature=abc"}
        )

    set_frame_store(InMemoryMaterialisationStore())
    # The site origin is the model-facing one. This fetch must not use it.
    upstream_settings = _upstream_settings(dfo_site_origin="https://site.example")
    async with _client(
        snapshot_handler, settings=upstream_settings, auth=id_token_auth()
    ) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "registered"
    assert [(r.url.host, r.url.path) for r in seen] == [
        ("localhost", _LINK),
        ("storage.example", "/signed"),
    ]
    assert seen[0].headers[TOKEN_HEADER] == ID_TOKEN
    assert TOKEN_HEADER not in seen[1].headers
    credential_headers = {"authorization", "cookie", "x-dfo-principal"}
    for request in seen:
        assert credential_headers.isdisjoint(request.headers.keys())


async def test_a_link_answering_410_degrades_without_retrying() -> None:
    data = _parquet_bytes()
    digest = _digest_for(data)
    manifest = _manifest(artifact_digest=digest)
    get_attempts = 0

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal get_attempts
        get_attempts += 1
        return httpx2.Response(410, json={"code": "artifact_expired"})

    set_frame_store(InMemoryMaterialisationStore())
    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
            manifest=manifest,
            settings=_settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )

    assert outcome.status == "degraded"
    assert get_attempts == 1


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
            options=_OPTIONS,
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
        raise AssertionError("frame_from_parquet_bytes must not be reached over the ceiling")

    monkeypatch.setattr(frame_materialisation, "frame_from_parquet_bytes", _unreachable)
    set_frame_store(InMemoryMaterialisationStore())

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
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
    manifest = _manifest(artifact_digest=digest, content_version=_content_version_of(data))

    def snapshot_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_payload(artifact_digest=digest))

    def get_handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=data)

    store = _RowCeilingMaterialisationStore(max_frame_rows=1)
    set_frame_store(store)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
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
    manifest = _manifest(artifact_digest=digest, content_version=_content_version_of(data))

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
                options=_OPTIONS,
                manifest=manifest,
                settings=_settings(),
                writer="tester",
                transport=httpx2.MockTransport(get_handler),
            )
        registered = await materialise_and_register(
            options=_OPTIONS,
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
            options=_OPTIONS,
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
            options=_OPTIONS,
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
    site): the frame registers under the manifest's own `content_version` whatever its
    `manifest.trace` carries.
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
            options=_OPTIONS,
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
    seeded_frame = frame_materialisation.frame_from_parquet_bytes(data)
    seeded_trace = MaterialisationTrace(
        upstream_trace={}, spec_id=_SPEC_ID, content_version=_CONTENT_VERSION
    )
    await store.put(ref, frame=seeded_frame, trace=seeded_trace, writer="stale-writer")
    set_frame_store(store)

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
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
        options=_TYPED_OPTIONS,
        manifest=_manifest(renderings={_ARTIFACT_DIGEST: _TYPED_OPTIONS}),
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
            options=_OPTIONS,
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
        family_summed=False,
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
            options=_OPTIONS,
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
            options=_OPTIONS,
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
        family_summed=False,
        frame_ref=_FRAME_REF,
    )
    assert store.put_calls == []


class _SlowMintRecordStore(InMemoryMaterialisationStore):
    """An in-memory store whose frame read takes 16.95s of `clock` and whose mint
    record write never ends."""

    def __init__(self, clock: _FakeClock) -> None:
        super().__init__()
        self._clock = clock

    async def open(self, ref: FrameRef) -> HeldFrame | None:
        held = await super().open(ref)
        self._clock.now += 16.95
        return held

    async def put_mint_record(self, record: MintRecord) -> None:
        del record
        await asyncio.sleep(60.0)


async def test_a_mint_record_write_that_outruns_the_deadline_leaves_the_frame_registered() -> None:
    """The read leaves 3.05s, and the write must end with 3s left, so it gets 0.05s."""
    clock = _FakeClock()
    store = _SlowMintRecordStore(clock)
    await store.put(
        _FRAME_REF,
        frame=frame_materialisation.frame_from_parquet_bytes(_parquet_bytes()),
        trace=MaterialisationTrace(
            upstream_trace={}, spec_id=_SPEC_ID, content_version=_CONTENT_VERSION
        ),
        writer="tester",
    )
    set_frame_store(store)

    async with asyncio.timeout(5.0):
        outcome = await materialise_and_register(
            options=_OPTIONS,
            manifest=_manifest(),
            settings=_settings(),
            writer="tester",
            deadline=CallDeadline.after(20.0, clock=clock),
        )

    assert outcome.status == "registered"
    assert outcome.frame_ref == _FRAME_REF


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
                options=_OPTIONS,
                manifest=_manifest(
                    artifact_digest=digest, content_version=_content_version_of(data)
                ),
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
    assert await store.get(FrameRef(spec_id=_SPEC_ID, content_version=_content_version_of(data)))


_UNIT_STRUCT = pa.struct(
    [("currency", pa.string()), ("prices", pa.string()), ("base_year", pa.int64())]
)


def _unit_struct_parquet(unit: dict[str, object]) -> bytes:
    """A one-row artifact whose `unit` column is the struct apis writes for a converted member."""
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
    frame = frame_materialisation.frame_from_parquet_bytes(_unit_struct_parquet(unit))
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


def _decoded(
    *,
    row_group_size: int | None = None,
    composed_units: bool = False,
    **columns: pa.Array | None,
) -> Any:
    """`frame_from_parquet_bytes` over a four-row artifact, each column in `columns`
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
    return frame_materialisation.frame_from_parquet_bytes(
        buf.getvalue(), composed_units=composed_units
    )


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


def _unit_cells(frame: Any) -> dict[str, object]:
    """The `unit` cell for each `observationAbout` entity, as `Frame.rows` gives it."""
    about, unit = frame.columns.index("observationAbout"), frame.columns.index("unit")
    return {row[about]: row[unit] for row in frame.rows}


def test_a_unit_struct_without_a_base_year_field_decodes_current_prices() -> None:
    frame = _decoded(unit=pa.array([{"currency": "USD", "prices": "current"}] * 4))

    assert set(_unit_cells(frame).values()) == {UnitRequest(currency="USD", prices="current")}


@pytest.mark.parametrize(
    "unit",
    [
        "dcid:USDollar",
        ComposedUnit(
            numerator="dcid:USDollar", denominator=StatVarPart(stat_var="dcid:Count_Person")
        ),
    ],
    ids=["ref", "composed"],
)
def test_a_unit_struct_in_the_frame_tagged_form_refuses_as_an_invalid_unit(
    unit: str | ComposedUnit,
) -> None:
    """An artifact's struct `unit` holds requests; the members of the frame's own
    tagged form are unknown request members."""
    with pytest.raises(FrameError) as raised:
        _decoded(unit=unit_cell_array(unit, 4))

    assert raised.value.category == "invalid_unit"


def test_a_text_unit_column_decodes_refs_and_converted_units_as_dfo_apis_writes_them() -> None:
    """apis writes a ref as its own text and a converted unit as JCS object text."""
    unit = pa.array(
        [
            "dcid:USDollar",
            '{"currency":"EUR","prices":"current"}',
            '{"base_year":2024,"currency":"USD","prices":"constant"}',
            None,
        ]
    )

    frame = _decoded(unit=unit)

    assert _unit_cells(frame) == {
        "country/C0": "dcid:USDollar",
        "country/C1": UnitRequest(currency="EUR", prices="current"),
        "country/C2": UnitRequest(currency="USD", prices="constant", base_year=2024),
        "country/C3": None,
    }


def test_a_derived_frame_decodes_composed_units_as_the_runner_writes_them() -> None:
    """The analysis runner writes a quotient's composed unit as the JCS text of its
    `{numerator, denominator}` object, beside refs and converted units."""
    unit = pa.array(
        [
            '{"denominator":"dcid:Person","numerator":"dcid:USDollar"}',
            '{"denominator":{"stat_var":"dcid:Count_Person"},'
            '"numerator":{"base_year":2024,"currency":"USD","prices":"constant"}}',
            "dcid:USDollar",
            None,
        ]
    )

    frame = _decoded(unit=unit, composed_units=True)

    assert _unit_cells(frame) == {
        "country/C0": ComposedUnit(numerator="dcid:USDollar", denominator="dcid:Person"),
        "country/C1": ComposedUnit(
            numerator=UnitRequest(currency="USD", prices="constant", base_year=2024),
            denominator=StatVarPart(stat_var="dcid:Count_Person"),
        ),
        "country/C2": "dcid:USDollar",
        "country/C3": None,
    }


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ('{"denominator":"dcid:Person"', "schema_violation"),
        ('{"denominator":"dcid:Person","numerator":"dcid:USDollar","scale":1}', "invalid_unit"),
        ('{"denominator":"","numerator":"dcid:USDollar"}', "invalid_unit"),
        ('{"numerator":"dcid:USDollar"}', "invalid_unit"),
    ],
    ids=["truncated", "unknown-member", "empty-side", "missing-side"],
)
def test_a_derived_frame_refuses_a_malformed_composed_unit(text: str, category: str) -> None:
    with pytest.raises(FrameError) as raised:
        _decoded(unit=pa.array([text] * 4), composed_units=True)

    assert raised.value.category == category


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ('{"currency":"EUR"', "schema_violation"),
        ("{1}", "schema_violation"),
        ('{"currency":"EUR","prices":"current","base_year":2024}', "invalid_unit"),
        ('{"currency":"EUR","prices":"current","scale":"millions"}', "invalid_unit"),
        (
            '{"denominator":{"stat_var":"dcid:Count_Person"},"numerator":"dcid:USDollar"}',
            "invalid_unit",
        ),
    ],
    ids=["truncated", "invalid-json", "current-with-base-year", "unknown-member", "composed"],
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
        frame_materialisation.frame_from_parquet_bytes(buf.getvalue())


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

    cells = _unit_cells(frame) if column == "unit" else _cells(frame, column)
    assert set(cells.values()) == {None}


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

    assert set(_unit_cells(frame).values()) == {None}


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
        frame_materialisation.frame_from_parquet_bytes(buf.getvalue())


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
        frame_materialisation.frame_from_parquet_bytes(buf.getvalue())

    assert raised.value.category == "schema_violation"


# A frame asked for place types: GDP with no place named, as apis renders it with
# `observationAbout_type` (apis `materialise.enrichment`): every `typeOf` sorted and
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
    frame = frame_materialisation.frame_from_parquet_bytes(_every_place_parquet())

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


@pytest.mark.parametrize("described", ["financingRecipient_party_kind", "financingRecipient_type"])
def test_an_empty_flow_party_kind_or_type_cell_reads_untyped(described: str) -> None:
    # apis leaves the kind cell empty for a party it declares no kind for. Filled, the
    # frame registers, and a sum across the parties refuses `place_levels_mixed`.
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:Oda"] * 3),
            "financingProvider": pa.array(["country/FRA"] * 3),
            "financingRecipient": pa.array(["country/KEN", "DACAfrica", "dcid:Unknown"]),
            "time_period": pa.array(["2022"] * 3),
            "value": pa.array([1.0, 2.0, 3.0]),
            described: pa.array(["party", "total", None]),
            "financingProvider_party_kind": pa.array([""] * 3),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)

    frame = frame_materialisation.frame_from_parquet_bytes(buf.getvalue())

    cells = dict(
        zip(
            frame.table.column("financingRecipient").to_pylist(),
            frame.table.column(described).to_pylist(),
            strict=True,
        )
    )
    assert cells == {"country/KEN": "party", "DACAfrica": "total", "dcid:Unknown": "untyped"}
    assert set(frame.table.column("financingProvider_party_kind").to_pylist()) == {"untyped"}


def test_a_partys_class_and_whole_cells_pass_through_and_an_empty_class_reads_untyped() -> None:
    """apis leaves the class cell empty for a total holding more than one class. The
    decoder writes that cell as `untyped` and every served cell reaches the frame as it
    was served."""
    table = pa.table(
        {
            "stat_var": pa.array(["dcid:Oda"] * 5),
            "financingProvider": pa.array(
                [
                    "country/FRA",
                    "GatesFoundation",
                    "DACOfficialDonors",
                    "DACUnspecified",
                    "DACAllDonors",
                ]
            ),
            "financingRecipient": pa.array(["country/ETH"] * 5),
            "time_period": pa.array(["2021"] * 5),
            "value": pa.array([1.0] * 5),
            "financingProvider_party_kind": pa.array(
                ["party", "party", "total", "residual", "total"]
            ),
            "financingProvider_party_class": pa.array(
                ["official", "private", "official", "residual", ""]
            ),
            "financingProvider_party_whole": pa.array(["false", "false", "true", "false", "false"]),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)

    frame = frame_materialisation.frame_from_parquet_bytes(buf.getvalue())

    rows = frame.table.select(
        ["financingProvider", "financingProvider_party_class", "financingProvider_party_whole"]
    ).to_pylist()
    assert {row["financingProvider"]: tuple(row.values())[1:] for row in rows} == {
        "country/FRA": ("official", "false"),
        "GatesFoundation": ("private", "false"),
        "DACOfficialDonors": ("official", "true"),
        "DACUnspecified": ("residual", "false"),
        "DACAllDonors": ("untyped", "false"),
    }


@pytest.mark.parametrize(
    "identity",
    [{"entities": {"origin": ["country/FRA"]}}, {"entities": {"recipient": ["country/KEN"]}}, None],
    ids=["flow", "place", "unheld"],
)
@pytest.mark.parametrize(
    "recorded",
    (*frame_materialisation.TYPE_COLUMN_SELECTIONS, ("kind",), ("party_kind",), KIND_COLUMNS),
    ids="+".join,
)
def test_a_replayed_selection_asks_a_flow_root_for_kinds(
    recorded: tuple[str, ...], identity: dict[str, Any] | None
) -> None:
    options = {"format": "parquet", "bundle": False, "layout": "frame"}
    if recorded:
        options["columns"] = list(recorded)

    replayed = frame_materialisation.replayed_mint_options(options, identity=identity)

    flow_root = identity is not None and "origin" in identity["entities"]
    expected = [column for column in recorded if column not in {"kind", *KIND_COLUMNS}]
    if flow_root:
        expected.extend(
            column
            for column in frame_materialisation.FLOW_ENRICHMENT_COLUMNS
            if column not in expected
        )
        assert replayed == options | {"columns": expected}
    elif recorded:
        if any(
            column in {"kind", *frame_materialisation.FLOW_ENRICHMENT_COLUMNS}
            for column in recorded
        ):
            expected.extend(KIND_COLUMNS)
        assert replayed == options | {"columns": expected}
    else:
        assert replayed == options


def test_a_place_type_column_of_nulls_reads_untyped() -> None:
    nulls = pa.array([None] * len(_EVERY_PLACE_ROWS), type=pa.string())
    frame = frame_materialisation.frame_from_parquet_bytes(_every_place_parquet(nulls))

    assert set(frame.table.column("observationAbout_type").to_pylist()) == {"untyped"}


def test_a_decoded_code_system_column_survives_beside_the_place_type_column() -> None:
    # A mint that names only `observationAbout_type` in `options.columns` keeps every default
    # column, so the Parquet carries `stat_var_<code system>` beside the type.
    decoded = {"stat_var_sector": pa.array(["Basic health care"] * len(_EVERY_PLACE_ROWS))}
    frame = frame_materialisation.frame_from_parquet_bytes(_every_place_parquet(decoded=decoded))

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


@pytest.mark.parametrize(
    ("columns", "recorded"),
    [
        (("observationAbout_type",), "cv_ec8eccaeb4265cbfca527550b24d3d3b"),
        (("kind",), "cv_5279bdb960ddb0938351c08b4ada5390"),
        (
            ("financingRecipient_type", "financingProvider_type"),
            "cv_3b6c33d24aa1d42e0de5d54574aa1ca5",
        ),
        (("party_kind",), "cv_aa6883b65fa153b6b5fd56e03a5dca14"),
        (KIND_COLUMNS, "cv_f91d484f1789bbcde7595066c052e2fe"),
    ],
)
def test_a_columns_only_pair_keeps_the_recorded_derivation(
    columns: tuple[str, ...], recorded: str
) -> None:
    """Recorded from the derivation before `frame_ref_for` took a groupby: stored
    typed frames and `mint_recovery`'s matching of recorded selections depend on it."""
    ref = frame_ref_for(spec_id=_SPEC_ID, content_version="cv_" + "2" * 32, columns=columns)

    assert ref == FrameRef(spec_id=_SPEC_ID, content_version=recorded)


def test_a_groupby_names_a_frame_apart_from_the_plain_and_the_typed_pair() -> None:
    summed = frame_ref_for(spec_id=_SPEC_ID, content_version=_CONTENT_VERSION, groupby=_SUM_GROUPBY)
    summed_typed = frame_ref_for(
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        columns=("observationAbout_type",),
        groupby=_SUM_GROUPBY,
    )
    typed = frame_ref_for(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION, columns=("observationAbout_type",)
    )

    assert (
        summed.content_version
        == "cv_"
        + hashlib.sha256(
            b'{"columns":[],"content_version":"'
            + _CONTENT_VERSION.encode()
            + b'","groupby":{"by":["time_period","crsPurpose"],"op":"sum"}}'
        ).hexdigest()[:32]
    )
    assert len({ref.content_version for ref in (_FRAME_REF, summed, summed_typed, typed)}) == 4


_KIND_OPTIONS: dict[str, object] = _OPTIONS | {"columns": ["party_kind"]}


@pytest.mark.parametrize(
    ("columns", "keyed_by_stamp"),
    [
        (("party_kind",), True),
        (KIND_COLUMNS, True),
        (("party_class",), True),
        (("party_whole",), True),
        (("observationAbout_type",), False),
        ((), False),
    ],
    ids=["kind", "kind-class-whole", "class", "whole", "types", "default"],
)
def test_the_enrichment_stamp_names_only_a_frame_that_asks_for_kinds(
    columns: tuple[str, ...], keyed_by_stamp: bool
) -> None:
    """The kind cells come from apis's code index, which moves with no change to the
    version, so a stamp of it names the frame. Types come from the graph at render time,
    and the default frame is the version's canonical one."""
    refs = {
        stamp: frame_ref_for(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            columns=columns,
            enrichment_stamp=stamp,
        )
        for stamp in (None, "stamp-a", "stamp-b")
    }

    assert (refs["stamp-a"] != refs["stamp-b"]) is keyed_by_stamp
    assert (refs["stamp-a"] != refs[None]) is keyed_by_stamp


def test_a_stamped_groupby_sum_of_kinds_is_named_apart_from_the_unstamped_one() -> None:
    refs = {
        frame_ref_for(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            columns=("party_kind",),
            groupby=_SUM_GROUPBY,
            enrichment_stamp=stamp,
        ).content_version
        for stamp in (None, "stamp-a", "stamp-b")
    }

    assert len(refs) == 3


async def test_two_manifests_differing_in_the_enrichment_stamp_register_two_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = _parquet_bytes()

    async def fake_fetch(**_: Any) -> bytes:
        return blob

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)
    store = InMemoryMaterialisationStore()
    set_frame_store(store)

    outcomes = {
        stamp: await materialise_and_register(
            options=_KIND_OPTIONS,
            manifest=_manifest(
                renderings={_digest_for(blob): _KIND_OPTIONS},
                stamps={_digest_for(blob): stamp},
            ),
            settings=_settings(),
            writer="tester",
        )
        for stamp in ("stamp-a", "stamp-b")
    }

    refs = {stamp: outcome.frame_ref for stamp, outcome in outcomes.items()}
    assert all(outcome.status == "registered" for outcome in outcomes.values())
    assert refs["stamp-a"] != refs["stamp-b"]
    for stamp, ref in refs.items():
        assert ref == frame_ref_for(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            columns=("party_kind",),
            enrichment_stamp=stamp,
        )
        record = await store.get_mint_record(ref)
        assert record is not None
        assert record.enrichment_stamp == stamp
        _, trace = await store.get(ref)
        assert trace.enrichment_stamp == stamp


async def test_a_manifest_without_a_stamp_keeps_todays_ref_and_a_record_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = _parquet_bytes()

    async def fake_fetch(**_: Any) -> bytes:
        return blob

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)
    store = InMemoryMaterialisationStore()
    set_frame_store(store)

    outcome = await materialise_and_register(
        options=_KIND_OPTIONS,
        manifest=_manifest(renderings={_digest_for(blob): _KIND_OPTIONS}),
        settings=_settings(),
        writer="tester",
    )

    assert outcome.frame_ref == frame_ref_for(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION, columns=("party_kind",)
    )
    assert outcome.frame_ref is not None
    record = await store.get_mint_record(outcome.frame_ref)
    assert record is not None
    assert record.enrichment_stamp is None
    assert b"enrichment_stamp" not in encode_mint_record(record)


async def test_a_frame_is_named_under_the_stamp_of_the_artifact_it_is_read_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One manifest lists a pinned kind file rendered under an earlier code index, which
    apis sends without a stamp, beside a kind file rendered under the current one. Each frame is named
    by its own file's stamp, and the pinned one unsalted."""
    blobs = _bytes_by_digest(2)
    pinned, current = blobs
    summed = _KIND_OPTIONS | {"groupby": _SUM_GROUPBY}

    async def fake_fetch(*, artifact_digest: str, **_: Any) -> bytes:
        return blobs[artifact_digest]

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)
    store = InMemoryMaterialisationStore()
    set_frame_store(store)
    manifest = _manifest(
        renderings={pinned: _KIND_OPTIONS, current: summed}, stamps={current: "stamp-a"}
    )

    for options, stamp in ((_KIND_OPTIONS, None), (summed, "stamp-a")):
        outcome = await materialise_and_register(
            options=options, manifest=manifest, settings=_settings(), writer="tester"
        )
        assert outcome.status == "registered"
        assert outcome.frame_ref == frame_ref_for(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            columns=("party_kind",),
            groupby=options.get("groupby"),
            enrichment_stamp=stamp,
        )
        record = await store.get_mint_record(outcome.frame_ref)
        assert record is not None
        assert record.enrichment_stamp == stamp
        _, trace = await store.get(outcome.frame_ref)
        assert trace.enrichment_stamp == stamp


async def test_a_stamped_default_mint_records_no_stamp() -> None:
    """A default frame is the version's canonical one, so its pair does not depend on the
    stamp and its record, a function of the pair, does not hold one."""
    canonical = _parquet_bytes()
    store = InMemoryMaterialisationStore()
    set_frame_store(store)

    outcome, _snapshot_paths, _fetched = await _register_default(
        snapshot=_snapshot_listing(
            (_digest_for(canonical), dict(CANONICAL_FRAME_RENDERING), _CANONICAL_LINK)
        ),
        served={_CANONICAL_LINK: canonical},
        manifest=_manifest(stamps={_ARTIFACT_DIGEST: "stamp-a"}),
    )

    assert outcome.frame_ref == _FRAME_REF
    assert await store.get_mint_record(_FRAME_REF) == MintRecord(
        frame=_FRAME_REF, dataset=_FRAME_REF, options=_OPTIONS, stat_var_count=1
    )


def test_a_stamped_manifest_selects_the_artifact_whose_rendering_equals_the_options() -> None:
    """The stamp salts the frame's name and takes no part in the selection: apis lists
    one artifact per rendering, and a rendering is the request's options as sent."""
    digests = list(_bytes_by_digest(3))
    manifest = _manifest(
        renderings=dict(zip(digests, [_OPTIONS, _KIND_OPTIONS, _TYPED_OPTIONS], strict=True)),
        stamps={digests[1]: "stamp-a"},
    )

    selected = frame_materialisation._select_requested_artifact(manifest, options=_KIND_OPTIONS)

    assert selected is not None
    assert selected.artifact_digest == digests[1]


# A mint response lists the requested rendering and its statistics. The snapshot of
# the same version lists them and the version's canonical frame.
_RENDERED_LINK = f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}/files/rendered"
_CANONICAL_LINK = f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}/files/canonical"


def _snapshot_listing(*entries: tuple[str, dict[str, object], str]) -> dict[str, Any]:
    """A snapshot response listing one artifact per `(digest, rendering, url)`."""
    payload = _snapshot_payload()
    payload["artifacts"] = [
        {**payload["artifacts"][0], "artifact_digest": digest, "rendering": rendering, "url": url}
        for digest, rendering, url in entries
    ]
    return payload


async def _register_default(
    *,
    snapshot: dict[str, Any] | Callable[[], Awaitable[httpx2.Response]],
    served: dict[str, bytes],
    manifest: Manifest,
    settings: Settings | None = None,
) -> tuple[MaterialisationOutcome, list[str], list[str]]:
    """`materialise_and_register` of a default mint whose snapshot read answers
    `snapshot`, or what it returns, and whose link GETs answer `served` by path, with
    the paths read."""
    snapshot_paths: list[str] = []
    fetched: list[str] = []

    async def snapshot_handler(request: httpx2.Request) -> httpx2.Response:
        snapshot_paths.append(request.url.path)
        if callable(snapshot):
            return await snapshot()
        return httpx2.Response(200, json=snapshot)

    def get_handler(request: httpx2.Request) -> httpx2.Response:
        fetched.append(request.url.path)
        return httpx2.Response(200, content=served[request.url.path])

    async with _client(snapshot_handler) as upstream:
        client_holder.set_client(upstream)
        outcome = await materialise_and_register(
            options=_OPTIONS,
            manifest=manifest,
            settings=settings or _settings(),
            writer="tester",
            transport=httpx2.MockTransport(get_handler),
        )
    return outcome, snapshot_paths, fetched


async def test_a_default_mint_registers_the_snapshots_canonical_frame() -> None:
    """The rendering the mint lists carries a decoded column the canonical frame does
    not, so registering it would hold a frame its pair does not name."""
    canonical = _parquet_bytes()
    rendered = _decoded_column_parquet()
    manifest = _manifest(artifact_digest=_digest_for(rendered))
    snapshot = _snapshot_listing(
        (_digest_for(rendered), _OPTIONS, _RENDERED_LINK),
        (_digest_for(canonical), dict(CANONICAL_FRAME_RENDERING), _CANONICAL_LINK),
    )
    set_frame_store(InMemoryMaterialisationStore())

    outcome, snapshot_paths, fetched = await _register_default(
        snapshot=snapshot,
        served={_RENDERED_LINK: rendered, _CANONICAL_LINK: canonical},
        manifest=manifest,
    )

    assert outcome.status == "registered"
    assert outcome.frame_ref == _FRAME_REF
    assert snapshot_paths == [f"/api/datasets/{_SPEC_ID}/{_CONTENT_VERSION}"]
    assert fetched == [_CANONICAL_LINK]
    held = await get_frame_store().get(_FRAME_REF)
    assert held is not None
    assert mint_content_version(frame_sha256=frame_sha256(held[0])) == _CONTENT_VERSION
    assert held[0] == frame_from_verified_parquet_bytes(canonical)
    assert held[1].options == _OPTIONS


async def test_a_snapshot_without_a_canonical_frame_refuses_without_fetching() -> None:
    rendered = _parquet_bytes()
    set_frame_store(InMemoryMaterialisationStore())

    outcome, _snapshot_paths, fetched = await _register_default(
        snapshot=_snapshot_listing((_digest_for(rendered), _OPTIONS, _RENDERED_LINK)),
        served={_RENDERED_LINK: rendered},
        manifest=_manifest(artifact_digest=_digest_for(rendered)),
    )

    assert outcome.status == "refused"
    assert outcome.attempts == 0
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert outcome.refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert fetched == []
    assert await get_frame_store().get(_FRAME_REF) is None


async def _unavailable() -> httpx2.Response:
    return httpx2.Response(503, json={"code": "service_unavailable", "retryable": True})


async def _slower_than_the_budget() -> httpx2.Response:
    await asyncio.sleep(1.0)
    return httpx2.Response(200, json=_snapshot_payload())


@pytest.mark.parametrize(
    ("snapshot", "budget_s"),
    [(_unavailable, 5.0), (_slower_than_the_budget, 0.05)],
    ids=["snapshot-unavailable", "snapshot-past-the-budget"],
)
async def test_a_failed_snapshot_read_degrades_without_fetching(
    snapshot: Callable[[], Awaitable[httpx2.Response]], budget_s: float
) -> None:
    canonical = _parquet_bytes()
    set_frame_store(InMemoryMaterialisationStore())

    outcome, snapshot_paths, fetched = await _register_default(
        snapshot=snapshot,
        served={_CANONICAL_LINK: canonical},
        manifest=_manifest(),
        settings=_settings(artifact_fetch_budget_s=budget_s),
    )

    assert outcome.status == "degraded"
    assert outcome.degraded_reason == "fetch"
    assert outcome.attempts == 0
    assert snapshot_paths
    assert fetched == []
    assert await get_frame_store().get(_FRAME_REF) is None
    assert await get_frame_store().get_mint_record(_FRAME_REF) is None


async def test_a_canonical_frame_that_does_not_mint_its_content_version_refuses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The bytes match the digest the snapshot lists, and their frame hashes to
    another version than the pair names. The refusal names the pair, and a warning
    records it with the token the frame re-mints."""
    other = _parquet_bytes(values=[2.5])
    set_frame_store(InMemoryMaterialisationStore())

    outcome, _snapshot_paths, fetched = await _register_default(
        snapshot=_snapshot_listing(
            (_digest_for(other), dict(CANONICAL_FRAME_RENDERING), _CANONICAL_LINK)
        ),
        served={_CANONICAL_LINK: other},
        manifest=_manifest(),
    )

    assert _content_version_of(other) != _CONTENT_VERSION
    assert outcome.status == "refused"
    assert outcome.attempts == 1
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert outcome.refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert fetched == [_CANONICAL_LINK]
    assert _SPEC_ID in outcome.refusal.message
    assert _CONTENT_VERSION in outcome.refusal.message
    assert [
        record.args
        for record in caplog.records
        if record.name == frame_materialisation.__name__ and record.levelname == "WARNING"
    ] == [(_SPEC_ID, _CONTENT_VERSION, _content_version_of(other))]
    assert await get_frame_store().get(_FRAME_REF) is None
    assert await get_frame_store().get_mint_record(_FRAME_REF) is None


def _reordered_canonical_parquet(order: list[int]) -> bytes:
    """A two-row canonical frame's table in the canonical schema with its rows taken
    in `order`: the schema `frame_from_verified_parquet_bytes` wraps, and rows no
    `Frame` holds in that order."""
    table = pq.read_table(
        io.BytesIO(
            _parquet_bytes(
                role_values=["country/FRA", "country/KEN"],
                time_periods=["2020", "2020"],
                values=[1.5, 2.5],
            )
        )
    ).replace_schema_metadata(None)
    buf = io.BytesIO()
    pq.write_table(table.take(order), buf)
    return buf.getvalue()


@pytest.mark.parametrize("order", [[1, 0], [0, 0]], ids=["out-of-order", "duplicated"])
async def test_a_canonical_frame_whose_rows_no_frame_holds_refuses(order: list[int]) -> None:
    """The schema is canonical, so the decode wraps the table, and hashing it is
    what finds the rows out of order."""
    data = _reordered_canonical_parquet(order)
    set_frame_store(InMemoryMaterialisationStore())

    outcome, _snapshot_paths, fetched = await _register_default(
        snapshot=_snapshot_listing(
            (_digest_for(data), dict(CANONICAL_FRAME_RENDERING), _CANONICAL_LINK)
        ),
        served={_CANONICAL_LINK: data},
        manifest=_manifest(),
    )

    assert outcome.status == "refused"
    assert outcome.attempts == 1
    assert outcome.refusal is not None
    assert outcome.refusal.code == ToolErrorCode.UPSTREAM_ERROR
    assert outcome.refusal.retry_mode == RetryMode.DO_NOT_RETRY
    assert fetched == [_CANONICAL_LINK]
    assert await get_frame_store().get(_FRAME_REF) is None
    assert await get_frame_store().get_mint_record(_FRAME_REF) is None


def _decoded_column_parquet() -> bytes:
    """`_parquet_bytes()`'s row in the rendering's frame layout, with the decoded
    `stat_var_dac2aAidType` column apis adds to it."""
    table = pq.read_table(io.BytesIO(_parquet_bytes())).replace_schema_metadata(None)
    table = table.add_column(2, "stat_var_dac2aAidType", pa.array(["A01"], type=pa.string()))
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


async def test_a_groupby_mint_registers_its_rendering_under_the_pair_naming_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sum is held apart from the version's canonical frame, read from the
    rendering the mint lists and with no snapshot read."""
    summed = _parquet_bytes(values=[9.0], stat_vars=[f"dcid:{_SPEC_ID}"])
    fetched: list[tuple[str, str | None]] = []

    async def fake_fetch(*, artifact_digest: str, link: str | None, **_: Any) -> bytes:
        fetched.append((artifact_digest, link))
        return summed

    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", fake_fetch)
    set_frame_store(InMemoryMaterialisationStore())

    outcome = await materialise_and_register(
        options=_SUM_SIBLING_RENDERING,
        manifest=_manifest(renderings={_digest_for(summed): _SUM_SIBLING_RENDERING}),
        settings=_settings(),
        writer="tester",
    )

    expected = frame_ref_for(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION, groupby=_SUM_GROUPBY
    )
    assert outcome.status == "registered"
    assert outcome.frame_ref == expected
    assert expected != _FRAME_REF
    assert fetched == [(_digest_for(summed), None)]
    held = await get_frame_store().get(expected)
    assert held is not None
    assert held[1].options == _SUM_SIBLING_RENDERING
    assert await get_frame_store().get(_FRAME_REF) is None
    assert await get_frame_store().get_mint_record(expected) == MintRecord(
        frame=expected, dataset=_FRAME_REF, options=_SUM_SIBLING_RENDERING, stat_var_count=1
    )
