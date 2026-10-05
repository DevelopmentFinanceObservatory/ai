"""The frame row budget against `GcsMaterialisationStore`, whose `open` reads only the
object's metadata: a verb claims its inputs' rows before any byte of them is
downloaded, so a call refused for capacity downloads nothing, and a call cancelled
mid-download keeps its rows claimed until the download's worker thread returns. An
object stored without a recorded row count claims the service's `max_frame_rows`, and
its read refuses a footer over that claim before decoding it.

The fake `_StorageClient`/`_Bucket`/`_Blob` here duplicate
`packages/common/tests/test_gcs_store.py`'s own fakes, since a service's tests don't
import another package's own test module (see `test_frame_store_restart.py`'s own
docstring). Their downloads are recorded and wait on a gate a test can close.
"""

from __future__ import annotations

import asyncio
import threading

import pyarrow as pa
import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.fixtures import (
    join_compatible_frame_pair,
    plain_single_unit_frame,
    seed_fixture_store,
)
from analysis_mcp.join_tool import join
from common.materialisation import (
    FrameRef,
    GcsMaterialisationStore,
    MaterialisationTrace,
    gcs_store,
    get_store,
    set_store,
)
from common.row_budget import RowBudget, row_budget, set_row_budget
from dfo_frame_grammar import Frame
from google.api_core.exceptions import NotFound, PreconditionFailed

_BUCKET = "dev-fin-observatory-frames"

# A lowered `MAX_FRAME_ROWS`, under the store's own 250,000-row ceiling.
_LOWERED_MAX_FRAME_ROWS = 100_000


class _FakeStorageClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.downloads: list[str] = []
        self.downloading = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def bucket(self, name: str) -> _FakeBucket:
        assert name == _BUCKET
        return _FakeBucket(self)

    def close(self) -> None:
        pass


class _FakeBucket:
    def __init__(self, client: _FakeStorageClient) -> None:
        self._client = client

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._client, name)


class _FakeBlob:
    def __init__(self, client: _FakeStorageClient, name: str) -> None:
        self._client = client
        self._name = name
        self.metadata: dict[str, str] | None = None
        self.size: int | None = None

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert if_generation_match == 0
        if self._name in self._client.objects:
            raise PreconditionFailed("already exists")
        self._client.objects[self._name] = data
        self._client.metadata[self._name] = dict(self.metadata or {})

    def download_as_bytes(self) -> bytes:
        self._client.downloads.append(self._name)
        self._client.downloading.set()
        self._client.gate.wait()
        return self._client.objects[self._name]

    def reload(self) -> None:
        if self._name not in self._client.objects:
            raise NotFound(f"no such object: {self._name}")
        self.metadata = dict(self._client.metadata[self._name])
        self.size = len(self._client.objects[self._name])


@pytest.fixture
async def gcs_seeded():
    client = _FakeStorageClient()
    store = GcsMaterialisationStore(bucket_name=_BUCKET, client=client)
    handles = await seed_fixture_store(store)
    set_store(store)
    yield client, handles
    set_store(None)


def _join_rows() -> int:
    left, right = join_compatible_frame_pair()
    return left.num_rows + right.num_rows


async def test_a_join_refused_for_capacity_downloads_neither_side(gcs_seeded) -> None:
    client, handles = gcs_seeded
    left, right = handles.join_pair
    set_row_budget(RowBudget(max_rows=_join_rows()))

    with row_budget().claim(1):
        refused = await join(frame_ref=left, right_frame_ref=right)

    assert refused.structured_content["code"] == "service_unavailable"
    assert client.downloads == []


async def test_a_join_cancelled_mid_download_keeps_its_rows_claimed_until_the_download_ends(
    gcs_seeded,
) -> None:
    client, handles = gcs_seeded
    left, right = handles.join_pair
    client.gate.clear()
    call = asyncio.create_task(join(frame_ref=left, right_frame_ref=right))
    try:
        assert await asyncio.to_thread(client.downloading.wait, 5)
        assert row_budget().in_flight == _join_rows()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        assert row_budget().in_flight == _join_rows()
    finally:
        client.gate.set()
    async with asyncio.timeout(5):
        while row_budget().in_flight:
            await asyncio.sleep(0.01)


def _object_name(handle: dict[str, str]) -> str:
    return f"frames/v2/{handle['spec_id']}/{handle['content_version']}.parquet"


async def _put_rows(rows: int) -> dict[str, str]:
    """A frame of `rows` copies of the plain fixture's one row, stored with its row
    count recorded."""
    one_row = plain_single_unit_frame()
    frame = Frame.from_table(
        columns=one_row.columns, table=one_row.table.take(pa.array([0] * rows))
    )
    ref = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "d" * 32)
    trace = MaterialisationTrace(
        upstream_trace={}, spec_id=ref.spec_id, content_version=ref.content_version
    )
    await get_store().put(ref, frame=frame, trace=trace, writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


@pytest.fixture
def lowered_max_frame_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FRAME_ROWS", str(_LOWERED_MAX_FRAME_ROWS))


@pytest.mark.usefixtures("lowered_max_frame_rows")
async def test_an_object_with_no_recorded_row_count_resolves_under_a_lowered_ceiling(
    gcs_seeded,
) -> None:
    client, handles = gcs_seeded
    del client.metadata[_object_name(handles.plain_single_unit)]["frame_rows"]

    result = await aggregate(frame_ref=handles.plain_single_unit, aggregation="sum")

    assert result.is_error is False
    assert row_budget().high_water == _LOWERED_MAX_FRAME_ROWS
    assert row_budget().in_flight == 0


@pytest.mark.usefixtures("lowered_max_frame_rows")
async def test_an_object_with_no_recorded_row_count_over_the_ceiling_refuses_without_decoding(
    gcs_seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _handles = gcs_seeded
    handle = await _put_rows(_LOWERED_MAX_FRAME_ROWS + 1)
    del client.metadata[_object_name(handle)]["frame_rows"]

    def _fail_if_called(data: bytes) -> Frame:
        raise AssertionError("the decode must not run past the claim")

    monkeypatch.setattr(gcs_store, "frame_from_verified_parquet_bytes", _fail_if_called)

    result = await aggregate(frame_ref=handle, aggregation="sum")

    assert result.structured_content["code"] == "frame_too_large"
    assert result.structured_content["fix"] == {
        "reason": f"stored frame has {_LOWERED_MAX_FRAME_ROWS + 1} rows, over this read's "
        f"{_LOWERED_MAX_FRAME_ROWS}-row limit"
    }
    assert row_budget().in_flight == 0


@pytest.mark.usefixtures("lowered_max_frame_rows")
async def test_a_recorded_row_count_over_the_ceiling_refuses_before_downloading(
    gcs_seeded,
) -> None:
    client, _handles = gcs_seeded
    handle = await _put_rows(_LOWERED_MAX_FRAME_ROWS + 1)

    result = await aggregate(frame_ref=handle, aggregation="sum")

    assert result.structured_content["code"] == "frame_too_large"
    assert result.structured_content["fix"] == {
        "row_count": _LOWERED_MAX_FRAME_ROWS + 1,
        "ceiling": _LOWERED_MAX_FRAME_ROWS,
    }
    assert client.downloads == []
