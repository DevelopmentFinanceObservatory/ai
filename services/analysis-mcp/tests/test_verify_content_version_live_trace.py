"""`GcsMaterialisationStore.get` with `verify_content_version_on_read=True`,
driven by the real registered deriver (`analysis_mcp.frame_ref_resolution
.derive_frame_content_version`, read back through `common.materialisation
.get_content_version_deriver` exactly as `build_store` sources it) over a
frame registered under a live-shaped `upstream_trace` -- not a synthetic
lambda.

`packages/common/tests/test_gcs_store.py`'s own `verify_content_version_on_read`
coverage injects a lambda that ignores its input entirely, so it never
exercises the deriver a real deployment relies on. This module does, with a
real `put`/`get` round trip.

The fake `_StorageClient`/`_Bucket`/`_Blob` here duplicate
`packages/common/tests/test_gcs_store.py`'s own fakes, since a service's
tests don't import another package's own test module (see
`test_frame_store_restart.py`'s own docstring).
"""

from __future__ import annotations

from dataclasses import replace

from analysis_mcp.fixtures import live_shaped_trace, plain_single_unit_frame
from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from common.materialisation import FrameRef, GcsMaterialisationStore, get_content_version_deriver
from dfo_frame_grammar import canonical_frame_bytes
from google.api_core.exceptions import NotFound, PreconditionFailed

_BUCKET = "dev-fin-observatory-frames"


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
    def __init__(
        self,
        objects: dict[str, bytes] | None = None,
        metadata: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.objects: dict[str, bytes] = objects if objects is not None else {}
        self.metadata: dict[str, dict[str, str]] = metadata if metadata is not None else {}

    def bucket(self, name: str) -> _FakeBucket:
        assert name == _BUCKET
        return _FakeBucket(self.objects, self.metadata)

    def close(self) -> None:
        pass


async def test_get_verifies_a_live_shaped_trace_through_the_real_deriver() -> None:
    """Importing `analysis_mcp.frame_ref_resolution` (above) already
    registered the real deriver at module import time; `get_content_version_deriver`
    reads back that same callable, exactly as `common.materialisation.factory
    .build_store` sources the check it wires into a durable store.

    `ref.content_version` is minted by `derive_frame_content_version`
    rather than an arbitrary stand-in, since `GcsMaterialisationStore.get`
    checks a root reference's recomputed content_version against `ref` itself.
    """
    frame = plain_single_unit_frame()
    live_trace = live_shaped_trace("live_plain_single_unit", frame)
    content_version = derive_frame_content_version(frame=frame)
    ref = FrameRef(spec_id=live_trace.spec_id, content_version=content_version)
    trace = replace(live_trace, content_version=content_version)
    deriver = get_content_version_deriver()
    assert deriver is not None
    store = GcsMaterialisationStore(
        bucket_name=_BUCKET,
        client=_FakeStorageClient(),
        verify_content_version_on_read=True,
        derive_content_version=deriver,
    )

    await store.put(ref, frame=frame, trace=trace, writer="test")
    result = await store.get(ref)

    assert result is not None
    round_tripped_frame, round_tripped_trace = result
    assert canonical_frame_bytes(round_tripped_frame) == canonical_frame_bytes(frame)
    assert round_tripped_trace.upstream_trace == trace.upstream_trace
