"""Acceptance test for the durable, bucket-backed store: a toolkit chain
split across a process restart produces the identical result an unbroken
in-memory run of the same chain produces.

Seeds a root frame into a `GcsMaterialisationStore` over a fake bucket, runs
`aggregate`, then reads back through a second, independently constructed
`GcsMaterialisationStore` over the same backing dicts (standing in for a
process restart -- no Python state survives between the two, only what the
fake bucket itself holds) to run `growth` over the aggregate's own
reference. The final `ocv_` must equal what an unbroken run of the identical
two-step chain against a plain `InMemoryMaterialisationStore` produces.

The fake `_StorageClient`/`_Bucket`/`_Blob` here duplicate
`packages/common/tests/test_gcs_store.py`'s own fakes, since a service's
tests don't import another package's own test module.
"""

from __future__ import annotations

from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.fixtures import plain_single_unit_frame
from analysis_mcp.growth_tool import growth
from common.materialisation import (
    FrameRef,
    GcsMaterialisationStore,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    set_store,
)
from fastmcp.tools import ToolResult
from google.api_core.exceptions import NotFound, PreconditionFailed

_BUCKET = "dev-fin-observatory-frames"

_ROOT_REF = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "1" * 32)


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
    def __init__(self, objects: dict[str, bytes], metadata: dict[str, dict[str, str]]) -> None:
        self._objects = objects
        self._metadata = metadata

    def bucket(self, name: str) -> _FakeBucket:
        assert name == _BUCKET
        return _FakeBucket(self._objects, self._metadata)

    def close(self) -> None:
        pass


def _root_trace() -> MaterialisationTrace:
    return MaterialisationTrace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": "1"}],
            "facets": [],
            "conversion": None,
        },
        spec_id=_ROOT_REF.spec_id,
        content_version=_ROOT_REF.content_version,
    )


async def _run_aggregate() -> ToolResult:
    return await aggregate(
        frame_ref={"spec_id": _ROOT_REF.spec_id, "content_version": _ROOT_REF.content_version},
        aggregation="sum",
        group_by=("reporter", "time_period"),
    )


async def _run_growth(aggregate_result: ToolResult) -> ToolResult:
    return await growth(
        frame_ref={
            "spec_id": aggregate_result.structured_content["spec_id"],
            "content_version": aggregate_result.structured_content["content_version"],
        },
        window_start="2021",
        window_end="2023",
        entity_by=("reporter",),
    )


async def test_a_chain_split_across_a_restart_matches_an_unbroken_in_memory_run() -> None:
    objects: dict[str, bytes] = {}
    metadata: dict[str, dict[str, str]] = {}

    first_store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    await first_store.put(
        _ROOT_REF, frame=plain_single_unit_frame(), trace=_root_trace(), writer="test"
    )
    set_store(first_store)
    try:
        aggregate_result = await _run_aggregate()
    finally:
        set_store(None)
    assert aggregate_result.is_error is False, aggregate_result.content

    # Stands in for a process restart: a second `GcsMaterialisationStore`
    # instance, over the same backing bucket, sharing no Python state with
    # the first -- only what actually landed in `objects`/`metadata`.
    second_store = GcsMaterialisationStore(
        bucket_name=_BUCKET, client=_FakeStorageClient(objects, metadata)
    )
    set_store(second_store)
    try:
        growth_result = await _run_growth(aggregate_result)
    finally:
        set_store(None)
    assert growth_result.is_error is False, growth_result.content

    in_memory_store = InMemoryMaterialisationStore()
    await in_memory_store.put(
        _ROOT_REF, frame=plain_single_unit_frame(), trace=_root_trace(), writer="test"
    )
    set_store(in_memory_store)
    try:
        in_memory_aggregate_result = await _run_aggregate()
        assert in_memory_aggregate_result.is_error is False, in_memory_aggregate_result.content
        in_memory_growth_result = await _run_growth(in_memory_aggregate_result)
    finally:
        set_store(None)
    assert in_memory_growth_result.is_error is False, in_memory_growth_result.content

    assert (
        growth_result.structured_content["content_version"]
        == in_memory_growth_result.structured_content["content_version"]
    )
