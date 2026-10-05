"""Tests for `GCSDefinitionStore` against an injected mock client.

`no_network` (autouse, from `net_guard.py`) blocks every real socket, so
every test here injects `client` explicitly -- never the real
`google.cloud.storage.Client()`, whose construction can reach the metadata
server for Application Default Credentials.
"""

from __future__ import annotations

from typing import Any

import pytest
from data_mcp.definition_store.errors import DefinitionStoreUnavailable
from data_mcp.definition_store.gcs import GCSDefinitionStore
from data_mcp.definition_store.identity import IdentityMismatch, spec_id_for
from data_mcp.definition_store.protocol import StoredDefinition
from google.api_core.exceptions import NotFound, PreconditionFailed, ServiceUnavailable

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture

_BUCKET = "dev-fin-observatory-definitions"


class _FakeBlob:
    def __init__(self, objects: dict[str, bytes], name: str, upload_log: list[str]) -> None:
        self._objects = objects
        self._name = name
        self._upload_log = upload_log

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert content_type == "application/json"
        assert if_generation_match == 0
        self._upload_log.append(self._name)
        if self._name in self._objects:
            raise PreconditionFailed("already exists")
        self._objects[self._name] = data

    def download_as_bytes(self) -> bytes:
        if self._name not in self._objects:
            raise NotFound(f"no such object: {self._name}")
        return self._objects[self._name]


class _FakeBucket:
    def __init__(self, objects: dict[str, bytes], upload_log: list[str]) -> None:
        self._objects = objects
        self._upload_log = upload_log

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._objects, name, self._upload_log)

    def exists(self) -> bool:
        return True


class _FakeStorageClient:
    """Fakes the narrow `_StorageClient` seam `gcs.py` calls through."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_log: list[str] = []
        self.closed = False

    def bucket(self, name: str) -> _FakeBucket:
        assert name == _BUCKET
        return _FakeBucket(self.objects, self.upload_log)

    def close(self) -> None:
        self.closed = True


class _AlwaysFailingBlob(_FakeBlob):
    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        raise ServiceUnavailable("write path is down")

    def download_as_bytes(self) -> bytes:
        raise ServiceUnavailable("read path is down")


class _PartiallyFailingClient(_FakeStorageClient):
    """Every write to `_failing_spec_id` raises; everything else succeeds."""

    def __init__(self, failing_spec_id: str) -> None:
        super().__init__()
        self._failing_spec_id = failing_spec_id

    def bucket(self, name: str) -> Any:
        return self

    def blob(self, name: str) -> _FakeBlob:
        if self._failing_spec_id in name:
            return _AlwaysFailingBlob(self.objects, name, self.upload_log)
        return _FakeBlob(self.objects, name, self.upload_log)


class _MissingBucket:
    """Stands in for a bucket that does not exist: every blob download 404s, `exists()` reports False."""

    def __init__(self, probe_log: list[str]) -> None:
        self._probe_log = probe_log

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob({}, name, [])

    def exists(self) -> bool:
        self._probe_log.append("probed")
        return False


class _MissingBucketClient:
    """Fakes a client whose configured bucket does not exist."""

    def __init__(self) -> None:
        self.probe_log: list[str] = []

    def bucket(self, name: str) -> _MissingBucket:
        return _MissingBucket(self.probe_log)

    def close(self) -> None:
        pass


def _definition(
    spec_id: str, schema_version: int = 1, payload: bytes = b'{"n":1}'
) -> StoredDefinition:
    return StoredDefinition(spec_id=spec_id, schema_version=schema_version, canonical_bytes=payload)


async def test_put_many_writes_under_v_schema_version_slash_spec_id() -> None:
    client = _FakeStorageClient()
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)
    definition = _definition("spec_abc", schema_version=1)

    await store.put_many([definition])

    assert client.objects == {"v1/spec_abc": definition.canonical_bytes}


async def test_put_many_treats_a_412_as_success() -> None:
    client = _FakeStorageClient()
    client.objects["v1/spec_abc"] = b"already here"
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    await store.put_many(
        [_definition("spec_abc", schema_version=1, payload=b"whatever")]
    )  # no raise

    # The pre-existing bytes are untouched -- this store never overwrites.
    assert client.objects["v1/spec_abc"] == b"already here"


async def test_put_many_dedups_repeated_spec_ids_before_writing() -> None:
    client = _FakeStorageClient()
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)
    same_definition = _definition("spec_dup", schema_version=1)

    await store.put_many([same_definition, same_definition])

    assert client.upload_log.count("v1/spec_dup") == 1


async def test_put_many_raises_once_after_attempting_every_write() -> None:
    client = _PartiallyFailingClient(failing_spec_id="spec_bad")
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    with pytest.raises(DefinitionStoreUnavailable):
        await store.put_many([_definition("spec_bad"), _definition("spec_good")])

    # The sibling write still landed -- `return_exceptions=True` let it run
    # to completion instead of being cancelled by the first failure.
    assert "v1/spec_good" in client.objects


async def test_get_returns_the_stored_definition_on_a_hit() -> None:
    client = _FakeStorageClient()
    canonical_bytes = b'{"n":1}'
    spec_id = spec_id_for(canonical_bytes)
    client.objects[f"v1/{spec_id}"] = canonical_bytes
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    result = await store.get(spec_id, schema_version=1)

    assert result == _definition(spec_id, schema_version=1, payload=canonical_bytes)


async def test_get_refuses_bytes_that_do_not_hash_to_the_requested_spec_id() -> None:
    client = _FakeStorageClient()
    # "spec_abc" is not `spec_id_for(b'{"n":1}')` -- stands in for a tampered
    # or bit-rotted object.
    client.objects["v1/spec_abc"] = b'{"n":1}'
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    with pytest.raises(IdentityMismatch):
        await store.get("spec_abc", schema_version=1)


async def test_get_returns_none_on_a_genuine_miss() -> None:
    client = _FakeStorageClient()
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    result = await store.get("spec_missing", schema_version=1)

    assert result is None


async def test_get_raises_definition_store_unavailable_when_the_backend_is_unreachable() -> None:
    client = _PartiallyFailingClient(failing_spec_id="spec_down")
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    with pytest.raises(DefinitionStoreUnavailable):
        await store.get("spec_down", schema_version=1)


async def test_get_returns_none_when_the_bucket_exists_but_the_object_does_not() -> None:
    client = _FakeStorageClient()
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    result = await store.get("spec_missing", schema_version=1)

    assert result is None


async def test_get_raises_definition_store_unavailable_when_the_bucket_does_not_exist() -> None:
    client = _MissingBucketClient()
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    with pytest.raises(DefinitionStoreUnavailable):
        await store.get("spec_a", schema_version=1)
    # A second miss must not fire a second probe -- the first verdict is
    # cached for the life of the store.
    with pytest.raises(DefinitionStoreUnavailable):
        await store.get("spec_b", schema_version=1)

    assert client.probe_log == ["probed"]


# The identity alone: companion objects beside it are neither written nor read


async def test_put_many_writes_the_identity_object_alone() -> None:
    client = _FakeStorageClient()
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    await store.put_many([_definition("spec_abc", schema_version=1)])

    assert client.objects == {"v1/spec_abc": b'{"n":1}'}


async def test_get_never_reads_a_companion_object() -> None:
    """Every companion name a bucket may hold fails on read here, and `get` still
    returns the identity, so none of them is ever downloaded."""
    canonical_bytes = b'{"n":1}'
    spec_id = spec_id_for(canonical_bytes)

    class _CompanionFailingClient(_FakeStorageClient):
        def bucket(self, name: str) -> Any:
            return self

        def blob(self, name: str) -> _FakeBlob:
            if name != f"v1/{spec_id}":
                return _AlwaysFailingBlob(self.objects, name, self.upload_log)
            return _FakeBlob(self.objects, name, self.upload_log)

    client = _CompanionFailingClient()
    client.objects[f"v1/{spec_id}"] = canonical_bytes
    for suffix in (".selection", ".elected_stat_vars", ".unconstrained_stat_var"):
        client.objects[f"v1/{spec_id}{suffix}"] = b'["Count_Person"]'
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    result = await store.get(spec_id, schema_version=1)

    assert result == StoredDefinition(
        spec_id=spec_id, schema_version=1, canonical_bytes=canonical_bytes
    )


async def test_aclose_closes_the_injected_client() -> None:
    client = _FakeStorageClient()
    store = GCSDefinitionStore(bucket_name=_BUCKET, client=client)

    await store.aclose()

    assert client.closed is True
