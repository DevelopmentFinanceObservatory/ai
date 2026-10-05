"""GCS-backed `DefinitionStore`.

Object name `v{schema_version}/{spec_id}`, uploaded with `content_type=
"application/json"` and `if_generation_match=0` -- the write only succeeds
if the object does not yet exist. Every definition is content-addressed by
construction (`identity.verify_write` checked the writer's own hash before
`put_many` was ever called), so a `PreconditionFailed` (412) means the bytes
are already held under this exact name, by whichever writer's identical
write landed first -- treated as success, not an error.

`get` re-hashes the downloaded bytes against the spec_id they were requested
under (`identity.verify_read`) before returning them, so a tampered or
bit-rotted object is refused rather than served.

The store holds the identity and nothing else. A bucket may also hold
companion objects beside a definition, named `.selection`,
`.elected_stat_vars` or `.unconstrained_stat_var`; this store never reads or
writes them, and the bucket's lifecycle rule clears them like any other
object.

`asyncio.to_thread` crosses the synchronous `google-cloud-storage` client
into this async protocol, matching the discipline in
`dfo-apis/src/artifact_store/inventory.py`. Retention is the bucket's own
lifecycle rule alone: no sweep and no read-side window filter live here.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Sequence
from typing import Protocol, cast

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from data_mcp.definition_store.errors import DefinitionStoreUnavailable
from data_mcp.definition_store.identity import verify_read
from data_mcp.definition_store.protocol import StoredDefinition

_CONTENT_TYPE = "application/json"


class _Blob(Protocol):
    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> object: ...
    def download_as_bytes(self) -> bytes: ...


class _Bucket(Protocol):
    def blob(self, name: str) -> _Blob: ...
    def exists(self) -> bool: ...


class _StorageClient(Protocol):
    """The narrow slice of `google.cloud.storage.Client` this module calls.

    `google-cloud-storage`'s own annotations are uneven; a test's mock is
    checked against this Protocol instead of the real client, so calls into
    it don't need scattered ignore comments.
    """

    def bucket(self, name: str) -> _Bucket: ...
    def close(self) -> None: ...


class GCSDefinitionStore:
    """`google-cloud-storage`-backed `DefinitionStore`.

    `client` is injected, never constructed here, so a test can run under the
    suite's `no_network` fixture, which raises on any `AF_INET` socket -- a
    bare `storage.Client()` reaches the metadata server for ADC. `bucket_name`
    is read from config by the caller constructing this, never hardcoded.
    """

    def __init__(self, *, bucket_name: str, client: storage.Client | None = None) -> None:
        self._bucket_name = bucket_name
        # `storage.Client`'s own methods carry no annotations, so `ty` can't
        # verify it satisfies `_StorageClient` structurally -- it does at
        # runtime, exercised by every test in this module that injects a
        # fake in its place. One cast here beats an ignore comment on every
        # call below.
        self._client = cast("_StorageClient", client if client is not None else storage.Client())
        # Set once, by `_bucket_missing`, the first time a download 404s.
        # `_get_one` runs on `asyncio.to_thread` worker threads, so this is
        # guarded by `_bucket_probe_lock` rather than left as a bare
        # read/write.
        self._bucket_missing_verdict: bool | None = None
        self._bucket_probe_lock = threading.Lock()

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        deduped = {(d.spec_id, d.schema_version): d for d in definitions}
        results = await asyncio.gather(
            *(asyncio.to_thread(self._put_one, definition) for definition in deduped.values()),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise DefinitionStoreUnavailable(
                f"{len(failures)} of {len(results)} definition write(s) failed"
            ) from failures[0]

    def _put_one(self, definition: StoredDefinition) -> None:
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(_object_name(definition))
        # The object already existing under this content-addressed name means
        # someone else's identical write won the race -- success, not an error.
        with contextlib.suppress(PreconditionFailed):
            blob.upload_from_string(
                definition.canonical_bytes,
                content_type=_CONTENT_TYPE,
                if_generation_match=0,
            )

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        try:
            canonical_bytes = await asyncio.to_thread(self._get_one, spec_id, schema_version)
        except Exception as exc:
            raise DefinitionStoreUnavailable(
                f"could not read {spec_id!r} (schema_version={schema_version}) from GCS"
            ) from exc
        if canonical_bytes is None:
            return None
        # Outside the try above: an `IdentityMismatch` is deterministic and
        # must reach the caller as itself, not get folded into
        # `DefinitionStoreUnavailable` and retried as if the backend were
        # merely unreachable.
        verify_read(canonical_bytes, spec_id)
        return StoredDefinition(
            spec_id=spec_id, schema_version=schema_version, canonical_bytes=canonical_bytes
        )

    def _get_one(self, spec_id: str, schema_version: int) -> bytes | None:
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(_object_name_for(spec_id, schema_version))
        try:
            return blob.download_as_bytes()
        except NotFound:
            # A missing bucket and a missing object both raise `NotFound`
            # from `download_as_bytes`, so a 404 here is ambiguous until the
            # bucket itself is checked. Probed only on this path -- a normal
            # hit costs no extra call.
            if self._bucket_missing(bucket):
                raise
            return None

    def _bucket_missing(self, bucket: _Bucket) -> bool:
        """Report whether `self._bucket_name` itself does not exist.

        Probed once and cached for the life of this store: a deleted or
        misconfigured bucket doesn't come back, so every later 404 can trust
        the first verdict without paying for another round trip.

        A probe failure (the bucket check itself unreachable) records no
        verdict and reports "not missing", leaving the object 404 that
        triggered the probe as a genuine miss rather than replacing it with
        a guess -- mirrors `UpstreamClient._probe_surface`'s "never replace
        one error with the probe's own failure".
        """
        with self._bucket_probe_lock:
            if self._bucket_missing_verdict is not None:
                return self._bucket_missing_verdict
            try:
                exists = bucket.exists()
            except Exception:
                return False
            self._bucket_missing_verdict = not exists
            return self._bucket_missing_verdict

    async def aclose(self) -> None:
        await asyncio.to_thread(self._client.close)


def _object_name(definition: StoredDefinition) -> str:
    return _object_name_for(definition.spec_id, definition.schema_version)


def _object_name_for(spec_id: str, schema_version: int) -> str:
    return f"v{schema_version}/{spec_id}"
