"""Picks the `MaterialisationStore` implementation a service's own lifespan
installs: `GcsMaterialisationStore` when a bucket is named, the in-memory
store otherwise.
"""

from __future__ import annotations

import logging
import os

from common.materialisation.content_version import (
    ContentVersionDeriverUnregistered,
    get_content_version_deriver,
)
from common.materialisation.gcs_store import GcsMaterialisationStore
from common.materialisation.holder import set_store, store_is_set
from common.materialisation.memory_store import InMemoryMaterialisationStore
from common.materialisation.store import MaterialisationStore

logger = logging.getLogger(__name__)


def build_store(
    *, bucket_name: str | None, verify_content_version_on_read: bool = False
) -> MaterialisationStore:
    """`GcsMaterialisationStore(bucket_name=bucket_name, ...)` when
    `bucket_name` is given, an unseeded `InMemoryMaterialisationStore()`
    otherwise. `verify_content_version_on_read` is forwarded to the GCS
    store; the in-memory store ignores it, since it holds live `Frame`
    objects with nothing to re-derive.

    Does not take a `derive_content_version` callable from its own caller --
    that would let whichever service calls this first pick the derivation
    every installed store checks against. When `verify_content_version_on_read`
    is set, the deriver instead comes from
    `common.materialisation.content_version`'s own registry, the one
    `analysis_mcp.frame_ref_resolution` registers into at its own import
    time, so every caller gets the identical deriver regardless of which
    service's lifespan builds the store. Raises `ContentVersionDeriverUnregistered`
    if the check is requested and nothing has registered a deriver: a check
    silently skipped is worse than a startup that refuses.

    The GCS store's own size/row/column ceilings stay its own module-level
    defaults, never threaded from a service's own settings: more than one
    service can build a store over the same bucket, and per-service ceilings
    would let them diverge on what the bucket itself serves back.

    This is the one place that decides which store implementation a process
    gets, called at most once per process (`install_store_if_unset` and each
    service's own lifespan guard it behind `store_is_set()`), so it's also
    the one place that can warn about the choice without warning more than
    once. On a Cloud Run instance (`K_SERVICE` set) an unnamed bucket means
    frames minted on this instance are invisible to any other, which is a
    silent correctness degradation for callers that resolve a reference
    against a different instance than the one that minted it.
    """
    if bucket_name is None:
        if os.environ.get("K_SERVICE"):
            logger.warning(
                "FRAME_STORE_BUCKET is unset on a Cloud Run instance "
                "(K_SERVICE=%s); falling back to an in-memory frame store "
                "that other instances can't see.",
                os.environ["K_SERVICE"],
            )
        return InMemoryMaterialisationStore()
    derive_content_version = None
    if verify_content_version_on_read:
        derive_content_version = get_content_version_deriver()
        if derive_content_version is None:
            raise ContentVersionDeriverUnregistered()
    return GcsMaterialisationStore(
        bucket_name=bucket_name,
        verify_content_version_on_read=verify_content_version_on_read,
        derive_content_version=derive_content_version,
    )


def install_store_if_unset(
    *, bucket_name: str | None, verify_content_version_on_read: bool = False
) -> MaterialisationStore | None:
    """`build_store(...)` installed into the shared holder, but only if nothing is
    installed there yet. Returns the store it installed, or `None` if an earlier
    call -- from this service's own lifespan or the other service's -- already won.

    This is the ownership rule for a process that mounts more than one service:
    each service's own lifespan calls this, and `combine_lifespans` runs every
    lifespan's `__aenter__` before the composed app starts serving, so whichever
    call runs first installs the store and every later call is a no-op,
    regardless of mount order. Two `GcsMaterialisationStore` instances over the
    same bucket are interchangeable, but two `InMemoryMaterialisationStore`
    instances are not: the second call would silently discard anything already
    written through the first, including a service's own fixture seeding.
    """
    if store_is_set():
        return None
    store = build_store(
        bucket_name=bucket_name, verify_content_version_on_read=verify_content_version_on_read
    )
    set_store(store)
    return store
