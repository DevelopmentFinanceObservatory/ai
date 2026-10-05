"""The `DefinitionStore` protocol and its value type.

Persists the canonical definition behind a spec_id for the window between
`search` (which mints ids without registering them) and the first `get_data`
(after which the platform registry holds the definition permanently).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, kw_only=True)
class StoredDefinition:
    """One canonical definition, keyed by `spec_id` and `schema_version`.

    `canonical_bytes` is the RFC 8785 canonical JSON of the spec1 identity
    core document (see `data_mcp.definition_store.identity`). The wire
    sidecar `{schema_version, spec_id, identity}` a caller hands back is
    reconstructed from these three fields. The store holds the identity and
    nothing else: `spec_id` is its hash, so every write of one key carries
    the same bytes and the first write is the only one that lands.
    """

    spec_id: str
    schema_version: int
    canonical_bytes: bytes


class DefinitionStore(Protocol):
    """Persists and retrieves `StoredDefinition`s by `(spec_id, schema_version)`."""

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        """Write every definition in `definitions`.

        Returns `None` on success. Raises `DefinitionStoreUnavailable` if any
        write failed, after every write in the batch has been attempted.
        """
        ...

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        """Return the stored definition for `(spec_id, schema_version)`.

        Returns `None` for a genuine miss -- the backend is up, the id just
        isn't registered. Raises `DefinitionStoreUnavailable` when the
        backend itself is unreachable. The two are never conflated: they map
        to different tool-facing refusals.
        """
        ...

    async def aclose(self) -> None:
        """Release any resources this store holds (connections, clients)."""
        ...
