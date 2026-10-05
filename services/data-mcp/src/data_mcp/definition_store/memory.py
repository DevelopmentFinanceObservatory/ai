"""In-memory `DefinitionStore`: the default when no bucket is configured."""

from __future__ import annotations

from collections.abc import Sequence

from data_mcp.definition_store.protocol import StoredDefinition


class InMemoryDefinitionStore:
    """Process-local `DefinitionStore` backed by a plain dict.

    The first write of a `(spec_id, schema_version)` wins and a later one is
    dropped, as on GCS (`gcs.GCSDefinitionStore`); both carry the same
    identity, since the spec_id is its hash.

    Definitions never expire and are lost on process restart -- acceptable
    because a harness session is itself in-memory and dies with the instance
    anyway. `aclose` is a no-op: there is no external resource to release.
    """

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, int], StoredDefinition] = {}

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        for definition in definitions:
            self._definitions.setdefault(
                (definition.spec_id, definition.schema_version), definition
            )

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        return self._definitions.get((spec_id, schema_version))

    async def aclose(self) -> None:
        pass
