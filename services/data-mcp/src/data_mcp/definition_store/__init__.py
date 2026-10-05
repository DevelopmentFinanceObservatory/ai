"""The definition store: persists the canonical definition behind a spec_id
for the window between `search` (which mints ids without registering them)
and the first `get_data` (after which the platform registry holds the
definition permanently).
"""

from __future__ import annotations

from data_mcp.definition_store.caching import CachingDefinitionStore
from data_mcp.definition_store.errors import DefinitionStoreUnavailable
from data_mcp.definition_store.gcs import GCSDefinitionStore
from data_mcp.definition_store.holder import aclose_store, get_store, set_store
from data_mcp.definition_store.identity import (
    IdentityMismatch,
    canonicalize,
    spec_id_for,
    verify_read,
    verify_write,
)
from data_mcp.definition_store.memory import InMemoryDefinitionStore
from data_mcp.definition_store.protocol import DefinitionStore, StoredDefinition

__all__ = [
    "CachingDefinitionStore",
    "DefinitionStore",
    "DefinitionStoreUnavailable",
    "GCSDefinitionStore",
    "IdentityMismatch",
    "InMemoryDefinitionStore",
    "StoredDefinition",
    "aclose_store",
    "canonicalize",
    "get_store",
    "set_store",
    "spec_id_for",
    "verify_read",
    "verify_write",
]
