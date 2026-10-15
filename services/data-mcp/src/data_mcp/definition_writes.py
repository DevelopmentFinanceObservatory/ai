"""Writing spec definitions from Search results into the definition store.

`search` and `resolve_definition` register every definition their own
envelope carries (`stored_definition`, `put_definitions`). Every write is
verified first: an identity must hash to the spec_id it is stored under, so
a write can only ever add the definition its spec_id names.
`caller_definition` checks a definition this service did not read from
Search itself, as the apis service's registry takes it, and `windowed_identity`
computes an identity over other years and the spec_id it hashes to.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import rfc8785
from fastmcp.tools import ToolResult

from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    StoredDefinition,
    canonicalize,
    get_store,
    spec_id_for,
    verify_write,
)
from data_mcp.tool_errors import ToolRefusal, to_tool_result

# A definition-store write failure carries no `UpstreamError` to compute a
# wait from, so a fixed floor stands in for one.
_DEFINITION_STORE_RETRY_AFTER_S = 5.0
_DEFINITION_STORE_SERVICE_NAME = "the definition store"

# "spec_" and the 32 lowercase hex characters
# `data_mcp.definition_store.spec_id_for` mints.
SPEC_ID_PATTERN = r"^spec_[0-9a-f]{32}$"

# The largest identity the apis service's registry takes, in bytes of compact UTF-8
# JSON (`spec_registry.identity._MAX_IDENTITY_DOCUMENT_BYTES`): a larger one
# could never be minted, so it is not stored either.
MAX_IDENTITY_BYTES = 65_536


def identity_definition(definition: Mapping[str, Any]) -> StoredDefinition:
    """A `{schema_version, spec_id, identity}` sidecar as a stored definition.

    Raises:
        IdentityMismatch: If the identity does not hash to the spec_id.
    """
    spec_id = definition["spec_id"]
    return StoredDefinition(
        spec_id=spec_id,
        schema_version=definition["schema_version"],
        canonical_bytes=verify_write(definition.get("identity") or {}, spec_id),
    )


def stored_definition(node: Mapping[str, Any]) -> StoredDefinition | None:
    """The verified definition a Search API Spec block carries, or `None` when it carries none.

    A block with no `definition`, or a `definition` with no `spec_id`, carries
    none.

    Raises:
        IdentityMismatch: If the definition's identity does not hash to its spec_id.
    """
    definition = node.get("definition")
    if not definition or not definition.get("spec_id"):
        return None
    return identity_definition(definition)


async def put_definitions(definitions: Sequence[StoredDefinition]) -> ToolResult | None:
    """Write `definitions`, returning a refusal when the store is unreachable and `None` otherwise."""
    if not definitions:
        return None
    try:
        await get_store().put_many(definitions)
    except DefinitionStoreUnavailable:
        return to_tool_result(
            ToolRefusal.service_unavailable(
                service=_DEFINITION_STORE_SERVICE_NAME,
                retry_after_s=_DEFINITION_STORE_RETRY_AFTER_S,
            )
        )
    return None


class IdentityUnusable(ValueError):  # noqa: N818 -- named for what it says of the identity, as `IdentityMismatch` is
    """An identity this service will not store: past `MAX_IDENTITY_BYTES`, or not
    representable as RFC 8785 JSON (an integer past 2**53, say), so no spec_id can
    be computed for it."""


def caller_definition(definition: Mapping[str, Any]) -> StoredDefinition:
    """A `{schema_version, spec_id, identity}` sidecar a caller handed in, checked as the apis service's registry takes it.

    For a definition this service did not read from the Search API itself,
    such as a Studio seed's. The identity's compact JSON is at most
    `MAX_IDENTITY_BYTES`, measured as the apis service's `spec_registry.identity`
    measures it, and it hashes to its spec_id (`identity_definition`).

    Raises:
        IdentityUnusable: If the identity is past the cap or has no RFC 8785 form.
        IdentityMismatch: If the identity does not hash to the spec_id.
        KeyError: If the sidecar carries no `spec_id`.
    """
    identity = definition.get("identity") or {}
    size = len(json.dumps(identity, separators=(",", ":"), ensure_ascii=False).encode())
    if size > MAX_IDENTITY_BYTES:
        raise IdentityUnusable(
            f"the identity is {size:,} bytes, past the {MAX_IDENTITY_BYTES:,}-byte cap"
        )
    try:
        return identity_definition(definition)
    except rfc8785.CanonicalizationError as exc:
        raise IdentityUnusable(f"the identity has no canonical JSON form: {exc}") from exc


def windowed_identity(
    identity: Mapping[str, Any], window: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """`identity` over `window` in place of its own years, and the spec_id that hashes from it.

    `window` replaces `requested_window` whole, in the identity's own form
    (`{"kind": "range", "start_year": 2024, "end_year": 2024}`). That is the
    whole of the scope surgery apis runs for `get_data`'s `scope.window`
    (`datasets_api.surgery.apply_scope_surgery`), and spec_ids hash the same
    RFC 8785 bytes on both sides, so the spec_id is the one that mint answers
    under; `tests/test_definition_writes.py` pins that against recorded dev
    answers. A served identity is already in the identity core's normal form,
    and a range window of two whole years is too, so hashing it as it stands
    is hashing what apis hashes.
    """
    windowed = {**identity, "requested_window": dict(window)}
    return spec_id_for(canonicalize(windowed)), windowed
