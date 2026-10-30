"""Registering a definition this service holds with dfo-apis.

`search` writes each definition it returns to the definition store, and dfo-apis learns
of it on the first mint that carries it. A `spec_id` passed alone, to a verb,
`describe_frame`, `write_component` or `export_data`, reaches dfo-apis with no definition
beside it, and an id dfo-apis has not registered refuses `spec_not_found`. The combined
server therefore registers each such id first (`POST /api/datasets/register`), so a bare
id fresh from `search` reads as `get_data` would mint it. The route is idempotent and
materialises nothing, and each id this process registered is remembered, so it is sent
once. dfo-apis may already hold an id, so a failure a retry may mend lets the call run.
A 4xx refuses it in the calling tool's name.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastmcp.tools import ToolResult

from data_mcp.client_holder import get_client
from data_mcp.deadline import CallDeadline
from data_mcp.definition_store import DefinitionStoreUnavailable, IdentityMismatch, get_store
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import ToolRefusal, to_tool_result

logger = logging.getLogger(__name__)

_REGISTER_ROUTE = "/api/datasets/register"
_SCHEMA_VERSION = 1

# What one id may take of a call: the store read, then the registry write.
_STORE_READ_LIMIT_S = 2.0
_REGISTER_DEADLINE_S = 5.0

# The ids this process registered, oldest first, since the registry keeps what it takes.
_REGISTERED_LIMIT = 10_000
_registered: dict[str, None] = {}


async def register_held_definition(spec_id: str, *, tool: str) -> ToolResult | None:
    """Register `spec_id` with dfo-apis when the definition store holds it, else do
    nothing, so dfo-apis answers for an id this service never saw. A store that cannot
    answer and a registration that failed for a reason a retry may mend are logged and
    passed the same way. Returns the refusal of `tool`, which was passed the id, for a
    registration dfo-apis refused with a 4xx."""
    if spec_id in _registered:
        return None
    try:
        async with asyncio.timeout(_STORE_READ_LIMIT_S):
            stored = await get_store().get(spec_id, schema_version=_SCHEMA_VERSION)
    except (TimeoutError, DefinitionStoreUnavailable, IdentityMismatch):
        logger.warning("definition of %s unread; not registered", spec_id, exc_info=True)
        return None
    if stored is None:
        return None
    definition = {
        "schema_version": _SCHEMA_VERSION,
        "spec_id": spec_id,
        "identity": json.loads(stored.canonical_bytes),
    }
    try:
        await get_client().request(
            "POST",
            _REGISTER_ROUTE,
            route=_REGISTER_ROUTE,
            json={"schema_version": _SCHEMA_VERSION, "definition": definition},
            deadline=CallDeadline.after(_REGISTER_DEADLINE_S),
        )
    except UpstreamError as exc:
        if exc.retryable or exc.status is None or exc.status >= 500:
            logger.warning("definition of %s unregistered: %s", spec_id, exc)
            return None
        return to_tool_result(
            ToolRefusal.spec_not_registered(
                tool=tool, spec_id=spec_id, upstream_code=exc.code, detail=exc.detail
            )
        )
    _registered[spec_id] = None
    while len(_registered) > _REGISTERED_LIMIT:
        del _registered[next(iter(_registered))]
    return None
