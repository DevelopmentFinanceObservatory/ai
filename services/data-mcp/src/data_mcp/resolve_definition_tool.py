"""The `resolve_definition` tool: one definition resolved again through the Search API's `dataset_resubmit` door.

The harness calls it for each Search result a Studio turn starts from. A
browser hands the harness a Spec block; the harness keeps the block's
spec_id and identity and asks Search what that identity resolves to now, so
every label, sentence and population fact the turn reads is Search's own
answer. The tool posts one `identity` entry, from which Search derives the
population, registers every definition the answer carries as `search` does,
and returns the envelope in `_meta` under `SEARCH_ENVELOPE_META_KEY`.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from data_mcp.definition_store import IdentityMismatch
from data_mcp.definition_writes import SPEC_ID_PATTERN, IdentityUnusable, caller_definition
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.search_tool import resolve_and_register
from data_mcp.tool_errors import ToolRefusal, to_tool_result

# The wall-clock budget for one resubmit, and its read timeout. apis gives a
# resubmit whose member walk reads a family's whole subject set 90 s, against
# 45 s for a search, so the read waits 95 s and the budget leaves room for the
# connect. The harness's `resolve_definition` call timeout
# (`compiler.capabilities`) sits above the budget.
RESUBMIT_READ_TIMEOUT_S = 95.0
RESUBMIT_CALL_BUDGET_S = 100.0


def resubmit_request(
    *, spec_id: str, identity: dict[str, Any], entity_groups: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The resolve request that re-resolves `identity` unchanged.

    Search names a part by the shape it verifies, whatever `variable_text`
    says, so the spec_id stands in as that echo. `entity_groups`, the sets a
    role's places were asked for as, rides beside the identity when given:
    Search keeps one the graph confirms and names it again.
    """
    entry: dict[str, Any] = {
        "entry_kind": "identity",
        "variable_text": spec_id,
        "identity": identity,
    }
    if entity_groups:
        entry["entity_groups"] = entity_groups
    return {
        "schema_version": 1,
        "input": {"kind": "dataset_resubmit", "entries": [entry]},
    }


async def resolve_definition(
    spec_id: Annotated[
        str,
        Field(
            pattern=SPEC_ID_PATTERN,
            description="The spec_id the identity hashes to, as Search serves it.",
        ),
    ],
    identity: Annotated[
        dict[str, Any], Field(description="The definition's `identity`, as Search serves it.")
    ],
    entity_groups: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "The Spec's `resolved.entity_groups`, as Search serves it, when a role's "
                "places were asked for as a set."
            )
        ),
    ] = None,
) -> ToolResult:
    """Resolve a Spec block's identity again through Search, so the caller reads Search's own answer.

    The identity must hash to `spec_id` and take at most 65,536 bytes as
    compact JSON; one that does not refuses `request_invalid` and sends
    nothing. Search answers with a one-part envelope, whose definitions are
    stored as `search` stores them. A `definite` part whose spec carries the
    same spec_id is the identity resolving as it did; any other answer says
    it no longer does.

    Returns:
        The envelope's `status` and the spec_id asked about as structured
        content, and the envelope in `_meta`.
    """
    try:
        caller_definition({"schema_version": 1, "spec_id": spec_id, "identity": identity})
    except (IdentityMismatch, IdentityUnusable) as exc:
        return to_tool_result(ToolRefusal.request_invalid(pointer="/identity", detail=str(exc)))
    envelope = await resolve_and_register(
        resubmit_request(spec_id=spec_id, identity=identity, entity_groups=entity_groups),
        budget_s=RESUBMIT_CALL_BUDGET_S,
        read_timeout_s=RESUBMIT_READ_TIMEOUT_S,
    )
    if isinstance(envelope, ToolResult):
        return envelope
    status = envelope.get("status")
    return ToolResult(
        content=[
            TextContent(type="text", text=f"Search resolved the identity of {spec_id}: {status}.")
        ],
        structured_content={"spec_id": spec_id, "status": status},
        meta={SEARCH_ENVELOPE_META_KEY: envelope},
    )
