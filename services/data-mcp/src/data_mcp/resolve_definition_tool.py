"""The `resolve_definition` tool: one definition resolved again through the Search API's `dataset_resubmit` door.

The harness calls it for each Search result a Studio turn starts from. A
browser hands the harness a Spec block; the harness keeps the block's
spec_id and identity and asks Search what that identity resolves to now, so
every label, sentence and population fact the turn reads is Search's own
answer. The tool posts one `identity` entry whose base is the identity
itself, a no-op edit, registers every definition the answer carries as
`search` does, and returns the envelope in `_meta` under
`SEARCH_ENVELOPE_META_KEY`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from data_mcp.definition_store import IdentityMismatch
from data_mcp.definition_writes import SPEC_ID_PATTERN, IdentityUnusable, caller_definition
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.search_tool import resolve_and_register
from data_mcp.tool_errors import ToolRefusal, to_tool_result

# The StatVars one `dataset_resubmit` identity entry takes
# (dfo-apis `search.models.request._RESUBMIT_MAX_RESOLVED_STAT_VARS`).
MAX_RESUBMIT_STAT_VARS = 2_000


def resubmit_request(
    *, spec_id: str, identity: dict[str, Any], stat_vars: Sequence[str] | None
) -> dict[str, Any]:
    """The resolve request that re-resolves `identity` unchanged.

    The identity is posted as its own base, which Search verifies by hashing
    it to `spec_id`. Search names a part by the shape it verifies, whatever
    `variable_text` says, so the spec_id stands in as that echo. `stat_vars`
    is the seed Search needs for a shape no measure family covers: Search
    re-reads each one in the graph and ignores them for a shape a family
    covers.
    """
    entry: dict[str, Any] = {
        "entry_kind": "identity",
        "variable_text": spec_id,
        "identity": identity,
        "base_spec_id": spec_id,
        "base_identity": identity,
    }
    if stat_vars:
        entry["resolved_stat_var_dcids"] = list(stat_vars)
    return {
        "schema_version": 1,
        "input": {"kind": "dataset_resubmit", "entries": [entry]},
        "response_options": {"include_sentence": True},
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
    stat_vars: Annotated[
        list[str] | None,
        Field(
            max_length=MAX_RESUBMIT_STAT_VARS,
            description=(
                "The StatVars Search listed for the spec (`resolved.stat_vars.items[].ref`), "
                "which Search checks in the graph before it uses them."
            ),
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
        resubmit_request(spec_id=spec_id, identity=identity, stat_vars=stat_vars)
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
