"""The published tool schemas, assembled from the platform's release document.

`release["intent_schemas_tolerant"]` already carries six tolerant per-type
schemas -- the platform is the source of truth for what a partial patch may
look like, not this service. What is built here is assembly, not derivation:
each member's own `$defs`, `$schema`, and `$id` are stripped, the six member
schemas become the branches of one `anyOf`, and only the `$defs` entries a
branch actually references by `$ref` are kept alongside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_INTENT_DESCRIPTION = (
    "Omit component_id to create a new component: intent must then be a complete "
    "member. line_chart, stat_card, bar_chart, and table require type, frame_ref, "
    "and select, with optional window or title. note_narration requires type and "
    "text. source_card requires type and frame_ref, not select. "
    "Pass component_id to edit that component: intent is then a partial patch, "
    "holding only the fields being changed. A patch's objects merge into the "
    "stored intent, its arrays replace theirs wholly, and setting a field to null "
    "clears it. fact_summary.change in the reply populates only for a stat_card "
    "intent that authors a change object with a basis. line_chart, bar_chart, "
    "table, note_narration, and source_card carry no change property, whatever "
    "the intent sends."
)

_COMPONENT_ID_DESCRIPTION = (
    "Omit to create a new component. Pass the id a previous write_component "
    "reply returned, with its write_token, to edit that component. The id is "
    "the only way back to a component: keep it, and reuse it to revise or "
    "describe the component."
)

_WRITE_TOKEN_DESCRIPTION = (
    "The write_token the component's create returned. Required with "
    "component_id. Without it, create a new component."
)

_INCLUDE_PROJECTION_DESCRIPTION = (
    "Whether the reply leads with a Markdown table of the component's data. "
    "auto sends it on a create and on an edit, except an edit whose patch "
    "changes only title, scale, or range, which gets one line naming what "
    "changed. A client drawing the component as an interactive view gets the "
    "lead without the table, as does one the server cannot identify. full "
    "always sends the table. none leaves it out, and withholds the chart "
    "image from a client that cannot draw the view."
)

_FRAME_REF_DESCRIPTION = (
    "A frame_ref is the spec_id and content_version pair a data tool's reply "
    "returned. Do not invent one. Any caller holding a frame_ref can use it, "
    "whichever caller minted it. A frame_ref whose spec_id starts with ops_ "
    "and whose content_version starts with ocv_ names an Analysis Toolkit "
    "result, and is presentable like any other frame_ref."
)

_DESCRIBE_COMPONENT_ID_DESCRIPTION = (
    "The id a previous write_component reply returned. The id is the only way back to a component."
)

_DESCRIBE_REVISION_DESCRIPTION = (
    "The revision count a previous write_component reply returned. Omit for the current revision."
)

_DESCRIBE_INCLUDE_PROJECTION_DESCRIPTION = (
    "Whether the reply leads with a Markdown table of the component's data. "
    "auto sends it, except to a client that draws the component as an "
    "interactive view, or one the server cannot identify, which gets that "
    "lead without the table. full always sends the table. none leaves it "
    "out, and withholds the chart image from a client that cannot draw the "
    "view."
)

# Kept in step with dfo-apis' presentation path-parameter checks and the
# client-side refusals in `client.py`.
_COMPONENT_ID_PATTERN = r"^cmp_[A-Za-z0-9_-]{1,64}$"
_WRITE_TOKEN_PATTERN = r"^cmw_[A-Za-z0-9_-]{1,64}$"
_SPEC_ID_PATTERN = r"^(?:spec|ops)_[a-z0-9]{1,64}$"
_CONTENT_VERSION_PATTERN = r"^(?:cv|ocv)_[a-z0-9]{1,64}$"

_MEMBER_HOUSEKEEPING = ("$defs", "$schema", "$id")


def _shared_defs(members: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """The one `$defs` map every member schema carries, asserted identical.

    Args:
        members: `release["intent_schemas_tolerant"]`, six member schemas.

    Returns:
        The shared `$defs` mapping.

    Raises:
        ValueError: A member's `$defs` differs from the first one seen. The
            six copies are expected to be identical; a mismatch means the
            release document changed shape in a way this assembly cannot
            safely reconcile, so it raises rather than silently picking one.
    """
    names = iter(members)
    first_name = next(names)
    first_defs = members[first_name]["$defs"]
    for name in names:
        defs = members[name]["$defs"]
        if defs != first_defs:
            msg = f"$defs for {name!r} differs from {first_name!r}; the release document is inconsistent"
            raise ValueError(msg)
    return first_defs


def _strip_member(schema: Mapping[str, Any]) -> dict[str, Any]:
    """A member schema with its own release housekeeping removed."""
    return {k: v for k, v in schema.items() if k not in _MEMBER_HOUSEKEEPING}


def _collect_refs(node: Any, defs: Mapping[str, Any], collected: set[str]) -> None:
    """Walk `node`, adding every `#/$defs/<name>` target reachable from it to `collected`.

    Recurses into a newly-found def's own body too, so a def that itself
    refs another def still pulls its target in.
    """
    if isinstance(node, Mapping):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            if name not in collected:
                collected.add(name)
                _collect_refs(defs[name], defs, collected)
        for value in node.values():
            _collect_refs(value, defs, collected)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, defs, collected)


def _reachable_defs(branches: list[dict[str, Any]], defs: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of `defs` transitively reachable by `$ref` from `branches`."""
    reachable: set[str] = set()
    for branch in branches:
        _collect_refs(branch, defs, reachable)
    return {name: defs[name] for name in sorted(reachable)}


def write_component_schema(*, release: Mapping[str, Any]) -> dict[str, Any]:
    """The advertised `inputSchema` for `write_component`.

    Args:
        release: The platform's release document, carrying
            `intent_schemas_tolerant`.
    """
    members = release["intent_schemas_tolerant"]
    defs = _shared_defs(members)
    branches = [_strip_member(schema) for schema in members.values()]
    reachable_defs = _reachable_defs(branches, defs)
    if "frame_ref" in reachable_defs:
        reachable_defs["frame_ref"] = {
            **reachable_defs["frame_ref"],
            "description": _FRAME_REF_DESCRIPTION,
        }
    return {
        "type": "object",
        "properties": {
            "intent": {
                "description": _INTENT_DESCRIPTION,
                "anyOf": branches,
            },
            "component_id": {
                "type": "string",
                "minLength": 1,
                "pattern": _COMPONENT_ID_PATTERN,
                "description": _COMPONENT_ID_DESCRIPTION,
            },
            "write_token": {
                "type": "string",
                "pattern": _WRITE_TOKEN_PATTERN,
                "description": _WRITE_TOKEN_DESCRIPTION,
            },
            "include_projection": {
                "type": "string",
                "enum": ["auto", "full", "none"],
                "default": "auto",
                "description": _INCLUDE_PROJECTION_DESCRIPTION,
            },
        },
        "required": ["intent"],
        "additionalProperties": False,
        "$defs": reachable_defs,
    }


def describe_frame_schema(*, release: Mapping[str, Any]) -> dict[str, Any]:
    """The advertised `inputSchema` for `describe_frame`.

    `frame_ref` is lifted verbatim from the shared `$defs`: a frame reference
    is always complete, there is no tolerant form of one.

    Args:
        release: The platform's release document, carrying
            `intent_schemas_tolerant`.
    """
    defs = _shared_defs(release["intent_schemas_tolerant"])
    frame_ref = dict(defs["frame_ref"])
    properties = frame_ref["properties"]
    frame_ref["properties"] = {
        **properties,
        "spec_id": {**properties["spec_id"], "pattern": _SPEC_ID_PATTERN},
        "content_version": {
            **properties["content_version"],
            "pattern": _CONTENT_VERSION_PATTERN,
        },
    }
    frame_ref["description"] = _FRAME_REF_DESCRIPTION
    return {
        "type": "object",
        "properties": {
            "frame_ref": frame_ref,
            "dimension": {
                "type": "string",
                "description": "Narrow the view to this one dimension's values.",
            },
            "query": {
                "type": "string",
                "description": "Keep only dimension values containing this text.",
            },
        },
        "required": ["frame_ref"],
        "additionalProperties": False,
    }


def describe_component_schema() -> dict[str, Any]:
    """The advertised `inputSchema` for `describe_component`.

    The schema has a fixed shape: `component_id`, `revision`, and
    `include_projection` are this service's own scalars. FastMCP parses
    incoming arguments against `describe_component`'s permissive function
    signature, so `minLength`/`pattern`/`minimum` here serve the client as
    documentation.
    The component-id refusal is enforced in `client.py`.
    """
    return {
        "type": "object",
        "properties": {
            "component_id": {
                "type": "string",
                "minLength": 1,
                "pattern": _COMPONENT_ID_PATTERN,
                "description": _DESCRIBE_COMPONENT_ID_DESCRIPTION,
            },
            "revision": {
                "type": "integer",
                "minimum": 1,
                "description": _DESCRIBE_REVISION_DESCRIPTION,
            },
            "include_projection": {
                "type": "string",
                "enum": ["auto", "full", "none"],
                "default": "auto",
                "description": _DESCRIBE_INCLUDE_PROJECTION_DESCRIPTION,
            },
        },
        "required": ["component_id"],
        "additionalProperties": False,
    }
