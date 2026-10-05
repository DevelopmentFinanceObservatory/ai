"""Renders `describe`'s node outcomes into the model-facing digest.

On a call with no unheld dcid, this digest is the entire model-facing
surface. The harness reads `structuredContent` in code, so a model only
ever sees this digest. `translator.discovery.fetch_definitions` reads a
`describe` answer's `nodes` for one `definition` string per source, and
`harness.mcp_client.McpExecutor._indicators_for` reads them for the More
Info panel's names. On a call with at least one unheld dcid, the tool
renders this same digest as a preamble ahead of the refusal message, so what
resolved still reaches the model even though the call as a whole is an
error.

`render_describe_digest` is the public seam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from data_mcp.node_lookup import (
    EntityNode,
    GenericNode,
    NodeOutcome,
    NotHeldNode,
    SourceNode,
    StatVarNode,
)

# At most this many containment entries are named per entity; the rest are
# only counted. Nothing upstream bounds `contained_in`'s length.
_CONTAINED_IN_CAP = 8

# At most this many properties are rendered per generic node; the rest are
# only counted. `GenericNode.properties` is unbounded in width, and one node
# could otherwise dominate an eight-id digest.
_GENERIC_PROPERTY_CAP = 12

# At most this many values are rendered per property; the rest are only
# counted. `GenericNode.properties` is unbounded in per-property depth too.
_GENERIC_VALUE_CAP = 5

# At most this many sources are named per StatVar; the rest are only
# counted. `StatVarNode.sources` is typed `Any` and nothing upstream bounds
# its length.
_SOURCES_CAP = 8

# At most this many facet axes are named per StatVar, and at most this many
# values per axis; the rest are only counted. `StatVarNode.facets` is typed
# `Any` and unbounded in both width and depth.
_FACET_AXIS_CAP = 8
_FACET_VALUE_CAP = 5

# What a null field renders as, so its absence is visible in the digest text
# rather than looking indistinguishable from a field the digest chose not to
# print.
_NOT_RECORDED = "not recorded"

# What `sources` or `facets` renders as when it arrives in a shape this
# module does not recognise -- named rather than dumped as a repr or dropped
# silently, since `Any` gives no guarantee about what upstream sends.
_UNRECOGNISED_SHAPE = "arrived in a shape this digest does not recognise"


def render_describe_digest(*, dcids: Sequence[str], outcomes: Mapping[str, NodeOutcome]) -> str:
    """Render `outcomes` into the model-facing digest, one block per dcid in `dcids`' order.

    Args:
        dcids: The dcids this call was asked to describe, in request order --
            duplicates included, so a dcid asked for twice renders its block
            twice, matching what the caller asked for.
        outcomes: dcid -> `NodeOutcome`, as `node_lookup.describe_nodes` returns it.

    Returns:
        One block per dcid, shaped by its node type, followed by a single
        trailing note naming every dcid this deployment's graph does not
        hold, if any.
    """
    blocks: list[str] = []
    not_held: list[str] = []
    seen_not_held: set[str] = set()
    for dcid in dcids:
        outcome = outcomes[dcid]
        if isinstance(outcome, NotHeldNode):
            if dcid not in seen_not_held:
                not_held.append(dcid)
                seen_not_held.add(dcid)
            continue
        blocks.append(_render_node(outcome))
    if not_held:
        blocks.append(_not_held_block(not_held))
    return "\n\n".join(blocks)


def _render_node(
    node: StatVarNode | SourceNode | EntityNode | GenericNode,
) -> str:
    if isinstance(node, StatVarNode):
        return _render_stat_var(node)
    if isinstance(node, SourceNode):
        return _render_source(node)
    if isinstance(node, EntityNode):
        return _render_entity(node)
    return _render_generic(node)


def _heading(node: StatVarNode | SourceNode | EntityNode | GenericNode) -> str:
    """`label (dcid)`, or the dcid alone when the node door served no label."""
    return f"{node.label} ({node.dcid})" if node.label else node.dcid


def _render_stat_var(node: StatVarNode) -> str:
    lines = [_heading(node)]
    if node.definition is not None:
        lines.append(node.definition)
    lines.append(f"roles: {', '.join(node.roles)}")
    if node.population_type:
        lines.append(f"measures a property of {node.population_type}")
    if node.processing_level is not None:
        lines.append(f"processing level: {node.processing_level}")
    if node.sources is not None:
        lines.append(_render_stat_var_sources(node.sources))
    if node.facets is not None:
        lines.extend(_render_stat_var_facets(node.facets))
    gap_line = _stat_var_gap_line(node)
    if gap_line is not None:
        lines.append(gap_line)
    return "\n".join(lines)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, str | int | float | bool)


def _is_scalar_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and all(_is_scalar(item) for item in value)
    )


def _render_stat_var_sources(sources: Any) -> str:
    """Render `sources` as a capped, comma-joined list, or name its shape as unrecognised."""
    if not _is_scalar_sequence(sources):
        return f"sources: {_UNRECOGNISED_SHAPE}"
    values = [str(item) for item in sources]
    return f"sources: {_joined_with_cap(values, _SOURCES_CAP)}"


def _render_stat_var_facets(facets: Any) -> list[str]:
    """Render `facets` as one capped line per axis, or name its shape as unrecognised."""
    if not isinstance(facets, Mapping):
        return [f"facet values: {_UNRECOGNISED_SHAPE}"]
    if not all(
        isinstance(axis, str) and _is_scalar_sequence(values) for axis, values in facets.items()
    ):
        return [f"facet values: {_UNRECOGNISED_SHAPE}"]
    items = list(facets.items())
    shown = items[:_FACET_AXIS_CAP]
    lines = []
    for axis, values in shown:
        assert isinstance(values, Sequence)
        lines.append(
            f"facet axis {axis}: {_joined_with_cap([str(v) for v in values], _FACET_VALUE_CAP)}"
        )
    remainder = len(items) - len(shown)
    if remainder > 0:
        lines.append(f"... and {remainder} more facet axes not shown.")
    return lines


def _stat_var_gap_line(node: StatVarNode) -> str | None:
    """Name whichever of facets/processing_level/sources this node carries no value for.

    Returns `None` once every field that arrived has been rendered above,
    since there is nothing left to name as missing. Without this line, a
    model reading an absent `sources` field could mistake "not served by
    the node door" for "this variable has no sources".
    """
    missing = []
    if node.facets is None:
        missing.append("the facet values per axis")
    if node.processing_level is None:
        missing.append("its processing level")
    if node.sources is None:
        missing.append("its sources")
    if not missing:
        return None
    return (
        f"Not available for this StatVar: {', '.join(missing)} -- "
        "the node door does not serve them."
    )


def _render_source(node: SourceNode) -> str:
    license_text = node.license or _NOT_RECORDED
    license_type_text = node.license_type or _NOT_RECORDED
    release = node.freshness.release_frequency or _NOT_RECORDED
    refresh = node.freshness.refresh_schedule or _NOT_RECORDED
    return "\n".join(
        [
            _heading(node),
            f"licence: {license_text} ({license_type_text})",
            f"freshness: release {release} · refresh {refresh}",
            f"description: {node.description or _NOT_RECORDED}",
            f"url: {node.url or _NOT_RECORDED}",
        ]
    )


def _render_entity(node: EntityNode) -> str:
    names = [member.label or member.dcid for member in node.contained_in]
    return "\n".join(
        [
            _heading(node),
            f"type: {node.entity_type or _NOT_RECORDED}",
            f"contained in: {_joined_with_cap(names, _CONTAINED_IN_CAP)}",
        ]
    )


def _render_generic(node: GenericNode) -> str:
    lines = [_heading(node), f"types: {', '.join(node.types)}"]
    if node.is_classification_value:
        lines.append(
            "This is a classification value, not necessarily the entity you meant. "
            "A place resolves through search."
        )
    items = list(node.properties.items())
    shown = items[:_GENERIC_PROPERTY_CAP]
    for key, values in shown:
        lines.append(f"{key}: {_joined_with_cap(values, _GENERIC_VALUE_CAP)}")
    remainder = len(items) - len(shown)
    if remainder > 0:
        lines.append(f"... and {remainder} more properties not shown.")
    return "\n".join(lines)


def _not_held_block(dcids: Sequence[str]) -> str:
    return f"Not held by this deployment's graph: {', '.join(dcids)}."


def _joined_with_cap(items: Sequence[str], cap: int) -> str:
    """Comma-join `items`, naming at most `cap` and summarising the rest."""
    shown = items[:cap]
    remainder = len(items) - len(shown)
    text = ", ".join(shown)
    if remainder <= 0:
        return text
    return f"{text}, and {remainder} more" if text else f"{remainder} more"
