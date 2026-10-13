"""Renders `describe`'s node outcomes into the model-facing digest.

On a call with no unheld dcid, this digest is the entire model-facing
surface. The harness reads `structuredContent` in code, so a model only
ever sees this digest. `translator.discovery.fetch_definitions` reads a
`describe` answer's `nodes` for one line per variable
(`data_mcp.node_lookup.variable_text`). On a call with at least one unheld
dcid, the tool renders this same digest as a preamble ahead of the refusal
message, so what resolved still reaches the model even though the call as a
whole is an error.

Every node renders its heading (`label (ref)`), its kind, its description
when it has one, the lines of its kind's block, and the gaps the node door
names. `render_describe_digest` is the public seam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from data_mcp.graph_wire import (
    AggregateNode,
    CodeNode,
    CodeSystemNode,
    ContinentalUnionNode,
    ContinentNode,
    CountryNode,
    DatasetNode,
    GroupNode,
    MembershipBlock,
    NodeRef,
    NotHeldEntry,
    OrganisationNode,
    OtherNode,
    PlaceBlock,
    PlaceOtherNode,
    ProvenanceNode,
    SourceBlock,
    SourceNode,
    StatVarNode,
    SubregionNode,
    UnitNode,
)
from data_mcp.node_lookup import HeldNode, NodeOutcome, definition_line

# At most this many references are named per list (containment, sources,
# gaps); the rest are only counted. The node door bounds none of them.
_LIST_CAP = 8

# What a null field renders as, so its absence is visible in the digest text
# rather than looking indistinguishable from a field the digest chose not to
# print.
_NOT_RECORDED = "not recorded"


def render_describe_digest(*, dcids: Sequence[str], outcomes: Mapping[str, NodeOutcome]) -> str:
    """Render `outcomes` into the model-facing digest, one block per dcid in `dcids`' order.

    Args:
        dcids: The dcids this call was asked to describe, in request order --
            duplicates included, so a dcid asked for twice renders its block
            twice, matching what the caller asked for.
        outcomes: dcid -> `NodeOutcome`, as `node_lookup.describe_nodes` returns it.

    Returns:
        One block per dcid, shaped by its node kind, followed by a single
        trailing note naming every dcid this deployment's graph does not
        hold, if any.
    """
    blocks: list[str] = []
    not_held: list[str] = []
    for dcid in dcids:
        outcome = outcomes[dcid]
        if isinstance(outcome, NotHeldEntry):
            if dcid not in not_held:
                not_held.append(dcid)
            continue
        blocks.append(_render_node(outcome))
    if not_held:
        blocks.append(f"Not held by this deployment's graph: {', '.join(not_held)}.")
    return "\n\n".join(blocks)


def _render_node(node: HeldNode) -> str:
    heading = f"{node.label} ({node.ref})" if node.label else node.ref
    lines = [heading, f"kind: {node.kind}"]
    if node.description:
        lines.append(node.description)
    lines.extend(_block_lines(node))
    if node.gaps:
        lines.append(f"not known: {_joined_with_cap(node.gaps)}")
    return "\n".join(lines)


def _block_lines(node: HeldNode) -> list[str]:
    match node:
        case StatVarNode():
            lines = _stat_var_lines(node)
        case (
            SourceNode(source=block) | DatasetNode(dataset=block) | ProvenanceNode(provenance=block)
        ):
            lines = _source_lines(block)
        case (
            CountryNode(country=place)
            | ContinentNode(continent=place)
            | SubregionNode(subregion=place)
            | ContinentalUnionNode(continental_union=place)
            | PlaceOtherNode(place_other=place)
        ):
            lines = _place_lines(place)
        case GroupNode():
            lines = _membership_lines(node.group)
        case OrganisationNode(organisation=party) | AggregateNode(aggregate=party):
            roles = ", ".join(party.roles or ()) or _NOT_RECORDED
            lines = [f"roles it can fill: {roles}"]
        case CodeSystemNode(code_system=system):
            lines = [
                f"codes of the {system.plain_name} property ({system.property}): {system.code_count}"
            ]
        case CodeNode(code=code):
            system = _name(code.code_system) if code.code_system else _NOT_RECORDED
            parent = _name(code.parent) if code.parent else "none"
            lines = [f"code {code.code} of {system} ({code.property})", f"parent: {parent}"]
        case UnitNode(unit=unit):
            lines = [f"unit class: {unit.unit_class}", f"currency: {unit.currency_noun or 'none'}"]
        case OtherNode():
            lines = _other_lines(node)
    return lines


def _stat_var_lines(node: StatVarNode) -> list[str]:
    definition = node.stat_var.definition
    lines = [definition_line(definition) or "definition: not recorded"]
    lines.append(f"roles: {', '.join(definition.roles)}")
    if definition.source_indicator_codes:
        lines.append(f"source indicator codes: {', '.join(definition.source_indicator_codes)}")
    availability = node.stat_var.availability
    if availability is None:
        lines.append("availability: not known")
        return lines
    summary = availability.summary
    first = summary.first_year or _NOT_RECORDED
    last = summary.last_year or _NOT_RECORDED
    lines.append(f"years held: {first} to {last}")
    names = [entry.source.label or entry.source.ref for entry in availability.sources[:_LIST_CAP]]
    if names:
        lines.append(f"sources ({summary.sources} in all): {', '.join(names)}")
    return lines


def _source_lines(block: SourceBlock) -> list[str]:
    freshness = block.freshness
    release = freshness.release_frequency_label or freshness.release_frequency or _NOT_RECORDED
    lines = [
        f"publisher: {_name(block.publisher) if block.publisher else _NOT_RECORDED}",
        f"licence: {block.license or _NOT_RECORDED} ({block.license_type or _NOT_RECORDED})",
        f"freshness: release {release} · refresh {freshness.refresh_type or _NOT_RECORDED}",
        f"url: {block.url or _NOT_RECORDED}",
    ]
    if block.data_commons is not None:
        lines.append(f"served by: {block.data_commons.label}")
    return lines


def _place_lines(place: PlaceBlock) -> list[str]:
    names = [_name(ref) for ref in place.contained_in]
    lines = [f"contained in: {_joined_with_cap(names) or 'nothing recorded'}"]
    if place.membership is not None:
        lines.extend(_membership_lines(place.membership))
    return lines


def _membership_lines(membership: MembershipBlock) -> list[str]:
    countries = next((p.count for p in membership.projections if p.projection == "countries"), None)
    return [
        f"membership: {membership.authority.name}",
        f"member countries: {_NOT_RECORDED if countries is None else countries}",
    ]


def _other_lines(node: OtherNode) -> list[str]:
    lines = [f"types: {', '.join(node.types) or _NOT_RECORDED}"]
    if node.types and all(t.endswith("Enum") for t in node.types):
        # A node whose every type is a classification enum, e.g.
        # `GeographicalIndicatorEnum`, labels a category a variable can bind
        # to. A caller who read it out of another digest more likely meant a
        # place.
        lines.append(
            "This is a classification value, not necessarily the entity you meant. "
            "A place resolves through search."
        )
    return lines


def _name(ref: NodeRef) -> str:
    return ref.label or ref.ref


def _joined_with_cap(items: Sequence[str]) -> str:
    """Comma-join `items`, naming at most `_LIST_CAP` and summarising the rest."""
    shown = items[:_LIST_CAP]
    remainder = len(items) - len(shown)
    text = ", ".join(shown)
    if remainder <= 0:
        return text
    return f"{text}, and {remainder} more"
