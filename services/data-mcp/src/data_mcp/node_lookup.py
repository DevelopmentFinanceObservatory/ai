"""Node lookup seam: dcids to node envelopes, via dfo-apis' `POST /api/graph/nodes`.

`describe_nodes` reads every dcid it is given through the node batch route, up to
`_MAX_DCIDS_PER_REQUEST` per request, and validates each entry through the generated
`data_mcp.graph_wire` models. An entry is one envelope per kind (`ref`, `label`, `kind`,
`types`, `description`, `links`, `gaps`, `built_from` and the kind's block), or a
`NotHeldEntry` for a dcid this deployment's graph does not hold. A failed request fails the
whole call.

`variable_text` is the line that stands for a variable in planner text: its `description`,
else a line composed from its card's `definition`.

Like `resolution.py`, this module returns plain data and never imports
`data_mcp.tool_errors`; the calling tool owns the refusal grammar, and every exception is
left to propagate.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping, Sequence

import httpx2
from pydantic import ValidationError

from data_mcp.client import GRAPH_PREFIX
from data_mcp.client_holder import get_client
from data_mcp.graph_wire import (
    AggregateNode,
    CodeNode,
    CodeSystemNode,
    ContinentalUnionNode,
    ContinentNode,
    CountryNode,
    DatasetNode,
    GroupNode,
    NodeRef,
    NodesResponse,
    NotHeldEntry,
    OrganisationNode,
    OtherNode,
    PlaceOtherNode,
    ProvenanceNode,
    SourceNode,
    StatVarNode,
    SubregionNode,
    UnitNode,
    VariableDefinition,
)

_NODES_ROUTE = f"{GRAPH_PREFIX}/nodes"

# The door's own published limit on `POST /api/graph/nodes`' `dcids` list
# (`caps.max_node_batch_dcids`).
_MAX_DCIDS_PER_REQUEST = 50

HeldNode = (
    StatVarNode
    | CodeSystemNode
    | CodeNode
    | GroupNode
    | CountryNode
    | ContinentNode
    | SubregionNode
    | ContinentalUnionNode
    | PlaceOtherNode
    | OrganisationNode
    | AggregateNode
    | UnitNode
    | SourceNode
    | DatasetNode
    | ProvenanceNode
    | OtherNode
)

NodeOutcome = HeldNode | NotHeldEntry


async def describe_nodes(dcids: Sequence[str]) -> dict[str, NodeOutcome]:
    """Read every dcid in `dcids` through the node batch route, keyed by dcid.

    Args:
        dcids: The dcids to look up. Deduplicated order-preservingly before dispatch, so
            the returned mapping carries one entry per unique dcid.

    Returns:
        dcid -> the node's envelope, or a `NotHeldEntry` for a dcid the graph does not hold.

    Raises:
        UpstreamError: a batch request failed. Propagates the first failure in dcid order,
            mirroring `resolution.py`'s own rule: a caller sees exactly one exception.
        pydantic.ValidationError: a 2xx body did not match the generated wire models.
        ValueError: a 2xx body answered a different number of dcids, or an entry named a
            different dcid than the one requested.
    """
    unique_dcids = list(dict.fromkeys(dcids))
    if not unique_dcids:
        return {}
    client = get_client()
    chunks = list(itertools.batched(unique_dcids, _MAX_DCIDS_PER_REQUEST, strict=False))
    responses = await asyncio.gather(
        *(
            client.request("POST", _NODES_ROUTE, route=_NODES_ROUTE, json={"dcids": list(chunk)})
            for chunk in chunks
        ),
        return_exceptions=True,
    )
    successful_responses: list[httpx2.Response] = []
    for response in responses:
        if isinstance(response, BaseException):
            raise response
        successful_responses.append(response)

    outcomes: dict[str, NodeOutcome] = {}
    for chunk, response in zip(chunks, successful_responses, strict=True):
        entries = NodesResponse.model_validate(response.json()).nodes
        # `strict=True` turns a length mismatch into a `ValueError` instead of attributing
        # one dcid's entry to another.
        for dcid, entry in zip(chunk, entries, strict=True):
            if entry.ref != dcid:
                # A body describing a different node is not this dcid's answer; storing it
                # under the dcid asked for would show the model a wrong identity.
                raise ValueError(
                    f"the nodes route answered {entry.ref!r} for a request for {dcid!r}"
                )
            outcomes[dcid] = entry
    return outcomes


def variable_text(node: StatVarNode) -> str | None:
    """The line that stands for `node` in planner text: its `description`, else one line
    composed from its definition; `None` when it has neither."""
    return node.description or definition_line(node.stat_var.definition)


def described_variable_text(entry: Mapping[str, object]) -> str | None:
    """`variable_text` of one entry of a `describe` answer's `nodes`, which the tool dumps
    from the envelopes `describe_nodes` returns; `None` for an entry that is no variable."""
    try:
        node = StatVarNode.model_validate(entry)
    except ValidationError:
        return None
    return variable_text(node)


def definition_line(definition: VariableDefinition) -> str | None:
    """One plain line naming what a variable measures, from its card's definition.

    `Measures <measured property> of <population type> as <measure>; <property> <value>.`
    Each part takes its label, else its dcid, and a part the card does not name is left out.
    `None` when the definition names no part.
    """
    subject = " of ".join(
        _name(part)
        for part in (definition.measured_property, definition.population_type)
        if part is not None
    )
    if definition.measure is not None:
        measure = definition.measure.label
        subject = f"{subject} as {measure}" if subject else measure
    clauses = [f"Measures {subject}"] if subject else []
    clauses += [f"{_name(c.property)} {_name(c.value)}" for c in definition.constraints or ()]
    return "; ".join(clauses) + "." if clauses else None


def _name(ref: NodeRef) -> str:
    return ref.label or ref.ref
