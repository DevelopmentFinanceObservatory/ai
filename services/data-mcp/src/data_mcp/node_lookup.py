"""Node lookup seam: dcids to shaped node descriptions, via dfo-apis' Graph node door.

`describe_nodes` resolves each dcid it's given to one of four node shapes, or
notes it as a dcid this deployment's graph does not hold. The node door
takes exactly one dcid per request -- there is no batch route -- so a dcid
the graph does not hold arrives as a failure on its own request rather than
as a field in a batch response. Only `node_not_held` (404) is partial in
this way: any other failure on one dcid (a 503 from pool exhaustion, a 504,
...) fails the whole call, even when every other dcid in the batch would
have succeeded on its own.

Facet values per axis, processing level, and sources are not served by the
node door today. `facets` and `sources` are typed permissively (`Any`,
defaulting to `None`) rather than mirroring the wire's own `None` typing, so
a later upstream population of either field doesn't break validation and
refuse every StatVar lookup.

Like `resolution.py`, this module returns plain data and never imports
`data_mcp.tool_errors`; the calling tool owns the refusal grammar. The one
exception this module does catch is an `UpstreamError` coded
`node_not_held`, which becomes a `NotHeldNode` data outcome rather than
propagating -- every other exception is left to propagate uncaught.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, computed_field

from data_mcp.client import GRAPH_PREFIX
from data_mcp.client_holder import get_client
from data_mcp.errors import UpstreamError

_NODE_ROUTE_TEMPLATE = f"{GRAPH_PREFIX}/node/{{dcid}}"

# This service's own bound on concurrent node reads, not a published upstream
# limit. The node door's own admission pool defaults to 4 concurrent
# readers, non-blocking -- a request arriving with every slot held is
# refused outright with a 503 rather than queued. An `entity` shape costs a
# second, sequential read after the first completes, so one request holds at
# most one slot at a time -- four concurrent requests demand four slots
# against a pool of four, and the bound matches. Bounding this side's own
# fan-out to the same number keeps an eight-dcid `describe` call from
# driving that admission gate into 503s. A 503 from the gate is retryable,
# and the client retries it before the batch can fail on it. This bound
# cannot guarantee admission on its own, since the resolve door shares the
# same executor.
_MAX_CONCURRENT_NODE_READS = 4

# Applied to every model in this module, mirroring `manifest.py`'s own
# reasoning: `forbid` would break every lookup the day upstream adds a
# field, `allow` would forward an unreviewed field straight to the model,
# and `ignore` does neither -- an unrecognised field is silently dropped.
_UNSEEN_FIELDS_ARE_DROPPED = ConfigDict(extra="ignore")


class Freshness(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    release_frequency: str | None = None
    refresh_schedule: str | None = None


class ContainedInPlace(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    dcid: str
    label: str | None = None


class StatVarNode(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    shape: Literal["stat_var"]
    dcid: str
    # The node door types `label` (every shape) and `definition` as
    # nullable; the UN System StatVars (`undata/sdg/*`, `undata/who/*`)
    # arrive with a label and no definition.
    label: str | None = None
    definition: str | None = None
    roles: list[str]
    population_type: str | None = None
    processing_level: str | None = None
    # `Any`, not `None`: mirroring the wire's own permanently-`None` typing
    # here would make this service fail validation -- and refuse every
    # StatVar lookup with `upstream_response_unusable` -- the day upstream
    # starts populating either field.
    facets: Any = None
    sources: Any = None


class SourceNode(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    shape: Literal["source"]
    dcid: str
    label: str | None = None
    license: str | None = None
    license_type: str | None = None
    description: str | None = None
    url: str | None = None
    freshness: Freshness


class EntityNode(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    shape: Literal["entity"]
    dcid: str
    label: str | None = None
    entity_type: str | None = None
    contained_in: list[ContainedInPlace]


class GenericNode(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    shape: Literal["node"]
    dcid: str
    label: str | None = None
    types: list[str]
    properties: dict[str, list[str]]

    @computed_field
    @property
    def is_classification_value(self) -> bool:
        """True when every declared type is a classification enum, e.g. `GeographicalIndicatorEnum`.

        A node like this labels a category a StatVar's schema can bind to. A
        caller who read this dcid out of another tool's digest more likely
        meant a place, dataset, or other entity.
        """
        return bool(self.types) and all(t.endswith("Enum") for t in self.types)


NodeShape = Annotated[
    StatVarNode | SourceNode | EntityNode | GenericNode, Field(discriminator="shape")
]

_NODE_SHAPE_ADAPTER: TypeAdapter[Any] = TypeAdapter(NodeShape)


@dataclass(frozen=True, kw_only=True)
class NotHeldNode:
    """A dcid the node door itself resolved as well-formed, but does not hold."""

    dcid: str


NodeOutcome = StatVarNode | SourceNode | EntityNode | GenericNode | NotHeldNode


def _outcome_from_response(payload: Any) -> StatVarNode | SourceNode | EntityNode | GenericNode:
    """Validate one node door response body into its shape.

    Raises:
        pydantic.ValidationError: `payload`'s `shape` is not one of the four
            this service recognises, or the body otherwise doesn't match the
            shape it claims.
    """
    return _NODE_SHAPE_ADAPTER.validate_python(payload)


async def _fetch_one(dcid: str, *, semaphore: asyncio.Semaphore) -> NodeOutcome:
    client = get_client()
    async with semaphore:
        try:
            response = await client.request(
                "GET", f"{GRAPH_PREFIX}/node/{dcid}", route=_NODE_ROUTE_TEMPLATE
            )
        except UpstreamError as exc:
            if exc.code == "node_not_held":
                return NotHeldNode(dcid=dcid)
            raise
    node = _outcome_from_response(response.json())
    if node.dcid != dcid:
        # A shape-valid body describing a different node -- an upstream
        # routing defect, a stale cache, a crossed response -- is not this
        # dcid's answer. Raising here, rather than storing it under the
        # dcid that was asked for, keeps the caller from rendering a wrong
        # identity to the model as though it were the one requested.
        raise ValueError(f"the node door returned dcid {node.dcid!r} for a request for {dcid!r}")
    return node


async def describe_nodes(dcids: Sequence[str]) -> dict[str, NodeOutcome]:
    """Resolve every dcid in `dcids` to its node outcome, keyed by dcid.

    Args:
        dcids: The dcids to look up. Deduplicated order-preservingly before
            dispatch, so a dcid repeated in the input costs one request, not
            one per occurrence, and the returned mapping carries one entry
            per unique dcid.

    Returns:
        dcid -> `NodeOutcome`: one of the four node shapes, or `NotHeldNode`
        for a dcid the node door reports as not held.

    Raises:
        UpstreamError: a request for one dcid failed with anything other
            than `node_not_held` -- pool exhaustion, a transport failure, an
            unmounted surface, and so on. Propagates the first such failure
            in dcid order, however many requests failed, mirroring
            `resolution.py`'s own rule: a caller sees exactly one exception.
        pydantic.ValidationError: a 2xx body didn't match any of the four
            node shapes this service recognises.
        ValueError: a 2xx body validated but named a different dcid than
            the one requested.
    """
    unique_dcids = list(dict.fromkeys(dcids))
    if not unique_dcids:
        return {}
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_NODE_READS)
    results = await asyncio.gather(
        *(_fetch_one(dcid, semaphore=semaphore) for dcid in unique_dcids),
        return_exceptions=True,
    )
    outcomes: dict[str, NodeOutcome] = {}
    for dcid, result in zip(unique_dcids, results, strict=True):
        if isinstance(result, BaseException):
            raise result
        outcomes[dcid] = result
    return outcomes
