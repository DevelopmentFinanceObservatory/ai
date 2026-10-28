"""The request trees dfo-apis' analysis runner computes a tool call from.

A verb call becomes a one-node tree (`verb_tree`): `{verb, arguments, inputs}`,
where `arguments` are the op-spec's parameters but its frame slots, and `inputs`
names the frames: `data`, and `other` for the second frame of a two-sided verb.
`data` and `other` fill the runner's slots as `_SECOND_SLOT` names them:
`right_frame_ref` for `sum_of` and `join`, `denominator_frame_ref` for `share_of`
and `quotient_of`. `compare` computes `data` minus `other`, the right side minus
the left, so its `right_frame_ref` is `data` and its `frame_ref` is `other`. A
two-sided call with no second frame sends `data` alone, and the runner reads it
on both sides, as these tools did.

A frame ref becomes a root node (`root_node`). An `ops_` pair is a frame the
runner stored, `{spec_id, content_version}`, which it computes again from its
stored recipe. A `spec_` pair is a frame `get_data` registered in the
materialisation store, whose mint record (`common.materialisation.MintRecord`)
names the dataset version apis holds and the options it was read with. Its root
is that version with the rendering the options state: `without_party_totals`,
`columns`, the enrichment columns, `[]` when the frame has none, and `groupby`
when the frame was fetched with a `breakdown` or a `stat_vars` sum, which the
dataset of the mint record is already summed for. A `spec_` pair with no mint
record is sent as a version read with no columns. The runner reads a root with
no other option, so a frame fetched with one refuses `rendering_not_recorded`.

`anchor_key` is no part of an op-spec, and the runner checks every rolling
literal against the latest period it finds itself, so no tree carries one. The
key names the anchor record `resolve_anchor` stored, whose `roots` are the frames
the anchor route was asked about. A rolling call sends them as the request's
`anchor_roots` (`Tree.anchor_roots`), each the root node `root_node` gives, so the
runner takes the latest period over the roots the literals were written from,
such as both sides of a pairing whose one side the call grows. A key that names
no readable record refuses `rolling_anchor_unresolved` before any call.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from common.materialisation import AnchorRecordIntegrityError, FrameRef, get_store

from analysis_mcp.identity import OpSpecVariant
from analysis_mcp.op_spec import Verb
from analysis_mcp.tool_errors import ToolRefusal

logger = logging.getLogger(__name__)

# The op-spec field holding each two-sided verb's second frame.
_SECOND_SLOT: Mapping[Verb, str] = {
    Verb.COMPARE: "right_frame_ref",
    Verb.SUM_OF: "right_frame_ref",
    Verb.JOIN: "right_frame_ref",
    Verb.SHARE_OF: "denominator_frame_ref",
    Verb.QUOTIENT_OF: "denominator_frame_ref",
}

# The mint options a runner root is read with: data-mcp's frame-layout Parquet, its
# enrichment columns, the party-total cut and the groupby of a breakdown or a sum
# (dfo-apis `analysis_roots.dataset_root`).
_READ_OPTIONS = frozenset(
    {"format", "bundle", "layout", "columns", "without_party_totals", "groupby"}
)

type Pair = tuple[str, str]


@dataclass(frozen=True, kw_only=True)
class Tree:
    """A request tree, and the frame ref each of its roots stands for, by the pair
    the root names: the runner's refusals name a root by that pair."""

    node: dict[str, Any]
    refs_by_root: Mapping[Pair, FrameRef]
    anchor_roots: tuple[dict[str, Any], ...] = ()


async def root_node(ref: FrameRef) -> tuple[dict[str, Any], Pair] | ToolRefusal:
    """The root node of `ref` and the pair it names, or the refusal for a frame the
    runner cannot read. See the module docstring."""
    if ref.spec_id.startswith("ops_"):
        if not ref.content_version.startswith("ocv_"):
            return ToolRefusal.frame_not_held(
                spec_id=ref.spec_id, content_version=ref.content_version
            )
        return ref.model_dump(), (ref.spec_id, ref.content_version)
    if not ref.content_version.startswith("cv_"):
        return ToolRefusal.frame_not_held(spec_id=ref.spec_id, content_version=ref.content_version)
    try:
        record = await get_store().get_mint_record(ref)
    except Exception:
        # Broad on purpose, as `presentation_mcp.client._platform_pair` reads it: the
        # store's failures belong to its backend. A derived pair sent as given is one
        # apis does not hold, which the runner refuses.
        logger.warning(
            "mint record of %s/%s unreadable; sending the pair as given",
            ref.spec_id,
            ref.content_version,
            exc_info=True,
        )
        record = None
    if record is None:
        node = {**ref.model_dump(), "rendering": {"without_party_totals": False, "columns": []}}
        return node, (ref.spec_id, ref.content_version)
    unread = sorted(set(record.options) - _READ_OPTIONS)
    if unread:
        return ToolRefusal.rendering_not_recorded(
            spec_id=ref.spec_id, content_version=ref.content_version, options=unread
        )
    dataset = record.dataset
    rendering = {
        "without_party_totals": record.options.get("without_party_totals") is True,
        "columns": list(record.options.get("columns") or ()),
    }
    if (groupby := record.options.get("groupby")) is not None:
        rendering["groupby"] = groupby
    node = {**dataset.model_dump(), "rendering": rendering}
    return node, (dataset.spec_id, dataset.content_version)


def _sides(op_spec: OpSpecVariant) -> tuple[FrameRef, FrameRef | None]:
    """The frames of `op_spec` that fill `data` and `other`."""
    slot = _SECOND_SLOT.get(op_spec.verb)
    second: FrameRef | None = None if slot is None else getattr(op_spec, slot)
    if second is None:
        return op_spec.frame_ref, None
    if op_spec.verb is Verb.COMPARE:
        return second, op_spec.frame_ref
    return op_spec.frame_ref, second


async def _anchor_roots(
    anchor_key: str, refs_by_root: dict[Pair, FrameRef]
) -> tuple[dict[str, Any], ...] | ToolRefusal:
    """The root node of each root of the anchor record under `anchor_key`, each
    pair added to `refs_by_root`, or the refusal for a record or root the runner
    cannot be sent."""
    try:
        record = await get_store().get_anchor_record(anchor_key)
    except AnchorRecordIntegrityError as exc:
        return ToolRefusal.rolling_anchor_unresolved(
            detail="anchor_record_unreadable", reason=exc.reason, context={"anchor_key": anchor_key}
        )
    if record is None:
        return ToolRefusal.rolling_anchor_unresolved(
            detail="anchor_record_missing",
            reason=f"no record is stored under {anchor_key}",
            context={"anchor_key": anchor_key},
        )
    nodes: list[dict[str, Any]] = []
    for ref in record.roots:
        root = await root_node(ref)
        if isinstance(root, ToolRefusal):
            return root
        node, pair = root
        nodes.append(node)
        refs_by_root.setdefault(pair, ref)
    return tuple(nodes)


async def verb_tree(op_spec: OpSpecVariant, *, anchor_key: str | None = None) -> Tree | ToolRefusal:
    """The one-node tree of `op_spec`, with the anchor roots of the record under
    `anchor_key` when given, or the refusal for a frame it reads."""
    slots = {"verb", "frame_ref", *_SECOND_SLOT.values()}
    arguments = op_spec.model_dump(mode="json", exclude=slots, exclude_none=True)
    inputs: dict[str, Any] = {}
    refs_by_root: dict[Pair, FrameRef] = {}
    data, other = _sides(op_spec)
    for name, ref in (("data", data), ("other", other)):
        if ref is None:
            continue
        root = await root_node(ref)
        if isinstance(root, ToolRefusal):
            return root
        inputs[name], pair = root
        refs_by_root[pair] = ref
    anchor_roots: tuple[dict[str, Any], ...] = ()
    if anchor_key is not None:
        anchored = await _anchor_roots(anchor_key, refs_by_root)
        if isinstance(anchored, ToolRefusal):
            return anchored
        anchor_roots = anchored
    node = {"verb": str(op_spec.verb), "arguments": arguments, "inputs": inputs}
    return Tree(node=node, refs_by_root=refs_by_root, anchor_roots=anchor_roots)
