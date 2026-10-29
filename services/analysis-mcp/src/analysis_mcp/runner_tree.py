"""The request trees dfo-apis' analysis runner computes a tool call from.

A verb call becomes a one-node tree (`verb_tree`) shaped `{verb, arguments,
inputs}`. The `arguments` are the op-spec's parameters but its frame slots, and
`inputs` names the frames `data` and, for the second frame of a two-sided verb,
`other`. They fill the runner's slots as `_SECOND_SLOT` names them:
`right_frame_ref` for `sum_of` and `join`, `denominator_frame_ref` for `share_of`
and `quotient_of`. `compare` computes `data` minus `other`, the right side minus
the left, so its `right_frame_ref` is `data` and its `frame_ref` is `other`. A
two-sided call with no second frame sends `data` alone, and the runner reads it
on both sides.

A frame ref becomes a root node (`root_node`). An `ops_` pair is a frame the
runner stored, `{spec_id, content_version}`, which it computes again from its
stored recipe. A `spec_` pair is a frame `get_data` registered in the
materialisation store, whose mint record (`common.materialisation.MintRecord`)
names the dataset version apis holds and the options it was read with. Its root
is that version with the rendering the options state: `without_party_totals`,
`columns`, the enrichment columns sorted and each listed once, `[]` when the frame
has none, and `groupby` when the frame was fetched with a `breakdown` or a
`stat_vars` sum, which the dataset of the mint record is already summed for. A `spec_` pair with no mint
record is sent as a version read with no columns. A record that does not decode
refuses `frame_integrity_failed`, and any other failed read of it
`service_unavailable`, which a retry may mend. The runner reads a root with
no other option, so a frame fetched with one refuses `rendering_not_recorded`.

A Spec root (`spec_root`) is the Spec alone, `{spec_id}`, which the runner reads
at its newest version. With no `rendering` it takes the enrichment columns the
dataset's roles need (dfo-apis `analysis_roots.enrichment_columns`), the columns
the public `get_data` adds, so its root record, and the `ops_` id of a call over
it, are those of the same call over the frame `get_data` returns. A bare
`spec_id` argument is held in the op-spec as `newest_ref`, a pair naming no
version, and `verb_tree` sends its Spec root in its place (`replaced`). A frame
fetched again after the runner lost it is the Spec root of its mint record's
rendering, or of no rendering when no record is held.

`place_level` puts a `filter` step keeping the places at that level under the
verb, on each input, so the call computes the tree an explicit `filter` call and
the verb over its result would, and gets its ids. The step's predicate reads the
`{role}_type` column of the first typed role (`common.place_type.TYPED_ROLES`)
the side keys by (`_place_role`), else of `observationAbout`, the one place role
whose mixed levels refuse `place_levels_mixed`.

The runner checks every rolling literal against the latest period it finds
itself, so no tree carries `anchor_key`. The key names the anchor record
`resolve_anchor` stored, whose `roots` are the frames the anchor route was asked
about. A rolling call sends them as the request's `anchor_roots`
(`runner_client.Anchor`), each the root node `root_node` gives, so the runner
takes the latest period over the roots the literals were written from, such as
both sides of a pairing whose one side the call grows. The anchor roots stay
apart from the frames the call passed (`Tree.refs_by_root`), so a refusal over
one of them names the record. A key that names no readable record, a record
naming a derived (`ops_`) root, and a call whose `rolling.population` is not
the record's populations refuse `rolling_anchor_unresolved` before any call. The
answer is checked to have read no root outside the record (`runner_client`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from common.materialisation import (
    AnchorRecordIntegrityError,
    FrameIntegrityError,
    FrameRef,
    MintRecord,
    get_store,
)
from common.place_type import PLACE_ROLE, TYPED_ROLES, PlaceLevel, type_column

from analysis_mcp.identity import OpSpecVariant
from analysis_mcp.op_spec import (
    AggregateOpSpec,
    FilterOpSpec,
    PlaceLevelPredicate,
    Populations,
    RankOpSpec,
    Verb,
)
from analysis_mcp.runner_client import DEFAULT_RETRY_AFTER_S, Anchor, Pair
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

# The op-spec frame slots, which the tree's `inputs` fill in place of `arguments`.
_FRAME_SLOTS = frozenset({"verb", "frame_ref", *_SECOND_SLOT.values()})

# The content_version a bare `spec_id` argument is held under until the runner reads
# the Spec (`newest_ref`). It names no version: `verb_tree` sends the Spec root in
# its place, so it reaches no request.
_NEWEST = "cv_" + "0" * 32


def newest_ref(spec_id: str) -> FrameRef:
    """The pair a bare `spec_id` argument is held under in an op-spec."""
    return FrameRef(spec_id=spec_id, content_version=_NEWEST)


@dataclass(frozen=True, kw_only=True)
class Tree:
    """A request tree, the caller's frame refs each of its roots stands for, by the
    pair the root names (the runner's refusals name a root by that pair), and a
    rolling call's anchor record."""

    node: dict[str, Any]
    refs_by_root: Mapping[Pair, tuple[FrameRef, ...]]
    anchor: Anchor | None = None


async def _mint_record(ref: FrameRef) -> MintRecord | None | ToolRefusal:
    """The mint record of the `get_data` frame `ref`, or the refusal for a failed read."""
    try:
        return await get_store().get_mint_record(ref)
    except FrameIntegrityError as exc:
        # A record that does not decode reads the same on every retry.
        return ToolRefusal.frame_integrity_failed(
            spec_id=ref.spec_id, content_version=ref.content_version, reason=exc.reason
        )
    except Exception:
        # Broad on purpose, as `presentation_mcp.client._platform_pair` reads it, since
        # the store's failures belong to its backend. Sent as given, the frame pair is
        # one apis does not hold, which would tell the caller its frame is gone.
        logger.warning(
            "mint record of %s/%s unreadable", ref.spec_id, ref.content_version, exc_info=True
        )
        return ToolRefusal.frame_store_unavailable(
            spec_id=ref.spec_id,
            content_version=ref.content_version,
            retry_after_s=DEFAULT_RETRY_AFTER_S,
        )


async def root_node(ref: FrameRef) -> tuple[dict[str, Any], Pair] | ToolRefusal:
    """The root node of `ref` and the pair it names, or the refusal for a frame the
    runner cannot read. See the module docstring."""
    derived = ref.spec_id.startswith("ops_")
    if not ref.content_version.startswith("ocv_" if derived else "cv_"):
        return ToolRefusal.frame_not_held(spec_id=ref.spec_id, content_version=ref.content_version)
    if derived:
        return ref.model_dump(), (ref.spec_id, ref.content_version)
    record = await _mint_record(ref)
    if isinstance(record, ToolRefusal):
        return record
    if record is None:
        node = {**ref.model_dump(), "rendering": {"without_party_totals": False, "columns": []}}
        return node, (ref.spec_id, ref.content_version)
    unread = sorted(set(record.options) - _READ_OPTIONS)
    if unread:
        return ToolRefusal.rendering_not_recorded(
            spec_id=ref.spec_id, content_version=ref.content_version, options=unread
        )
    dataset = record.dataset
    node = {**dataset.model_dump(), "rendering": _rendering(record)}
    return node, (dataset.spec_id, dataset.content_version)


def _rendering(record: MintRecord) -> dict[str, Any]:
    """The rendering of a root read as `record`'s options state."""
    rendering: dict[str, Any] = {
        "without_party_totals": record.options.get("without_party_totals") is True,
        # Sorted and listed once, as dfo-apis keeps a root's columns
        # (`analysis_roots.mint_options`), so the recipe's normal form is the one
        # dfo-apis computes.
        "columns": sorted(set(record.options.get("columns") or ())),
    }
    if (groupby := record.options.get("groupby")) is not None:
        rendering["groupby"] = groupby
    return rendering


async def spec_root(ref: FrameRef) -> tuple[dict[str, Any], MintRecord | None]:
    """The Spec root that fetches the `spec_` frame `ref` again at its newest version,
    with the rendering its mint record states, and the record. With no record, or
    one that cannot be read or read again, the root takes the default rendering, as
    `get_data` with default arguments fetches it."""
    record = await _mint_record(ref)
    if not isinstance(record, MintRecord) or set(record.options) - _READ_OPTIONS:
        return {"spec_id": ref.spec_id}, None
    return {"spec_id": ref.spec_id, "rendering": _rendering(record)}, record


def _sides(op_spec: OpSpecVariant) -> tuple[FrameRef, FrameRef | None]:
    """The frames of `op_spec` that fill `data` and `other`."""
    slot = _SECOND_SLOT.get(op_spec.verb)
    second: FrameRef | None = None if slot is None else getattr(op_spec, slot)
    if second is None:
        return op_spec.frame_ref, None
    if op_spec.verb is Verb.COMPARE:
        return second, op_spec.frame_ref
    return op_spec.frame_ref, second


async def _anchor(anchor_key: str, *, populations: Populations) -> Anchor | ToolRefusal:
    """The anchor record under `anchor_key` as a rolling call over `populations`
    sends it, or the refusal for a record the call cannot be sent with."""
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
    stated = [population.model_dump(mode="json") for population in populations]
    if stated != [population.model_dump(mode="json") for population in record.populations]:
        return ToolRefusal.rolling_anchor_unresolved(
            detail="populations_differ",
            reason=f"rolling.population is not the populations {anchor_key} was resolved over",
            context={"anchor_key": anchor_key},
        )
    nodes: list[dict[str, Any]] = []
    pairs: set[Pair] = set()
    for ref in record.roots:
        if ref.spec_id.startswith("ops_"):
            # The runner takes a latest period over dataset roots only, and
            # `resolve_anchor` stores no record over a derived one.
            return ToolRefusal.rolling_anchor_unresolved(
                detail="anchor_record_unreadable",
                reason=f"the record under {anchor_key} names the derived frame {ref.spec_id}",
                context={"anchor_key": anchor_key},
            )
        root = await root_node(ref)
        if isinstance(root, ToolRefusal):
            return root
        node, pair = root
        nodes.append(node)
        pairs.add(pair)
    return Anchor(key=anchor_key, nodes=tuple(nodes), pairs=frozenset(pairs))


def _place_role(keys: tuple[str, ...]) -> str:
    """The role whose places a `place_level` step keeps on a side keyed by `keys`."""
    return next((key for key in keys if key in TYPED_ROLES), PLACE_ROLE)


def _side_keys(op_spec: OpSpecVariant) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The columns `op_spec` keys `data` and `other` by."""
    if isinstance(op_spec, RankOpSpec):
        return op_spec.rank_by, op_spec.rank_by
    if isinstance(op_spec, AggregateOpSpec):
        return op_spec.group_by, op_spec.group_by
    numerator_by: tuple[str, ...] = getattr(op_spec, "numerator_by", ())
    denominator_by: Mapping[str, str] = getattr(op_spec, "denominator_by", {})
    return numerator_by, tuple(denominator_by.get(key, key) for key in numerator_by)


def _level_step(
    node: dict[str, Any], ref: FrameRef, *, level: PlaceLevel, keys: tuple[str, ...]
) -> dict[str, Any]:
    """`node` under the `filter` step an explicit `filter` call keeping the places at
    `level` sends, its arguments built as `verb_tree` builds every verb's."""
    predicate = PlaceLevelPredicate(column=type_column(_place_role(keys)), level=level)
    step = FilterOpSpec(frame_ref=ref, predicates=(predicate,))
    arguments = step.model_dump(mode="json", exclude=set(_FRAME_SLOTS), exclude_none=True)
    return {"verb": str(Verb.FILTER), "arguments": arguments, "inputs": {"data": node}}


async def verb_tree(
    op_spec: OpSpecVariant,
    *,
    anchor_key: str | None = None,
    place_level: PlaceLevel | None = None,
    replaced: Mapping[Pair, dict[str, Any]] | None = None,
) -> Tree | ToolRefusal:
    """The one-node tree of `op_spec`, with the anchor roots of the record under
    `anchor_key` when given, or the refusal for a frame it reads. A frame ref whose
    pair is in `replaced` is sent as the root node it maps to, and `place_level`
    filters each input to that level first (see the module docstring)."""
    arguments = op_spec.model_dump(mode="json", exclude=set(_FRAME_SLOTS), exclude_none=True)
    inputs: dict[str, Any] = {}
    refs_by_root: dict[Pair, tuple[FrameRef, ...]] = {}
    data, other = _sides(op_spec)
    keys = dict(zip(("data", "other"), _side_keys(op_spec), strict=True))
    for name, ref in (("data", data), ("other", other)):
        if ref is None:
            continue
        if replaced is not None and (ref.spec_id, ref.content_version) in replaced:
            node = replaced[ref.spec_id, ref.content_version]
            pair: Pair | None = (
                (node["spec_id"], node["content_version"]) if "content_version" in node else None
            )
        else:
            root = await root_node(ref)
            if isinstance(root, ToolRefusal):
                return root
            node, pair = root
        inputs[name] = (
            node
            if place_level is None
            else _level_step(node, ref, level=place_level, keys=keys[name])
        )
        # A Spec root names no pair: the runner names the version it read.
        if pair is not None:
            held = refs_by_root.get(pair, ())
            refs_by_root[pair] = held if ref in held else (*held, ref)
    anchor: Anchor | None = None
    # `op_spec.with_anchor_key` gives a key to every rolling call and to no other.
    rolling = getattr(op_spec, "rolling", None)
    if anchor_key is not None and rolling is not None:
        anchored = await _anchor(anchor_key, populations=rolling.population)
        if isinstance(anchored, ToolRefusal):
            return anchored
        anchor = anchored
    node = {"verb": str(op_spec.verb), "arguments": arguments, "inputs": inputs}
    return Tree(node=node, refs_by_root=refs_by_root, anchor=anchor)
