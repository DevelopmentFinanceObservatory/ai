"""The conversation-scoped store for the manifest lineage of each frame.

A frame's *lineage* is the ordered tuple of `ProvenanceInput`s it rests on:
one for a retrieval's own frame, and one per upstream retrieval for a
derived frame (a join, a compare, a share of two datasets, and whatever is
built on top of them). Component assembly
(`harness.mcp_client.McpExecutor._assemble_component`) reads the lineage of
the frame a component was written from to build its citation, sources,
warnings and freshness, and withholds the card when the frame has none.

A retrieval's own lineage is written by the direct-fetch path
(`McpExecutor._get_data`, called from the orchestrator's `fetch` tool) and by
the approved-plan path (a `get_data` step run through
`compiler.surface.ToolSurface`, observed by `harness.studio_translate.
_RecordingSurface`). `alias` writes a terminal frame's lineage from the
retrievals under it, with the count of distinct StatVars the frame itself
carries when the step that produced it declared one. A derived frame can
carry fewer StatVars than its retrievals resolved (a family frame filtered
to one member), so a card counts its indicators from that figure
(`ProvenanceStore.stat_var_count`). A frame can also carry more than its
declared count says: a summed retrieval's frame holds one family ref that
stands for every StatVar the sum added up, and so does any frame derived
from it, so that ref counts as those StatVars, each counted once however
many carried retrievals hold it (`_carried_stat_var_count`). The frame of a
verb that writes its own StatVar (`BOTH_SIDES_VERBS`: `sum_of`,
`quotient_of`, `share_of`) holds that one ref, which counts as the distinct
StatVars of the retrievals under its sides. Both writers store a manifest
with its whole `trace.resolved_stat_vars`, read back first when the
`get_data` result left a long one out (`manifest_with_stat_vars`).

Scoped per conversation (`harness.conversation.current_conversation_id`),
like `McpExecutor`'s other per-conversation state. Bounded two ways: an LRU
cap on frame keys within one conversation, and an LRU cap on the number of
conversation buckets held at once. The conversation-count cap does the
bounding. Nothing in this process observes a conversation ending
(`harness.studio_state.StudioStore` has the same gap) and the studio CLI
mints a fresh conversation id per run, so each run opens one more bucket
that nothing closes. A single conversation rarely holds more than a handful
of frames awaiting a `write_component`, so the per-bucket cap is headroom.

The store is in-process, like `StudioStore` and the default in-memory
session service. A multi-replica deployment moves all of them behind a
shared backend at once.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from data_mcp.stat_var_list import with_stat_var_list
from fastmcp.exceptions import ResourceError

logger = logging.getLogger(__name__)

# Per-conversation entry ceiling; least-recently-used eviction past it.
_MAX_MANIFESTS_PER_CONVERSATION = 64

# Global ceiling on the number of conversation buckets held at once;
# least-recently-used bucket eviction past it. See the module docstring for
# why this, not the per-bucket cap above, is the bound that matters.
_MAX_CONVERSATIONS = 256

_FrameKey = tuple[str | None, str | None]


@dataclass(frozen=True, slots=True)
class OwnStatVar:
    """How a card names the StatVar a verb writes for its own result:
    `joiner` goes between the two sides' names, and `description` says what
    the verb did with them. `equal_sides` goes before the one name of two
    sides holding the same StatVars, in place of the joined form. dfo-apis
    words the label of a published frame the same way
    (`presentation_store.composed_stat_vars`)."""

    joiner: str
    description: str
    equal_sides: str


@dataclass(frozen=True, slots=True)
class BothSidesVerb:
    """A verb whose result carries the StatVars of both frames it reads.

    `second_frame` is the argument it takes its second frame under, absent
    from a step whose two sides read one frame. `own_stat_var` is set for a
    verb that writes every result row under `dcid:<its spec_id>`, a ref no
    graph node describes, which stands for the StatVars under both sides.
    """

    second_frame: str
    own_stat_var: OwnStatVar | None = None


# The analysis-mcp verbs whose result carries both sides' StatVars, by tool
# name. `compare`'s right side supplies values only, so it is not here.
BOTH_SIDES_VERBS: Mapping[str, BothSidesVerb] = {
    "join": BothSidesVerb(second_frame="right_frame_ref"),
    "sum_of": BothSidesVerb(
        second_frame="right_frame_ref",
        own_stat_var=OwnStatVar(
            joiner=" plus ",
            description="The two added together for each year both report.",
            equal_sides="",
        ),
    ),
    "quotient_of": BothSidesVerb(
        second_frame="denominator_frame_ref",
        own_stat_var=OwnStatVar(
            joiner=" per ",
            description="The first divided by the second for each year both report.",
            equal_sides="Ratio of ",
        ),
    ),
    "share_of": BothSidesVerb(
        second_frame="denominator_frame_ref",
        own_stat_var=OwnStatVar(
            joiner=" as a share of ",
            description="The first as a share of the second, in per cent, for each year both "
            "report.",
            equal_sides="Share of ",
        ),
    ),
}

# One part of a verb's side: a retrieval's frame key, or the StatVar ref an
# earlier verb wrote for its own result, itself a key of `OwnStatVars`.
type SidePart = _FrameKey | str

# The StatVar ref a verb wrote for its own result (`BothSidesVerb
# .own_stat_var`), mapped to the verb's tool name and the parts of its first
# side and of its second.
type OwnStatVars = Mapping[str, tuple[str, tuple[SidePart, ...], tuple[SidePart, ...]]]


@dataclass(frozen=True, slots=True)
class ProvenanceInput:
    """One retrieval a frame rests on: its frame key and its mint manifest."""

    spec_id: str | None
    content_version: str | None
    manifest: dict[str, Any]

    @property
    def served_ref(self) -> dict[str, str | None]:
        """The `{spec_id, content_version}` the dfo-apis datasets door serves this
        retrieval under: the manifest's `definition.spec_id` and `content_version`.

        The frame key can differ. A frame minted with extra columns is held under a
        `content_version` of its own (`retrieval_frame_ref`), which the datasets door
        does not serve. A manifest that states no pair gives the frame key.
        """
        served = served_frame_ref(self.manifest)
        if served["spec_id"] and served["content_version"]:
            return served
        return {"spec_id": self.spec_id, "content_version": self.content_version}


# The retrievals a frame rests on, in workflow order. Never empty.
type Lineage = tuple[ProvenanceInput, ...]


@dataclass(frozen=True, slots=True)
class _Entry:
    """What the store holds for one frame: its lineage, the distinct
    StatVars the frame carries when its producing step declared a count,
    and the StatVar refs the verbs of the frame's run wrote for their own
    results."""

    lineage: Lineage
    stat_var_count: int | None = None
    own_stat_vars: OwnStatVars | None = None


class ProvenanceStore:
    """Frame lineages, keyed by conversation and `(spec_id, content_version)`."""

    def __init__(self) -> None:
        self._by_conversation: OrderedDict[str, OrderedDict[_FrameKey, _Entry]] = OrderedDict()
        # Keys dropped by either eviction path, most-recent last, itself
        # capped so it cannot become the unbounded structure this store
        # exists to avoid. Lets `was_evicted` tell a caller "this lineage
        # was here and got dropped" apart from "this was never written" --
        # a distinction `get`'s bare `None` cannot carry without changing
        # its return type for every caller.
        self._evicted: OrderedDict[tuple[str, _FrameKey], None] = OrderedDict()

    def put(
        self,
        *,
        conversation: str,
        spec_id: str | None,
        content_version: str | None,
        manifest: dict[str, Any],
    ) -> None:
        """Store `manifest` as the one-input lineage of the retrieval frame it was minted for."""
        self._store(
            conversation=conversation,
            key=(spec_id, content_version),
            entry=_Entry((ProvenanceInput(spec_id, content_version, manifest),)),
        )

    def put_retrieval(
        self, *, conversation: str, manifest: dict[str, Any]
    ) -> dict[str, str | None]:
        """Store a successful `get_data` result's manifest as the lineage of its
        frame, and return the pair the frame is described and drawn by
        (`ProvenanceInput.served_ref`).

        The lineage is keyed by `retrieval_frame_ref`, the pair later steps name
        the frame by, and also by the served pair when it differs: dfo-apis serves
        a final retrieval under that pair, and its card reads its lineage there.
        Both keys hold the one retrieval.
        """
        frame = retrieval_frame_ref(manifest)
        retrieval = ProvenanceInput(frame["spec_id"], frame["content_version"], manifest)
        served = retrieval.served_ref
        entry = _Entry((retrieval,))
        for key in dict.fromkeys(
            [
                (frame["spec_id"], frame["content_version"]),
                (served["spec_id"], served["content_version"]),
            ]
        ):
            self._store(conversation=conversation, key=key, entry=entry)
        return served

    def alias(
        self,
        *,
        conversation: str,
        spec_id: str | None,
        content_version: str | None,
        sources: Sequence[_FrameKey],
        stat_var_count: int | None = None,
        carried: Sequence[_FrameKey] = (),
        summed: Sequence[Sequence[_FrameKey]] = (),
        own_stat_vars: OwnStatVars | None = None,
    ) -> bool:
        """Write the lineage of the frame `(spec_id, content_version)` from the frames in `sources`.

        `stat_var_count` is the distinct StatVars the frame carries, as the
        step that produced it declared them (`analysis_mcp.result.VerbResult
        .stat_var_count`, or `get_data`'s `stat_var_count` for a retrieval's
        own frame), or `None` when it declared none. `carried` names the
        retrievals among `sources` whose StatVars the frame carries. A summed
        one carries a single family ref standing for the StatVars it sums
        (`summed_stat_var_count`), so the count held with the lineage, and
        read back by `stat_var_count`, counts that ref as those StatVars
        (`_carried_stat_var_count`). `summed` holds one entry per ref a verb
        wrote for its own result (a `sum_of`'s, a `quotient_of`'s, a
        `share_of`'s) that the frame carries, the retrievals among `sources`
        under that verb's sides, and the count takes that ref as their
        distinct StatVars. `own_stat_vars` is held for `own_stat_vars` to read back.

        The lineage is every `sources` lineage flattened in order, with a
        retrieval that appears twice kept once. All-or-nothing: when any
        source has no lineage in the store (never written, or evicted),
        nothing is written and this returns `False`, since a lineage missing
        one of the retrievals a frame rests on would state a narrower
        provenance than the frame has. `True` means the lineage was written.
        """
        members: dict[_FrameKey, ProvenanceInput] = {}
        for source_spec_id, source_content_version in sources:
            lineage = self.get(
                conversation=conversation,
                spec_id=source_spec_id,
                content_version=source_content_version,
            )
            if lineage is None:
                return False
            for member in lineage:
                members.setdefault((member.spec_id, member.content_version), member)
        if not members:
            return False
        if stat_var_count:
            stat_var_count = _carried_stat_var_count(
                stat_var_count,
                [members[key] for key in dict.fromkeys(carried) if key in members],
                summed=[
                    [members[key] for key in dict.fromkeys(under) if key in members]
                    for under in summed
                ],
            )
        self._store(
            conversation=conversation,
            key=(spec_id, content_version),
            entry=_Entry(tuple(members.values()), stat_var_count, own_stat_vars),
        )
        return True

    def _store(self, *, conversation: str, key: _FrameKey, entry: _Entry) -> None:
        bucket = self._by_conversation.get(conversation)
        if bucket is None:
            bucket = OrderedDict()
            self._by_conversation[conversation] = bucket
            while len(self._by_conversation) > _MAX_CONVERSATIONS:
                evicted_id, evicted_bucket = self._by_conversation.popitem(last=False)
                for evicted_key in evicted_bucket:
                    self._record_eviction(evicted_id, evicted_key)
                logger.warning(
                    "provenance: dropped conversation %r (%d frame(s)) past the "
                    "%d-conversation bound",
                    evicted_id,
                    len(evicted_bucket),
                    _MAX_CONVERSATIONS,
                )
        else:
            self._by_conversation.move_to_end(conversation)
        bucket[key] = entry
        bucket.move_to_end(key)
        while len(bucket) > _MAX_MANIFESTS_PER_CONVERSATION:
            evicted_key, _ = bucket.popitem(last=False)
            self._record_eviction(conversation, evicted_key)
            logger.warning(
                "provenance: dropped frame %r for conversation %r past the "
                "%d-per-conversation bound",
                evicted_key,
                conversation,
                _MAX_MANIFESTS_PER_CONVERSATION,
            )

    def get(
        self, *, conversation: str, spec_id: str | None, content_version: str | None
    ) -> Lineage | None:
        entry = self._entry(conversation, (spec_id, content_version))
        return entry.lineage if entry is not None else None

    def stat_var_count(
        self, *, conversation: str, spec_id: str | None, content_version: str | None
    ) -> int | None:
        """The distinct StatVars the frame `(spec_id, content_version)` carries,
        as `alias` recorded them. `None` for a frame the store does not hold
        and for one written without a count: a retrieval's own frame that
        `put` wrote and no `alias` has counted, whose StatVars are the ones
        its manifest resolved or summed."""
        entry = self._entry(conversation, (spec_id, content_version))
        return entry.stat_var_count if entry is not None else None

    def own_stat_vars(
        self, *, conversation: str, spec_id: str | None, content_version: str | None
    ) -> OwnStatVars:
        """Each StatVar ref a verb of the run behind the frame
        `(spec_id, content_version)` wrote for its own result, with the
        verb's tool name and the parts of its first and second side
        (`SidePart`), as `alias` recorded them. Empty when there are none."""
        entry = self._entry(conversation, (spec_id, content_version))
        return (entry.own_stat_vars if entry is not None else None) or {}

    def _entry(self, conversation: str, key: _FrameKey) -> _Entry | None:
        """The entry held for `key`, marked most recently used."""
        bucket = self._by_conversation.get(conversation)
        if bucket is None:
            return None
        self._by_conversation.move_to_end(conversation)
        entry = bucket.get(key)
        if entry is not None:
            bucket.move_to_end(key)
        return entry

    def was_evicted(
        self, *, conversation: str, spec_id: str | None, content_version: str | None
    ) -> bool:
        """True if this key was written and later dropped by one of the two
        bounds, as opposed to never written at all. A caller that gets
        `None` from `get` and wants to tell "no such lineage" apart from
        "this store discarded it under pressure" checks here."""
        return (conversation, (spec_id, content_version)) in self._evicted

    def _record_eviction(self, conversation: str, key: _FrameKey) -> None:
        record = (conversation, key)
        self._evicted[record] = None
        self._evicted.move_to_end(record)
        while len(self._evicted) > _MAX_MANIFESTS_PER_CONVERSATION:
            self._evicted.popitem(last=False)


def retrieval_frame_ref(result: Mapping[str, Any]) -> dict[str, str | None]:
    """The frame reference a successful `get_data` result states beside its manifest.

    The manifest's own `{spec_id, content_version}` names dfo-apis' materialisation.
    A frame minted with extra columns is held under a `content_version` of its own
    (`data_mcp.frame_materialisation.frame_ref_for`), so a retrieval's lineage is
    keyed by `frame_ref`, the reference every later step and `ProvenanceStore.alias`
    name it by, as well as by the manifest's pair (`ProvenanceStore.put_retrieval`).
    A member the result does not state is `None`.
    """
    ref = result.get("frame_ref")
    if not isinstance(ref, Mapping):
        ref = {}
    return {"spec_id": ref.get("spec_id"), "content_version": ref.get("content_version")}


def served_frame_ref(manifest: Mapping[str, Any]) -> dict[str, str | None]:
    """The manifest's own `{spec_id, content_version}`: its `definition.spec_id` and
    `content_version`, the pair dfo-apis mints and serves the frame under. A member
    the manifest does not state is `None`."""
    definition = manifest.get("definition")
    spec_id = definition.get("spec_id") if isinstance(definition, Mapping) else None
    return {"spec_id": spec_id, "content_version": manifest.get("content_version")}


def merged_manifest(lineage: Lineage) -> dict[str, Any]:
    """Fold a lineage into one manifest-shaped dict.

    A one-input lineage returns that input's manifest object itself, so a
    frame resting on one retrieval reads exactly as that retrieval's
    manifest. A longer lineage returns a new dict carrying the fields
    `harness.mcp_client` reads, combined so the result claims only what an
    input claims:

    - `citation.sources`: the inputs' sources, one per `ref` (`label` when a
      source has no `ref`), in lineage order. A source with neither is left
      out.
    - `citation.text`: the distinct citation texts, in lineage order, joined
      with "; ".
    - `artifacts`: one entry whose `created_at` is the oldest input date,
      since a derived result is only as fresh as its stalest input. Absent
      when any input has no readable `created_at`.
    - `coverage.window.end_year`: the earliest input end year, since a
      derived result cannot extend past the least-covered input. Absent when
      any input has no readable end year.
    - `trace.resolved_stat_vars`: the inputs' entries, one per `ref`. An entry
      with no `ref` is left out.
    - `warnings`: every input's warnings, each tagged with the `spec_id` of
      the retrieval it came from.
    """
    if len(lineage) == 1:
        return lineage[0].manifest
    citation_texts: list[str] = []
    sources: dict[str, dict[str, Any]] = {}
    stat_vars: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    created: list[date | None] = []
    end_years: list[int | None] = []
    for member in lineage:
        manifest = member.manifest
        citation = manifest.get("citation") or {}
        text = citation.get("text")
        if text and text not in citation_texts:
            citation_texts.append(str(text))
        for source in citation.get("sources") or []:
            source_key = source.get("ref") or source.get("label")
            if source_key:
                sources.setdefault(str(source_key), source)
        for stat_var in (manifest.get("trace") or {}).get("resolved_stat_vars") or []:
            if stat_var.get("ref"):
                stat_vars.setdefault(str(stat_var["ref"]), stat_var)
        for warning in manifest.get("warnings") or []:
            tagged = dict(warning)
            if member.spec_id is not None:
                tagged["spec_id"] = member.spec_id
            if tagged not in warnings:
                warnings.append(tagged)
        created.append(_iso_date(data_artifact(manifest.get("artifacts") or []).get("created_at")))
        window = (manifest.get("coverage") or {}).get("window") or {}
        end_years.append(_year(window.get("end_year")))
    merged: dict[str, Any] = {
        "citation": {"text": "; ".join(citation_texts), "sources": list(sources.values())},
        "trace": {"resolved_stat_vars": list(stat_vars.values())},
        "warnings": warnings,
    }
    known_dates = [when for when in created if when is not None]
    if len(known_dates) == len(created):
        merged["artifacts"] = [{"created_at": min(known_dates).isoformat()}]
    known_years = [year for year in end_years if year is not None]
    if len(known_years) == len(end_years):
        merged["coverage"] = {"window": {"end_year": min(known_years)}}
    return merged


async def manifest_with_stat_vars(manifest: dict[str, Any]) -> dict[str, Any]:
    """`manifest`, a `get_data` result, with the `trace.resolved_stat_vars` the
    result left out read back (`data_mcp.stat_var_list.with_stat_var_list`).

    A read that fails is logged and `manifest` kept as it is, since a trace
    that lists no refs leaves `_carried_stat_var_count` the count that never
    overstates.
    """
    try:
        return await with_stat_var_list(manifest)
    except ResourceError as exc:
        logger.warning("provenance: keeping a manifest without its StatVar list: %s", exc)
        return manifest


def summed_stat_var_count(manifest: dict[str, Any]) -> int | None:
    """The StatVars a summed retrieval added into its family total, from
    `get_data`'s `summed_stat_var_count`, or `None` when nothing was summed."""
    count = manifest.get("summed_stat_var_count")
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def _carried_stat_var_count(
    declared: int,
    carried: Sequence[ProvenanceInput],
    *,
    summed: Sequence[Sequence[ProvenanceInput]] = (),
) -> int:
    """The distinct StatVars a frame carries, a summed family's ref and a
    verb's own ref (a `sum_of`'s, a `quotient_of`'s, a `share_of`'s) each
    counted as the StatVars it stands for. "Sum" below reads as any of the
    three verbs.

    `declared` counts the frame's distinct refs, one per summed family and
    one per sum, and is the answer when no retrieval in `carried` is summed
    and `summed` is empty. A summed retrieval's contributors are the
    StatVars its trace lists with rows (`_refs_with_rows`), which dfo-apis
    counts as `summed_stat_var_count`. A sum's are the distinct StatVars of
    the retrievals under it, one of its `summed` entries, each named the
    same way (`_named_stat_vars`). The frame's other refs, `declared` less
    one per family and per sum, come from its unsummed retrievals. When
    every family and every sum names its members and the other refs number
    the refs with rows its unsummed retrievals list, the frame carries all
    of them, and the count is their union, so two families sharing members,
    a family beside one of its own members, or one StatVar summed across two
    places, count each StatVar once.

    Otherwise the members cannot be named, and the count never overstates:
    each family counts as its summed count, each sum as the largest count
    under it, and the other refs as one more part, added when every part's
    trace lists refs and no two parts share one, else the largest part
    alone.
    """
    families = [
        (member, count)
        for member in carried
        if (count := summed_stat_var_count(member.manifest)) is not None
    ]
    if not families and not summed:
        return declared
    unsummed = [member for member in carried if summed_stat_var_count(member.manifest) is None]
    others = max(0, declared - len({member.spec_id for member, _ in families}) - len(summed))
    other_refs = frozenset[str]().union(*(_refs_with_rows(member.manifest) for member in unsummed))
    named = [_named_stat_vars(member) for member, _ in families] + [
        _named_union(under) for under in summed
    ]
    if others == len(other_refs) and all(refs is not None for refs in named):
        return len(other_refs.union(*(refs for refs in named if refs is not None)))
    parts = [(count, _trace_refs(member.manifest)) for member, count in families]
    parts += [
        (
            max((summed_stat_var_count(member.manifest) or 1 for member in under), default=1),
            frozenset[str]().union(*(_trace_refs(member.manifest) for member in under)),
        )
        for under in summed
    ]
    if others:
        parts.append(
            (others, frozenset[str]().union(*(_trace_refs(member.manifest) for member in unsummed)))
        )
    listed = [refs for _, refs in parts]
    if all(listed) and len(frozenset[str]().union(*listed)) == sum(len(refs) for refs in listed):
        return sum(count for count, _ in parts)
    return max(count for count, _ in parts)


def _named_stat_vars(member: ProvenanceInput) -> frozenset[str] | None:
    """The StatVars a retrieval's frame carries, named from its trace: a
    summed family's contributors when they number its summed count, an
    unsummed retrieval's refs with rows when it lists any; else `None`."""
    refs = _refs_with_rows(member.manifest)
    count = summed_stat_var_count(member.manifest)
    if count is not None:
        return refs if len(refs) == count else None
    return refs or None


def _named_union(under: Sequence[ProvenanceInput]) -> frozenset[str] | None:
    """The distinct StatVars under a sum, or `None` when a retrieval under it
    cannot be named (`_named_stat_vars`)."""
    named = [_named_stat_vars(member) for member in under]
    if not named or any(refs is None for refs in named):
        return None
    return frozenset[str]().union(*(refs for refs in named if refs is not None))


def _trace_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The manifest's `trace.resolved_stat_vars` entries that carry a `ref`."""
    entries = (manifest.get("trace") or {}).get("resolved_stat_vars") or []
    return [entry for entry in entries if isinstance(entry, dict) and entry.get("ref")]


def _trace_refs(manifest: dict[str, Any]) -> frozenset[str]:
    """Every StatVar ref the manifest's trace lists."""
    return frozenset(str(entry["ref"]) for entry in _trace_entries(manifest))


def _refs_with_rows(manifest: dict[str, Any]) -> frozenset[str]:
    """The refs the manifest's trace lists with rows in the frame.

    dfo-apis counts each member's rows after filtering as
    `observation_count`, and sums exactly the members with rows. An entry
    stating no count is kept, since it may have rows.
    """
    kept: set[str] = set()
    for entry in _trace_entries(manifest):
        rows = entry.get("observation_count")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows > 0:
            kept.add(str(entry["ref"]))
    return frozenset(kept)


def data_artifact(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """The first entry in `artifacts` that is not the statistics artifact, or `{}`.

    A manifest with `series_count` not `None` carries one `rendering: {"statistics":
    "series"}` entry (`data_mcp.series_statistics.find_statistics_artifact`'s own match)
    alongside its data artifact, in no wire-guaranteed order -- reading `artifacts[0]`
    blindly for a `created_at` would read the statistics artifact's own, unrelated to the
    data itself, whenever it happens to sort first. `harness.mcp_client`'s single-manifest
    freshness reading uses this too.
    """
    return next((a for a in artifacts if "statistics" not in (a.get("rendering") or {})), {})


def _iso_date(created_at: object) -> date | None:
    """The date part of an ISO-8601 timestamp, or `None` when there is none to read."""
    if not created_at:
        return None
    try:
        return date.fromisoformat(str(created_at).split("T", maxsplit=1)[0])
    except ValueError:
        return None


def _year(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None


__all__ = [
    "Lineage",
    "ProvenanceInput",
    "ProvenanceStore",
    "merged_manifest",
    "summed_stat_var_count",
]
