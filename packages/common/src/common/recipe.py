"""A derived frame's recipe: the tree of op-specs that computed it, down to the
root datasets, and the two hashes apis indexes it by.

`build_recipe` walks the materialisation store from a derived frame's pair. Each
derived frame's stored trace carries the canonical op-spec that produced it
(`execution_record.content`, the RFC 8785 text `analysis_mcp.identity
.canonicalize_op_spec` wrote), and the op-spec names its input frames. An input
with an `ops_` id is derived and is followed to its own trace. An input with a
`spec_` id is a root, described by its mint record
(`common.materialisation.MintRecord`). `common` does not import `analysis_mcp`,
so a trace is read by attribute and an op-spec as the parsed JSON object.

The recipe is the JSON object the apis service's publish door stores:

- `tree`: a node per frame. A derived node is `{"frame", "op_spec", "inputs"}`,
  where `op_spec` is the parsed canonical op-spec and `inputs` maps each of its
  frame slots to the node of the frame in that slot. The slots are `frame_ref`,
  and every parameter whose name ends `_frame_ref` and is set, keyed by that
  name. A root node is `{"frame", "root"}`, with `root` an index into `roots`.
- `roots`: each root's mint record, once per root frame, in the order a
  depth-first walk of `tree` first reaches them (slots in the order above,
  parameters by name).
- `normal_form_hash`, `replay_key`, `execution_identity`, and `anchor`.

The normal form (`normal_form`) is the recipe with everything a replay on newer
data changes taken out. A derived node becomes its verb, its parameters and its
inputs' normal forms, so no frame id appears. A root becomes its dataset's
`spec_id` and the mint `options` that select its rendering, without its
`content_version`. A literal a rolling overlay governs is dropped, since a replay
rewrites it: on an op-spec whose `rolling` parameter is an object, each member
that is a rolling reference (an object with an `anchor`) names the sibling
parameter it governs, and a predicate whose `rolling` is set has its `value`
governed. The overlay itself stays. Two frames computed by one recipe from
different versions of the same datasets therefore share a normal form.

`normal_form_hash` is `rnf_` and 32 hex digits of the SHA-256 of the RFC 8785
form of `{"identity_version": "rnf1", "recipe": <normal form>}`.

`replay_key` is `rpk_` and 32 hex digits of the SHA-256 of the RFC 8785 form of
`{"identity_version": "rpk1", "normal_form_hash", "roots", "execution_identities"}`.
`roots` holds each root's identity (`root_identity`) in `roots` order: its apis
pair, and the enrichment stamp its frame was named under when it has one. A root
with no stamp is identified by its pair alone. `execution_identities` holds every
distinct `ExecutionIdentity.canonical_string()` in the tree, sorted. A replay
computes the key before it runs: the hash from the stored recipe, the roots'
current identities, and the one identity of the engine replaying. A frame whose
ancestors were computed by different engine builds lists
each of them, so its key is not the key of a replay under any single build.

`execution_identity` is the identity of the frame the recipe is for.

`anchor` is null for a tree with no rolling op-spec. For a tree with one it is
`{"period", "rule", "left_out"}`, read from the anchor record
(`common.materialisation.AnchorRecord`) the rolling calls were checked against:
the latest period, the rule that chose it, and each series that holds no value
at it as `{"root", "coordinates"}`, `root` indexing the populations the rolling
op-specs carry. `recipe_anchor` finds the record by recomputing its key from
those populations and the root frames the tree's rolling periods are taken over
(`common.rolling.rolling_roots`), which are the roots under its rolling op-specs. A
dataset joined beside a rolling step is no part of the key.

A frame can be part of a wider plan than its own tree: its rolling op-specs then
carry populations naming a dataset none of its rolling periods reads, and the
period was taken over that dataset too. Such a frame has no anchor of its own.
`build_recipe` raises `RecipeAnchorError` with reason
`populations_outside_reach` and the rest of the recipe on the error, so a caller
can replay it over the populations of its own tree
(`common.rolling.anchor_populations`), which gives a frame with an anchor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import rfc8785
from dfo_frame_grammar import FrameError, RollingError, shift_period

from common.materialisation import (
    AnchorRecordIntegrityError,
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    MaterialisationStore,
    MintRecord,
    MintRecordUnrecoverableError,
    StoredFrameTrace,
)
from common.rolling import anchor_key, rolling_roots

__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "AncestryReads",
    "RecipeAncestorMissingError",
    "RecipeAnchorError",
    "RecipeError",
    "RecipeRootUnrecoverableError",
    "RecipeTraceUnusableError",
    "RecoverMintRecord",
    "StateRootOptions",
    "build_recipe",
    "normal_form",
    "normal_form_hash",
    "reached_roots",
    "recipe_anchor",
    "replay_key",
    "root_identity",
    "root_mint_record",
    "stored_trace",
]

RECIPE_SCHEMA_VERSION = 1

NORMAL_FORM_IDENTITY_VERSION = "rnf1"
NORMAL_FORM_HASH_PREFIX = "rnf_"
REPLAY_KEY_IDENTITY_VERSION = "rpk1"
REPLAY_KEY_PREFIX = "rpk_"

_DIGEST_HEX_LENGTH = 32
_ROOT_SPEC_ID_PREFIX = "spec_"
_OP_SPEC_HASH_PREFIX = "ops_"
_PRIMARY_SLOT = "frame_ref"
_SLOT_SUFFIX = "_frame_ref"
_ROLLING = "rolling"
_OUTSIDE_REACH = "populations_outside_reach"

# Rebuilds the mint record of a root the store holds none for, or raises
# `MintRecordUnrecoverableError`. `data_mcp.mint_recovery.recover_mint_record` with its
# dataset-versions reader bound is one.
RecoverMintRecord = Callable[[FrameRef], Awaitable[MintRecord]]

# The `options` a recipe states for a root, given the root's mint record.
# `harness.recipe_replay` states a replayed root's stored options this way.
StateRootOptions = Callable[[MintRecord], Mapping[str, Any]]


class RecipeError(Exception):
    """No recipe can be built. `ref` is the frame the walk stopped at."""

    def __init__(self, ref: FrameRef, message: str) -> None:
        self.ref = ref
        super().__init__(message)


class RecipeAncestorMissingError(RecipeError):
    """The store holds no trace for the derived frame `ref`, which is the frame the
    recipe was asked for or one of its ancestors."""

    def __init__(self, ref: FrameRef) -> None:
        super().__init__(
            ref, f"no trace is held for the derived frame {ref.spec_id}, {ref.content_version}"
        )


class RecipeTraceUnusableError(RecipeError):
    """The store holds something under the derived frame `ref` that no op-spec can
    be read from: the object fails a store check, its trace carries no canonical
    op-spec, or the op-spec does not hash to `ref.spec_id`."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.reason = reason
        super().__init__(
            ref,
            f"the trace of the derived frame {ref.spec_id}, {ref.content_version} "
            f"is unusable: {reason}",
        )


class RecipeRootUnrecoverableError(RecipeError):
    """The root frame `ref` has no mint record and none could be rebuilt. `reason` is
    `no_mint_record` when no recovery routine was given, else the routine's own."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.reason = reason
        super().__init__(
            ref,
            f"no mint record for the root frame {ref.spec_id}, {ref.content_version}: {reason}",
        )


class RecipeAnchorError(RecipeError):
    """The tree of the derived frame `ref` holds a rolling op-spec and its anchor
    cannot be stated. `reason` is `populations_differ` or `rules_differ` when two
    rolling op-specs of the tree disagree, `populations_outside_reach` when their
    populations name a dataset no rolling period of the tree reads,
    `no_anchor_record` when the store holds no record under the key the tree
    gives, `anchor_record_unreadable` when it holds one that fails its check, and
    `period_differs` when a rolling literal is not the record's period at its
    offset.

    `recipe` is set by `build_recipe` for `populations_outside_reach` alone: the
    recipe it would have returned, with a null `anchor`."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.reason = reason
        self.recipe: dict[str, Any] | None = None
        super().__init__(
            ref,
            f"no anchor for the rolling recipe of {ref.spec_id}, {ref.content_version}: {reason}",
        )


def _digest(preimage: Mapping[str, Any], *, prefix: str) -> str:
    return f"{prefix}{hashlib.sha256(rfc8785.dumps(preimage)).hexdigest()[:_DIGEST_HEX_LENGTH]}"


def _pair(ref: FrameRef) -> dict[str, str]:
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _input_slots(op_spec: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Each frame slot `op_spec` fills, with the pair in it: `frame_ref`, then every
    set parameter named `*_frame_ref`, by name."""
    parameters = op_spec["parameters"]
    return [
        (_PRIMARY_SLOT, op_spec[_PRIMARY_SLOT]),
        *(
            (name, parameters[name])
            for name in sorted(parameters)
            if name.endswith(_SLOT_SUFFIX) and parameters[name] is not None
        ),
    ]


def _is_rolling_ref(value: object) -> bool:
    return isinstance(value, Mapping) and "anchor" in value


def _normal_predicates(predicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`predicates` without the `value` of each one a rolling reference governs, in
    the order of their RFC 8785 bytes, which is the order an op-spec keeps its
    predicates in before any literal is dropped."""
    normal = [
        {
            key: value
            for key, value in predicate.items()
            if not (key == "value" and predicate.get(_ROLLING) is not None)
        }
        for predicate in predicates
    ]
    return sorted(normal, key=rfc8785.dumps)


def _normal_parameters(parameters: Mapping[str, Any], *, slots: Iterable[str]) -> dict[str, Any]:
    """`parameters` without its filled frame slots and without every literal a
    rolling overlay governs."""
    overlay = parameters.get(_ROLLING)
    governed = (
        {name for name, value in overlay.items() if _is_rolling_ref(value)}
        if isinstance(overlay, Mapping)
        else set()
    )
    dropped = governed | set(slots)
    normal = {name: value for name, value in parameters.items() if name not in dropped}
    if isinstance(normal.get("predicates"), list):
        normal["predicates"] = _normal_predicates(normal["predicates"])
    return normal


def normal_form(tree: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The normal form of a recipe's `tree` over its `roots`. See the module docstring."""
    if "root" in tree:
        root = roots[tree["root"]]
        return {"spec_id": root["dataset"]["spec_id"], "rendering": root["options"]}
    op_spec = tree["op_spec"]
    inputs = tree["inputs"]
    return {
        "identity_version": op_spec["identity_version"],
        "verb": op_spec["verb"],
        "parameters": _normal_parameters(op_spec["parameters"], slots=inputs),
        "inputs": {slot: normal_form(node, roots) for slot, node in inputs.items()},
    }


def normal_form_hash(normal: Mapping[str, Any]) -> str:
    """`rnf_` and 32 hex digits naming the normal form `normal`."""
    return _digest(
        {"identity_version": NORMAL_FORM_IDENTITY_VERSION, "recipe": normal},
        prefix=NORMAL_FORM_HASH_PREFIX,
    )


def root_identity(dataset: Mapping[str, str], *, enrichment_stamp: str | None) -> dict[str, str]:
    """A root's identity: the apis `{spec_id, content_version}` its frame was
    minted from, and the enrichment stamp the frame was named under
    (`MintRecord.enrichment_stamp`) when it has one. apis keeps a version's
    `content_version` across a change of party kinds, so the stamp is what tells two
    frames of one version apart (`data_mcp.frame_materialisation.frame_ref_for`).
    Without a stamp the identity is the pair alone."""
    identity = {"spec_id": dataset["spec_id"], "content_version": dataset["content_version"]}
    if enrichment_stamp is not None:
        identity["enrichment_stamp"] = enrichment_stamp
    return identity


def replay_key(
    *,
    normal_form_hash: str,
    root_datasets: Sequence[Mapping[str, str]],
    execution_identities: Iterable[str],
) -> str:
    """`rpk_` and 32 hex digits naming one replay of a recipe.

    Args:
        normal_form_hash: The recipe's `normal_form_hash`.
        root_datasets: Each root's identity (`root_identity`), in the recipe's
            `roots` order. A bare `{spec_id, content_version}` is a root with no stamp.
        execution_identities: The `ExecutionIdentity.canonical_string()` of every
            frame in the tree. Order and repeats do not matter.
    """
    return _digest(
        {
            "identity_version": REPLAY_KEY_IDENTITY_VERSION,
            "normal_form_hash": normal_form_hash,
            "roots": [
                root_identity(root, enrichment_stamp=root.get("enrichment_stamp"))
                for root in root_datasets
            ],
            "execution_identities": sorted(set(execution_identities)),
        },
        prefix=REPLAY_KEY_PREFIX,
    )


def _rolling_literals(parameters: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Each literal period of `parameters` a rolling reference governs, with the
    reference: a window bound named in `rolling`, and a rolling predicate's `value`."""
    overlay = parameters.get(_ROLLING)
    if isinstance(overlay, Mapping):
        for name, ref in overlay.items():
            if _is_rolling_ref(ref):
                yield parameters[name], ref
    for predicate in parameters.get("predicates", ()):
        if _is_rolling_ref(predicate.get(_ROLLING)):
            yield predicate["value"], predicate[_ROLLING]


def _derived_nodes(tree: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    if "root" in tree:
        return
    yield tree
    for node in tree["inputs"].values():
        yield from _derived_nodes(node)


async def recipe_anchor(
    tree: Mapping[str, Any], store: MaterialisationStore
) -> dict[str, Any] | None:
    """The recipe's `anchor`: the period its rolling references resolved to and the
    series that period left out, or `None` for a tree with no rolling op-spec.
    This is the one place a recipe's anchor is decided.

    Args:
        tree: The recipe's tree.
        store: The store holding the anchor record the rolling calls were checked
            against.

    Raises:
        RecipeAnchorError: The tree is rolling and no anchor can be stated for it.
    """
    ref = FrameRef(**tree["frame"])
    populations: list[Any] = []
    literals: list[tuple[str, Mapping[str, Any]]] = []
    for node in _derived_nodes(tree):
        parameters = node["op_spec"]["parameters"]
        overlay = parameters.get(_ROLLING)
        if not isinstance(overlay, Mapping):
            continue
        if overlay["population"] not in populations:
            populations.append(overlay["population"])
        literals.extend(_rolling_literals(parameters))
    if not populations:
        return None
    if len(populations) > 1:
        raise RecipeAnchorError(ref, reason="populations_differ")
    rules = {reference["rule"] for _, reference in literals}
    if len(rules) != 1:
        raise RecipeAnchorError(ref, reason="rules_differ")
    (rule,) = rules

    reached = [node["frame"] for node in rolling_roots(tree)]
    if {population["spec_id"] for population in populations[0]} - {
        frame["spec_id"] for frame in reached
    }:
        raise RecipeAnchorError(ref, reason=_OUTSIDE_REACH)
    key = anchor_key(populations=populations[0], roots=reached, rule=rule)
    try:
        record = await store.get_anchor_record(key)
    except AnchorRecordIntegrityError as exc:
        raise RecipeAnchorError(ref, reason="anchor_record_unreadable") from exc
    if record is None:
        raise RecipeAnchorError(ref, reason="no_anchor_record")
    for literal, reference in literals:
        try:
            implied = shift_period(literal, -int(reference["offset"]))
        except (RollingError, FrameError) as exc:
            raise RecipeAnchorError(ref, reason="period_differs") from exc
        if implied != record.period:
            raise RecipeAnchorError(ref, reason="period_differs")
    return {
        "period": record.period,
        "rule": record.rule,
        "left_out": [series.model_dump(mode="json") for series in record.left_out],
    }


class AncestryReads:
    """The stored traces of derived frames and the mint records of roots that
    walks of the store read, each read once by its pair and kept for later
    walks. `build_recipe` and `common.steps.assemble_steps` given one
    `AncestryReads` read each ancestor's trace and each root's mint record once
    between them. Only a read that succeeds is kept.

    Args:
        store: The store the traces and mint records are read from.
        recover: Rebuilds the mint record of a root that has none. Without it
            such a root fails the read.
    """

    def __init__(
        self, *, store: MaterialisationStore, recover: RecoverMintRecord | None = None
    ) -> None:
        self._store = store
        self._recover = recover
        self._traces: dict[tuple[str, str], StoredFrameTrace] = {}
        self._mint_records: dict[tuple[str, str], MintRecord] = {}

    async def trace(self, ref: FrameRef) -> StoredFrameTrace:
        """`stored_trace` of the derived frame `ref`."""
        key = (ref.spec_id, ref.content_version)
        if key not in self._traces:
            self._traces[key] = await stored_trace(ref, store=self._store)
        return self._traces[key]

    async def mint_record(self, ref: FrameRef) -> MintRecord:
        """`root_mint_record` of the root frame `ref`."""
        key = (ref.spec_id, ref.content_version)
        if key not in self._mint_records:
            self._mint_records[key] = await root_mint_record(
                ref, store=self._store, recover=self._recover
            )
        return self._mint_records[key]


class _Walk:
    """One walk of the store from a derived frame to its roots."""

    def __init__(self, *, reads: AncestryReads) -> None:
        self._reads = reads
        self._nodes: dict[tuple[str, str], dict[str, Any]] = {}
        self.roots: list[MintRecord] = []
        # Each derived frame's `ExecutionIdentity.canonical_string()`, by pair.
        self.execution_identities: dict[tuple[str, str], str] = {}

    async def node(self, ref: FrameRef, *, trace: StoredFrameTrace | None = None) -> dict[str, Any]:
        """The tree node for `ref`, built once however many slots name it. `trace` is
        `ref`'s own, for a caller that has already read it."""
        key = (ref.spec_id, ref.content_version)
        if key not in self._nodes:
            self._nodes[key] = (
                await self._root(ref)
                if ref.spec_id.startswith(_ROOT_SPEC_ID_PREFIX)
                else await self._derived(ref, trace=trace)
            )
        return self._nodes[key]

    async def _root(self, ref: FrameRef) -> dict[str, Any]:
        record = await self._reads.mint_record(ref)
        self.roots.append(record)
        return {"frame": _pair(ref), "root": len(self.roots) - 1}

    async def _derived(self, ref: FrameRef, *, trace: StoredFrameTrace | None) -> dict[str, Any]:
        if trace is None:
            trace = await self._reads.trace(ref)
        op_spec = _op_spec_of(ref, trace)
        self.execution_identities[(ref.spec_id, ref.content_version)] = _execution_identity_of(
            ref, trace
        )
        inputs: dict[str, Any] = {}
        for slot, pair in _input_slots(op_spec):
            inputs[slot] = await self.node(_slot_ref(ref, slot, pair))
        return {"frame": _pair(ref), "op_spec": op_spec, "inputs": inputs}


async def root_mint_record(
    ref: FrameRef, *, store: MaterialisationStore, recover: RecoverMintRecord | None
) -> MintRecord:
    """The mint record of the root frame `ref`: the store's, else the one
    `recover` rebuilds.

    Raises:
        RecipeRootUnrecoverableError: The store's record fails its check, or
            the store holds none and none was rebuilt.
    """
    try:
        record = await store.get_mint_record(ref)
    except FrameIntegrityError as exc:
        raise RecipeRootUnrecoverableError(ref, reason=exc.reason) from exc
    if record is not None:
        return record
    if recover is None:
        raise RecipeRootUnrecoverableError(ref, reason="no_mint_record")
    try:
        return await recover(ref)
    except MintRecordUnrecoverableError as exc:
        raise RecipeRootUnrecoverableError(ref, reason=exc.reason) from exc


async def stored_trace(ref: FrameRef, *, store: MaterialisationStore) -> StoredFrameTrace:
    """The trace the store holds for the derived frame `ref`.

    Raises:
        RecipeAncestorMissingError: The store holds none, or it has expired.
        RecipeTraceUnusableError: The stored object fails a store check.
    """
    try:
        trace = await store.get_trace(ref)
    except FrameExpired as exc:
        raise RecipeAncestorMissingError(ref) from exc
    except (FrameIntegrityError, FrameTooLarge) as exc:
        raise RecipeTraceUnusableError(ref, reason=exc.reason) from exc
    if trace is None:
        raise RecipeAncestorMissingError(ref)
    return trace


def _slot_ref(ref: FrameRef, slot: str, pair: Mapping[str, Any]) -> FrameRef:
    """The frame reference in `slot` of the op-spec of `ref`."""
    try:
        return FrameRef(**pair)
    except (TypeError, ValueError) as exc:
        raise RecipeTraceUnusableError(
            ref, reason=f"its op-spec's {slot} is no frame reference"
        ) from exc


def _op_spec_of(ref: FrameRef, trace: StoredFrameTrace) -> dict[str, Any]:
    """The parsed canonical op-spec `trace` carries, checked against `ref.spec_id`."""
    content = getattr(getattr(trace, "execution_record", None), "content", None)
    if not isinstance(content, str):
        raise RecipeTraceUnusableError(ref, reason="it carries no canonical op-spec")
    digest = hashlib.sha256(content.encode()).hexdigest()[:_DIGEST_HEX_LENGTH]
    if f"{_OP_SPEC_HASH_PREFIX}{digest}" != ref.spec_id:
        raise RecipeTraceUnusableError(
            ref, reason="its op-spec does not hash to the frame's spec_id"
        )
    op_spec = json.loads(content)
    if not isinstance(op_spec.get("parameters"), dict) or _PRIMARY_SLOT not in op_spec:
        raise RecipeTraceUnusableError(ref, reason="its op-spec names no frame_ref or parameters")
    return op_spec


def _execution_identity_of(ref: FrameRef, trace: StoredFrameTrace) -> str:
    identity = getattr(trace, "execution_identity", None)
    canonical_string = getattr(identity, "canonical_string", None)
    if canonical_string is None:
        raise RecipeTraceUnusableError(ref, reason="it carries no execution identity")
    return canonical_string()


async def reached_roots(
    ref: FrameRef, *, store: MaterialisationStore, trace: StoredFrameTrace | None = None
) -> list[FrameRef]:
    """Every root frame the frame held under `ref` was computed from, each once.

    A root frame is its own only root. A derived frame's are found by following
    its stored op-spec and each derived input's, so the walk reads one trace for
    each distinct derived frame (`MaterialisationStore.get_trace`, which decodes
    no row) and nothing for a root. `trace` is `ref`'s own, for a caller that has
    already read it.

    Raises:
        RecipeAncestorMissingError: The store holds no trace for `ref` or for a
            derived ancestor.
        RecipeTraceUnusableError: A trace on the way carries no usable op-spec.
    """
    found: dict[tuple[str, str], FrameRef] = {}
    seen: set[tuple[str, str]] = set()
    pending: list[tuple[FrameRef, StoredFrameTrace | None]] = [(ref, trace)]
    while pending:
        current, held = pending.pop()
        key = (current.spec_id, current.content_version)
        if key in seen:
            continue
        seen.add(key)
        if current.spec_id.startswith(_ROOT_SPEC_ID_PREFIX):
            found[key] = current
            continue
        if held is None:
            held = await stored_trace(current, store=store)
        for slot, pair in _input_slots(_op_spec_of(current, held)):
            pending.append((_slot_ref(current, slot, pair), None))
    return [found[key] for key in sorted(found)]


async def build_recipe(
    ref: FrameRef,
    *,
    store: MaterialisationStore,
    trace: StoredFrameTrace | None = None,
    recover: RecoverMintRecord | None = None,
    root_options: StateRootOptions | None = None,
    reads: AncestryReads | None = None,
) -> dict[str, Any]:
    """The recipe of the derived frame held under `ref`. See the module docstring.

    Args:
        ref: A derived frame's `ops_`/`ocv_` pair.
        store: The materialisation store holding the frame's ancestors' traces and
            its roots' mint records. Only traces and mint records are read; no
            frame's rows are decoded.
        trace: `ref`'s own stored trace, when the caller has already read it.
        recover: Rebuilds the mint record of a root that has none. Without it such
            a root fails the build.
        root_options: The `options` to state for each root, in place of its mint
            record's. They enter the root's entry in `roots` and the normal form.
        reads: The reads of `store` to share with another walk of the same
            ancestry, such as `common.steps.assemble_steps`. Its own `recover`
            stands in for `recover`. Without it the build reads afresh.

    Raises:
        RecipeAncestorMissingError: The store holds no trace for `ref` or for a derived
            ancestor. `ref` on the error is the missing pair.
        RecipeTraceUnusableError: A trace on the way carries no usable op-spec.
        RecipeRootUnrecoverableError: A root has no mint record and none was rebuilt.
        RecipeAnchorError: The tree is rolling and no anchor can be stated for it.
        ValueError: `ref` is a root frame's pair, which has no recipe.
    """
    if ref.spec_id.startswith(_ROOT_SPEC_ID_PREFIX):
        raise ValueError(f"{ref.spec_id} names a root frame, which has no recipe")
    walk = _Walk(reads=reads or AncestryReads(store=store, recover=recover))
    tree = await walk.node(ref, trace=trace)
    roots = [record.model_dump(mode="json") for record in walk.roots]
    if root_options is not None:
        for root, record in zip(roots, walk.roots, strict=True):
            root["options"] = dict(root_options(record))
    hashed = normal_form_hash(normal_form(tree, roots))
    recipe: dict[str, Any] = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "normal_form_hash": hashed,
        "replay_key": replay_key(
            normal_form_hash=hashed,
            root_datasets=[
                root_identity(record.dataset.model_dump(), enrichment_stamp=record.enrichment_stamp)
                for record in walk.roots
            ],
            execution_identities=walk.execution_identities.values(),
        ),
        "execution_identity": walk.execution_identities[(ref.spec_id, ref.content_version)],
        "tree": tree,
        "roots": roots,
        "anchor": None,
    }
    try:
        recipe["anchor"] = await recipe_anchor(tree, store)
    except RecipeAnchorError as exc:
        if exc.reason == _OUTSIDE_REACH:
            exc.recipe = recipe
        raise
    return recipe
