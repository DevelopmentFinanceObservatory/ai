"""Tests for `common.recipe`: the tree, the roots, the normal form and the two hashes.

Traces here are plain stand-ins carrying the attributes `build_recipe` reads off
`analysis_mcp.trace.Trace`, since `common` does not depend on `analysis_mcp`.
`services/analysis-mcp/tests/test_recipe_from_trace.py` builds a recipe from real
traces.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest
import rfc8785
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MintRecord,
    MintRecordUnrecoverableError,
)
from common.recipe import (
    RecipeAncestorMissingError,
    RecipeRootUnrecoverableError,
    RecipeTraceUnusableError,
    build_recipe,
    normal_form,
    normal_form_hash,
    replay_key,
)

_IDENTITY = "0.3.0|1.0.0|0.2.0"
_OPTIONS: dict[str, Any] = {"format": "parquet", "bundle": False, "layout": "frame"}
_TYPED_OPTIONS = _OPTIONS | {"columns": ["observationAbout_type"]}
_KIND_OPTIONS = _OPTIONS | {"columns": ["party_kind"]}


@dataclass(frozen=True)
class _Identity:
    value: str

    def canonical_string(self) -> str:
        return self.value


@dataclass(frozen=True)
class _ExecutionRecord:
    content: str


@dataclass(frozen=True)
class _Trace:
    execution_record: _ExecutionRecord
    execution_identity: _Identity
    frame_content_version: str = "cv_" + "0" * 32
    upstream_trace: dict[str, Any] = field(default_factory=dict)


def _hex(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


async def _root(
    store: InMemoryMaterialisationStore,
    *,
    dataset: str = "a",
    version: str = "1",
    options: dict[str, Any] = _OPTIONS,
    recorded: bool = True,
    enrichment_stamp: str | None = None,
) -> FrameRef:
    """A root frame's pair, with its mint record in `store` unless `recorded` is off.
    A column selection gives the frame a `content_version` other than its dataset's,
    and an enrichment stamp one other again."""
    spec_id = "spec_" + dataset * 32
    dataset_ref = FrameRef(spec_id=spec_id, content_version="cv_" + version * 32)
    named = (
        (version, "columns") if enrichment_stamp is None else (version, "columns", enrichment_stamp)
    )
    ref = (
        FrameRef(spec_id=spec_id, content_version=f"cv_{_hex(*named)}")
        if "columns" in options
        else dataset_ref
    )
    if recorded:
        await store.put_mint_record(
            MintRecord(
                frame=ref, dataset=dataset_ref, options=options, enrichment_stamp=enrichment_stamp
            )
        )
    return ref


def _op_spec(
    verb: str, parameters: dict[str, Any], inputs: dict[str, FrameRef]
) -> tuple[str, dict[str, Any]]:
    """An op-spec in `canonicalize_op_spec`'s shape, as canonical text and as an object."""
    frame_refs = {slot: ref.model_dump() for slot, ref in inputs.items()}
    op_spec = {
        "identity_version": "ops1",
        "verb": verb,
        "frame_ref": frame_refs.pop("frame_ref"),
        "parameters": parameters | frame_refs,
    }
    return rfc8785.dumps(op_spec).decode(), op_spec


async def _derive(
    store: InMemoryMaterialisationStore,
    verb: str,
    parameters: dict[str, Any],
    *,
    identity: str = _IDENTITY,
    **inputs: FrameRef,
) -> FrameRef:
    """Stores the trace of `verb` over `inputs` and returns the derived frame's pair,
    whose ids hash what the toolkit's do: the op-spec, and its inputs and identity."""
    content, _ = _op_spec(verb, parameters, inputs)
    ref = FrameRef(
        spec_id=f"ops_{hashlib.sha256(content.encode()).hexdigest()[:32]}",
        content_version=f"ocv_{_hex(content, identity)}",
    )
    trace = _Trace(_ExecutionRecord(content), _Identity(identity))
    await store.put(ref, frame=object(), trace=trace, writer="test")
    return ref


async def _ranked_share(
    store: InMemoryMaterialisationStore,
    *,
    version: str = "1",
    limit: int = 5,
    options: dict[str, Any] = _OPTIONS,
) -> FrameRef:
    """`rank(share_of(filter(a), b))`: three derived frames over two roots."""
    numerator = await _root(store, dataset="a", version=version, options=options)
    denominator = await _root(store, dataset="b", version=version)
    filtered = await _derive(
        store,
        "filter",
        {"predicates": [{"kind": "category_match", "column": "sector", "values": ["health"]}]},
        frame_ref=numerator,
    )
    share = await _derive(
        store,
        "share_of",
        {"numerator_by": []},
        frame_ref=filtered,
        denominator_frame_ref=denominator,
    )
    return await _derive(store, "rank", {"limit": limit, "rank_by": ["recipient"]}, frame_ref=share)


async def test_a_recipe_holds_the_tree_down_to_its_roots_and_their_mint_records() -> None:
    store = InMemoryMaterialisationStore()
    numerator = await _root(store, dataset="a")
    denominator = await _root(store, dataset="b")
    filter_parameters = {"predicates": [{"kind": "category_match", "column": "c", "values": ["x"]}]}
    filtered = await _derive(store, "filter", filter_parameters, frame_ref=numerator)
    share = await _derive(
        store,
        "share_of",
        {"numerator_by": []},
        frame_ref=filtered,
        denominator_frame_ref=denominator,
    )

    recipe = await build_recipe(share, store=store)

    assert recipe["tree"] == {
        "frame": share.model_dump(),
        "op_spec": _op_spec(
            "share_of",
            {"numerator_by": []},
            {"frame_ref": filtered, "denominator_frame_ref": denominator},
        )[1],
        "inputs": {
            "frame_ref": {
                "frame": filtered.model_dump(),
                "op_spec": _op_spec("filter", filter_parameters, {"frame_ref": numerator})[1],
                "inputs": {"frame_ref": {"frame": numerator.model_dump(), "root": 0}},
            },
            "denominator_frame_ref": {"frame": denominator.model_dump(), "root": 1},
        },
    }
    assert recipe["roots"] == [
        {
            "schema_version": 1,
            "frame": root.model_dump(),
            "dataset": root.model_dump(),
            "options": _OPTIONS,
            "scope": {},
            "stat_var_count": None,
        }
        for root in (numerator, denominator)
    ]
    assert recipe["schema_version"] == 1
    assert recipe["execution_identity"] == _IDENTITY
    assert recipe["anchor"] is None
    assert re.fullmatch(r"rnf_[0-9a-f]{32}", recipe["normal_form_hash"])
    assert re.fullmatch(r"rpk_[0-9a-f]{32}", recipe["replay_key"])
    assert set(recipe) == {
        "schema_version",
        "normal_form_hash",
        "replay_key",
        "execution_identity",
        "tree",
        "roots",
        "anchor",
    }
    json.dumps(recipe)  # the publish body carries it as JSON


async def test_one_recipe_on_two_root_versions_shares_a_normal_form_and_not_a_replay_key() -> None:
    store = InMemoryMaterialisationStore()
    first = await build_recipe(await _ranked_share(store, version="1"), store=store)
    second = await build_recipe(await _ranked_share(store, version="2"), store=store)

    assert first["tree"]["frame"] != second["tree"]["frame"]
    assert first["normal_form_hash"] == second["normal_form_hash"]
    assert first["replay_key"] != second["replay_key"]


async def test_a_roots_enrichment_stamp_enters_the_replay_key_and_no_stamp_keeps_the_old_key() -> (
    None
):
    keys: dict[str | None, str] = {}
    normal_forms: set[str] = set()
    for stamp in ("a" * 64, "b" * 64, None):
        store = InMemoryMaterialisationStore()
        root = await _root(store, options=_KIND_OPTIONS, enrichment_stamp=stamp)
        summed = await _derive(
            store, "aggregate", {"aggregation": "sum", "group_by": ["time_period"]}, frame_ref=root
        )
        recipe = await build_recipe(summed, store=store)
        keys[stamp] = recipe["replay_key"]
        normal_forms.add(recipe["normal_form_hash"])

    assert len(normal_forms) == 1
    assert keys["a" * 64] != keys["b" * 64]
    # A root with no stamp keys on its pair alone.
    pair_key = {
        "execution_identities": [_IDENTITY],
        "identity_version": "rpk1",
        "normal_form_hash": normal_forms.pop(),
        "roots": [{"content_version": "cv_" + "1" * 32, "spec_id": "spec_" + "a" * 32}],
    }
    assert keys[None] == f"rpk_{hashlib.sha256(rfc8785.dumps(pair_key)).hexdigest()[:32]}"


@pytest.mark.parametrize(
    "changed",
    [{"limit": 10}, {"options": _TYPED_OPTIONS}],
    ids=["a-parameter", "a-roots-column-rendering"],
)
async def test_a_different_recipe_differs_in_both_hashes(changed: dict[str, Any]) -> None:
    store = InMemoryMaterialisationStore()
    first = await build_recipe(await _ranked_share(store), store=store)
    second = await build_recipe(await _ranked_share(store, **changed), store=store)

    assert first["normal_form_hash"] != second["normal_form_hash"]
    assert first["replay_key"] != second["replay_key"]


async def test_a_root_two_slots_reach_is_listed_once() -> None:
    store = InMemoryMaterialisationStore()
    root = await _root(store)
    left = await _derive(store, "filter", {"predicates": []}, frame_ref=root)
    joined = await _derive(
        store, "join", {"join_keys": ["a"]}, frame_ref=left, right_frame_ref=root
    )

    recipe = await build_recipe(joined, store=store)

    assert len(recipe["roots"]) == 1
    assert recipe["tree"]["inputs"]["right_frame_ref"] == {"frame": root.model_dump(), "root": 0}
    assert recipe["tree"]["inputs"]["frame_ref"]["inputs"]["frame_ref"]["root"] == 0


async def test_a_missing_ancestor_trace_fails_naming_the_missing_pair() -> None:
    built = InMemoryMaterialisationStore()
    root = await _root(built)
    middle = await _derive(built, "filter", {"predicates": []}, frame_ref=root)
    leaf = await _derive(built, "rank", {"rank_by": ["recipient"]}, frame_ref=middle)
    held = InMemoryMaterialisationStore()
    await _root(held)
    leaf_trace = await built.get_trace(leaf)
    assert leaf_trace is not None
    await held.put(leaf, frame=object(), trace=leaf_trace, writer="test")

    with pytest.raises(RecipeAncestorMissingError) as raised:
        await build_recipe(leaf, store=held)

    assert raised.value.ref == middle


async def test_the_frames_own_trace_can_be_passed_in_place_of_a_store_read() -> None:
    built = InMemoryMaterialisationStore()
    leaf = await _derive(built, "rank", {"rank_by": []}, frame_ref=await _root(built))
    roots_only = InMemoryMaterialisationStore()
    await _root(roots_only)

    with pytest.raises(RecipeAncestorMissingError) as raised:
        await build_recipe(leaf, store=roots_only)
    assert raised.value.ref == leaf

    recipe = await build_recipe(leaf, store=roots_only, trace=await built.get_trace(leaf))
    assert recipe == await build_recipe(leaf, store=built)


async def test_a_root_without_a_mint_record_fails_or_is_recovered() -> None:
    store = InMemoryMaterialisationStore()
    root = await _root(store, recorded=False)
    leaf = await _derive(store, "rank", {"rank_by": []}, frame_ref=root)
    recovered = MintRecord(frame=root, dataset=root, options=_OPTIONS)

    async def recover(ref: FrameRef) -> MintRecord:
        assert ref == root
        return recovered

    async def cannot_recover(ref: FrameRef) -> MintRecord:
        raise MintRecordUnrecoverableError(ref, reason="groupby_unrecorded")

    with pytest.raises(RecipeRootUnrecoverableError) as unrecorded:
        await build_recipe(leaf, store=store)
    assert (unrecorded.value.ref, unrecorded.value.reason) == (root, "no_mint_record")

    with pytest.raises(RecipeRootUnrecoverableError) as unrecoverable:
        await build_recipe(leaf, store=store, recover=cannot_recover)
    assert (unrecoverable.value.ref, unrecoverable.value.reason) == (root, "groupby_unrecorded")

    recipe = await build_recipe(leaf, store=store, recover=recover)
    assert recipe["roots"] == [recovered.model_dump(mode="json")]


async def test_a_trace_whose_op_spec_is_not_the_frames_own_fails() -> None:
    store = InMemoryMaterialisationStore()
    root = await _root(store)
    leaf = await _derive(store, "rank", {"rank_by": []}, frame_ref=root)
    misfiled = FrameRef(spec_id="ops_" + "e" * 32, content_version="ocv_" + "e" * 32)
    leaf_trace = await store.get_trace(leaf)
    assert leaf_trace is not None
    await store.put(misfiled, frame=object(), trace=leaf_trace, writer="test")
    bare = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "d" * 32)
    # A trace with no op-spec, as a root's `MaterialisationTrace` is.
    await store.put(bare, frame=object(), trace=object(), writer="test")

    for ref in (misfiled, bare):
        with pytest.raises(RecipeTraceUnusableError) as raised:
            await build_recipe(ref, store=store)
        assert raised.value.ref == ref


async def test_a_root_frame_has_no_recipe() -> None:
    store = InMemoryMaterialisationStore()
    with pytest.raises(ValueError, match="root frame"):
        await build_recipe(await _root(store), store=store)


async def test_ancestors_from_two_engine_builds_do_not_share_a_replay_key_with_one() -> None:
    """A replay runs every step under one build, so a frame whose ancestors ran under
    two must not answer for it."""
    one_build = InMemoryMaterialisationStore()
    two_builds = InMemoryMaterialisationStore()
    older = "0.2.0|1.0.0|0.2.0"
    recipes = []
    for store, ancestor_identity in ((one_build, _IDENTITY), (two_builds, older)):
        root = await _root(store)
        middle = await _derive(
            store, "filter", {"predicates": []}, frame_ref=root, identity=ancestor_identity
        )
        leaf = await _derive(store, "rank", {"rank_by": []}, frame_ref=middle)
        recipes.append(await build_recipe(leaf, store=store))
    single, mixed = recipes

    assert single["execution_identity"] == mixed["execution_identity"] == _IDENTITY
    assert single["normal_form_hash"] == mixed["normal_form_hash"]
    assert single["replay_key"] != mixed["replay_key"]


_ROOT_NODE = {
    "frame": {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32},
    "root": 0,
}
_ROOTS = [{"dataset": {"spec_id": "spec_" + "a" * 32}, "options": _OPTIONS}]
_LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}
_TEN_BACK = {"anchor": "latest", "offset": -10, "rule": "half1"}


def _node(verb: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """A hand-built derived node over one root."""
    frame_ref = _ROOT_NODE["frame"]
    return {
        "frame": {"spec_id": "ops_" + "0" * 32, "content_version": "ocv_" + "0" * 32},
        "op_spec": {
            "identity_version": "ops1",
            "verb": verb,
            "frame_ref": frame_ref,
            "parameters": parameters,
        },
        "inputs": {"frame_ref": _ROOT_NODE},
    }


def _growth(start: str, end: str, rolling: dict[str, Any] | None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"window_start": start, "window_end": end, "entity_by": ["r"]}
    return _node("growth", parameters if rolling is None else parameters | {"rolling": rolling})


def _threshold(value: str, rolling: dict[str, Any] | None) -> dict[str, Any]:
    predicate = {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": value}
    return _node(
        "filter",
        {"predicates": [predicate if rolling is None else predicate | {"rolling": rolling}]},
    )


def test_the_normal_form_names_no_frame_and_no_root_version() -> None:
    assert normal_form(_growth("2013", "2023", None), _ROOTS) == {
        "identity_version": "ops1",
        "verb": "growth",
        "parameters": {"window_start": "2013", "window_end": "2023", "entity_by": ["r"]},
        "inputs": {"frame_ref": {"spec_id": "spec_" + "a" * 32, "rendering": _OPTIONS}},
    }


def test_the_normal_form_drops_each_literal_a_rolling_overlay_governs() -> None:
    both = {"window_start": _TEN_BACK, "window_end": _LATEST}

    assert normal_form(_growth("2013", "2023", both), _ROOTS)["parameters"] == {
        "entity_by": ["r"],
        "rolling": both,
    }
    # Only the governed end is dropped, and a fixed window keeps both.
    assert normal_form(_growth("2013", "2023", {"window_end": _LATEST}), _ROOTS)["parameters"] == {
        "window_start": "2013",
        "entity_by": ["r"],
        "rolling": {"window_end": _LATEST},
    }
    assert normal_form(_threshold("2023", _LATEST), _ROOTS)["parameters"] == {
        "predicates": [
            {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "rolling": _LATEST}
        ]
    }


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        (
            _growth("2013", "2023", {"window_start": _TEN_BACK, "window_end": _LATEST}),
            _growth("2014", "2024", {"window_start": _TEN_BACK, "window_end": _LATEST}),
            True,
        ),
        (_growth("2013", "2023", None), _growth("2014", "2024", None), False),
        (
            _growth("2013", "2023", {"window_end": _LATEST}),
            _growth("2014", "2024", {"window_end": _LATEST}),
            False,
        ),
        (
            _growth("2013", "2023", {"window_start": _TEN_BACK, "window_end": _LATEST}),
            _growth("2013", "2023", {"window_start": _LATEST, "window_end": _LATEST}),
            False,
        ),
        (_threshold("2023", _LATEST), _threshold("2024", _LATEST), True),
        (_threshold("2023", None), _threshold("2024", None), False),
        (_threshold("2023", _LATEST), _threshold("2023", None), False),
    ],
    ids=[
        "rolling-window-moved",
        "fixed-window-moved",
        "half-rolling-window-moved-at-its-fixed-end",
        "rolling-offset-changed",
        "rolling-threshold-moved",
        "fixed-threshold-moved",
        "rolling-against-fixed-threshold",
    ],
)
def test_only_a_rolling_literal_can_move_without_moving_the_normal_form_hash(
    first: dict[str, Any], second: dict[str, Any], same: bool
) -> None:
    hashes = [normal_form_hash(normal_form(tree, _ROOTS)) for tree in (first, second)]

    assert (hashes[0] == hashes[1]) is same


def test_the_two_hashes_are_sha256_of_their_documented_preimages() -> None:
    """Pinned to bytes written out here: apis indexes both values, so a change to
    either preimage orphans every stored recipe."""
    normal = {"spec_id": "spec_" + "a" * 32, "rendering": {"layout": "frame"}}
    normal_bytes = (
        b'{"identity_version":"rnf1","recipe":{"rendering":{"layout":"frame"},'
        b'"spec_id":"spec_' + b"a" * 32 + b'"}}'
    )
    hashed = normal_form_hash(normal)
    roots = [
        {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "1" * 32},
        {"spec_id": "spec_" + "b" * 32, "content_version": "cv_" + "2" * 32},
    ]
    replay_bytes = (
        b'{"execution_identities":["0.2.0|1.0.0|0.2.0","0.3.0|1.0.0|0.2.0"],'
        b'"identity_version":"rpk1","normal_form_hash":"' + hashed.encode() + b'","roots":['
        b'{"content_version":"cv_' + b"1" * 32 + b'","spec_id":"spec_' + b"a" * 32 + b'"},'
        b'{"content_version":"cv_' + b"2" * 32 + b'","spec_id":"spec_' + b"b" * 32 + b'"}]}'
    )

    assert hashed == f"rnf_{hashlib.sha256(normal_bytes).hexdigest()[:32]}"
    assert (
        replay_key(
            normal_form_hash=hashed,
            root_datasets=roots,
            execution_identities=[_IDENTITY, "0.2.0|1.0.0|0.2.0", _IDENTITY],
        )
        == f"rpk_{hashlib.sha256(replay_bytes).hexdigest()[:32]}"
    )
    # Root order is part of the key: it says which root holds which version.
    assert replay_key(
        normal_form_hash=hashed, root_datasets=roots[::-1], execution_identities=[_IDENTITY]
    ) != replay_key(normal_form_hash=hashed, root_datasets=roots, execution_identities=[_IDENTITY])
