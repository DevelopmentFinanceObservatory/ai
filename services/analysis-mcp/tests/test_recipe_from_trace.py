"""`common.recipe.build_recipe` over the traces this service's verbs store.

`common` reads a trace by attribute and an op-spec as parsed JSON, since it does
not depend on `analysis_mcp`. This runs two real verbs and builds the recipe of
the result, so a change to `Trace` or to the op-spec preimage that the recipe
walk cannot read fails here.
"""

from __future__ import annotations

from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.fixtures import plain_single_unit_frame
from analysis_mcp.growth_tool import growth
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    set_store,
)
from common.recipe import build_recipe, normal_form

_OPTIONS = {"format": "parquet", "bundle": False, "layout": "frame"}


async def _growth_of_a_sum(store: InMemoryMaterialisationStore, root: FrameRef) -> FrameRef:
    """Registers `root` with its mint record and returns `growth(aggregate(root))`."""
    trace = MaterialisationTrace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": root.content_version}],
            "facets": [],
            "conversion": None,
        },
        spec_id=root.spec_id,
        content_version=root.content_version,
    )
    await store.put(root, frame=plain_single_unit_frame(), trace=trace, writer="test")
    await store.put_mint_record(MintRecord(frame=root, dataset=root, options=_OPTIONS))
    summed = await aggregate(
        frame_ref=root.model_dump(), aggregation="sum", group_by=("reporter", "time_period")
    )
    assert summed.is_error is False, summed.content
    grown = await growth(
        frame_ref={key: summed.structured_content[key] for key in ("spec_id", "content_version")},
        window_start="2021",
        window_end="2023",
        entity_by=("reporter",),
    )
    assert grown.is_error is False, grown.content
    return FrameRef(
        spec_id=grown.structured_content["spec_id"],
        content_version=grown.structured_content["content_version"],
    )


async def test_a_recipe_is_built_from_the_traces_two_chained_verbs_store() -> None:
    store = InMemoryMaterialisationStore()
    set_store(store)
    spec_id = "spec_" + "a" * 32
    first_root = FrameRef(spec_id=spec_id, content_version="cv_" + "1" * 32)
    second_root = FrameRef(spec_id=spec_id, content_version="cv_" + "2" * 32)

    first = await build_recipe(await _growth_of_a_sum(store, first_root), store=store)
    second = await build_recipe(await _growth_of_a_sum(store, second_root), store=store)

    summed = first["tree"]["inputs"]["frame_ref"]
    assert first["tree"]["op_spec"]["verb"] == "growth"
    assert first["tree"]["op_spec"]["frame_ref"] == summed["frame"]
    assert summed["op_spec"]["verb"] == "aggregate"
    assert summed["inputs"] == {"frame_ref": {"frame": first_root.model_dump(), "root": 0}}
    assert first["execution_identity"] == current_execution_identity().canonical_string()
    assert normal_form(first["tree"], first["roots"]) == {
        "identity_version": "ops1",
        "verb": "growth",
        "parameters": first["tree"]["op_spec"]["parameters"],
        "inputs": {
            "frame_ref": {
                "identity_version": "ops1",
                "verb": "aggregate",
                "parameters": summed["op_spec"]["parameters"],
                "inputs": {"frame_ref": {"spec_id": spec_id, "rendering": _OPTIONS}},
            }
        },
    }
    # The same two verbs over another version of the dataset: every frame id moves,
    # the normal form does not.
    assert first["tree"]["frame"] != second["tree"]["frame"]
    assert first["normal_form_hash"] == second["normal_form_hash"]
    assert first["replay_key"] != second["replay_key"]
