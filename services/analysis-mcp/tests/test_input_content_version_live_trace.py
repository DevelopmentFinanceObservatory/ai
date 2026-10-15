"""Verbs over a frame registered under a live-shaped `upstream_trace`: a real
apis manifest's `facets` entries carry bookkeeping members beyond
`{axis, ref, basis}`, a `basis` of `"deployment_fallback"`, and an entry with
`ref: None` (an axis with no facet selected).

Checked here: a verb's output `frame_content_version` is one token for one
frame whatever lineage its input carries; a verb resolves and runs over a
frame registered under such a trace (verb level); a *second* verb call over
the first verb's own `{ops_, ocv_}` result also succeeds, reading the first
result's own stored `frame_content_version`; and, at the identity level, that
a chained call's `input_content_versions` binds exactly the value the prior
hop stored, and that a root-frame call's `input_content_versions` binds
exactly the reference's own `content_version`.
"""

from __future__ import annotations

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.fixtures import live_shaped_trace, plain_single_unit_frame, seed_fixture_store
from analysis_mcp.growth_tool import growth
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar.content_version import frame_sha256, mint_content_version


async def test_a_verbs_frame_content_version_is_the_same_under_either_lineage(seeded) -> None:
    """One frame registered under two references whose `upstream_trace`s differ
    in `graph_revision`, `source_revisions` and `facets`. The same verb call
    over each writes byte-identical output frames, so both results carry one
    `frame_content_version`, the token `mint_content_version` gives that frame.
    """
    store, handles = seeded
    frame = plain_single_unit_frame()
    live_trace = live_shaped_trace("live_plain_single_unit", frame)
    live_ref = FrameRef(spec_id=live_trace.spec_id, content_version=live_trace.content_version)
    await store.put(live_ref, frame=frame, trace=live_trace, writer="test")
    fixture_ref = FrameRef(**handles.plain_single_unit)
    _fixture_frame, fixture_trace = await store.get(fixture_ref)
    assert fixture_trace.upstream_trace != live_trace.upstream_trace

    results = [
        await aggregate(
            frame_ref={"spec_id": ref.spec_id, "content_version": ref.content_version},
            aggregation="sum",
            group_by=("reporter", "time_period"),
        )
        for ref in (fixture_ref, live_ref)
    ]

    output_frame, _output_trace = await store.get(
        FrameRef(
            spec_id=results[0].structured_content["spec_id"],
            content_version=results[0].structured_content["content_version"],
        )
    )
    expected = mint_content_version(frame_sha256=frame_sha256(output_frame))
    assert [r.structured_content["frame_content_version"] for r in results] == [expected, expected]
    assert (
        results[0].structured_content["content_version"]
        != results[1].structured_content["content_version"]
    )


@pytest.fixture
async def seeded_live():
    store = InMemoryMaterialisationStore()
    frame = plain_single_unit_frame()
    trace = live_shaped_trace("live_plain_single_unit", frame)
    ref = FrameRef(spec_id=trace.spec_id, content_version=trace.content_version)
    await store.put(ref, frame=frame, trace=trace, writer="test")
    set_store(store)
    yield store, ref
    set_store(None)


async def test_a_verb_resolves_and_runs_over_a_live_shaped_input(seeded_live) -> None:
    _store, ref = seeded_live

    result = await aggregate(
        frame_ref={"spec_id": ref.spec_id, "content_version": ref.content_version},
        aggregation="sum",
        group_by=("reporter", "time_period"),
    )

    assert result.is_error is False
    assert result.structured_content["spec_id"].startswith("ops_")
    assert result.structured_content["content_version"].startswith("ocv_")


async def test_a_second_verb_over_the_first_verbs_own_result_also_succeeds(seeded_live) -> None:
    """The first call's own output stores `frame_content_version`, derived
    once at write time (`derive_frame_content_version`). The second call's
    own input resolution reads that stored value (`input_content_version`).
    """
    _store, ref = seeded_live

    first = await aggregate(
        frame_ref={"spec_id": ref.spec_id, "content_version": ref.content_version},
        aggregation="sum",
        group_by=("reporter", "time_period"),
    )
    assert first.is_error is False
    first_pair = {
        "spec_id": first.structured_content["spec_id"],
        "content_version": first.structured_content["content_version"],
    }

    second = await aggregate(frame_ref=first_pair, aggregation="sum")

    assert second.is_error is False


@pytest.fixture
async def seeded():
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)
    set_store(store)
    yield store, handles
    set_store(None)


async def test_root_frame_hops_input_content_versions_onto_the_references_own_content_version(
    seeded,
) -> None:
    """A root frame's `input_content_versions` binds the manifest token the
    caller's `frame_ref` already named. `MaterialisationTrace.
    frame_content_version` is `content_version` under its own name."""
    store, handles = seeded

    result = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="sum", group_by=("reporter",)
    )
    assert result.is_error is False
    output_ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    _frame, stored_trace = await store.get(output_ref)

    assert stored_trace.input_content_versions == (handles.plain_single_unit["content_version"],)


async def test_chained_growth_hops_input_content_versions_onto_the_first_hops_frame_content_version(
    seeded,
) -> None:
    """`growth`'s own input resolution, over `aggregate`'s own result, reads
    that result's own stored `frame_content_version`, the value
    `aggregate` derived once when it wrote that result."""
    store, handles = seeded

    first = await aggregate(
        frame_ref=handles.plain_single_unit,
        aggregation="sum",
        group_by=("reporter", "time_period"),
    )
    assert first.is_error is False
    first_ref = FrameRef(
        spec_id=first.structured_content["spec_id"],
        content_version=first.structured_content["content_version"],
    )
    _first_frame, first_trace = await store.get(first_ref)

    second = await growth(
        frame_ref={"spec_id": first_ref.spec_id, "content_version": first_ref.content_version},
        window_start="2021",
        window_end="2023",
        entity_by=("reporter",),
    )
    assert second.is_error is False
    second_ref = FrameRef(
        spec_id=second.structured_content["spec_id"],
        content_version=second.structured_content["content_version"],
    )
    _second_frame, second_trace = await store.get(second_ref)

    assert second_trace.input_content_versions == (first_trace.frame_content_version,)
