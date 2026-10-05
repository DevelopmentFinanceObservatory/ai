"""`frame_ref_resolution.derive_frame_content_version` against a live-shaped
`upstream_trace`: a real dfo-apis manifest's `facets` entries carry
bookkeeping members beyond `dfo_frame_grammar.content_version.mint_content_version`'s
closed `{axis, ref, basis}` shape, a `basis` value
(`"deployment_fallback"`) outside its `bound`/`declared_default`
vocabulary, and an entry with `ref: None` (an axis with no facet selected)
-- all raise `ContentVersionError` if passed through unmodified
(see `frame_ref_resolution._sanitised_facets`'s own docstring).

Checked here: the derivation itself succeeds and matches a hand-projected
mint (unit level); a verb resolves and runs over a frame registered under
such a trace (verb level); a *second* verb call over the first verb's own
`{ops_, ocv_}` result also succeeds, reading the first result's own stored
`frame_content_version`; and, at the identity level, that a chained call's
`input_content_versions` binds exactly the value the prior hop stored, and
that a root-frame call's `input_content_versions` binds exactly the
reference's own `content_version`.
"""

from __future__ import annotations

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.fixtures import live_shaped_trace, plain_single_unit_frame, seed_fixture_store
from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from analysis_mcp.growth_tool import growth
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar.content_version import frame_sha256, mint_content_version


def test_derive_frame_content_version_sanitises_a_live_shaped_upstream_trace() -> None:
    frame = plain_single_unit_frame()
    upstream_trace = {
        "graph_revision": "rev_live_1",
        "source_revisions": [{"source": "live_ledger", "revision": "2026-08-01"}],
        "facets": [
            {
                "axis": "unit",
                "ref": "dcid:USDollar",
                "basis": "deployment_fallback",
                "default_policy": "latest",
                "default_policy_version": "v3",
                "dispatch_facet_id": "df_live_1",
                "selection_reason": "no explicit pin",
            }
        ],
        "conversion": None,
    }

    result = derive_frame_content_version(frame=frame, upstream_trace=upstream_trace)

    expected = mint_content_version(
        frame_sha256=frame_sha256(frame),
        graph_revision="rev_live_1",
        source_revisions=[{"source": "live_ledger", "revision": "2026-08-01"}],
        facets=[{"axis": "unit", "ref": "dcid:USDollar", "basis": "declared_default"}],
    )
    assert result == expected
    assert result.startswith("cv_")


def test_derive_frame_content_version_drops_a_null_ref_facet() -> None:
    """A facet entry's `ref` can be `None`: an axis with no facet selected,
    carrying `selection_reason: "absent_on_selected_facet"`.
    `mint_content_version` refuses a `None` ref, and an absent axis pins
    nothing, so `_sanitised_facets` drops the entry from the derivation's
    preimage.
    """
    frame = plain_single_unit_frame()
    upstream_trace = {
        "graph_revision": "rev_live_1",
        "source_revisions": [{"source": "live_ledger", "revision": "2026-08-01"}],
        "facets": [
            {
                "axis": "unit",
                "ref": "dcid:USDollar",
                "basis": "declared_default",
            },
            {
                "axis": "measurementMethod",
                "ref": None,
                "basis": "declared_default",
                "selection_reason": "absent_on_selected_facet",
            },
        ],
        "conversion": None,
    }

    result = derive_frame_content_version(frame=frame, upstream_trace=upstream_trace)

    expected = mint_content_version(
        frame_sha256=frame_sha256(frame),
        graph_revision="rev_live_1",
        source_revisions=[{"source": "live_ledger", "revision": "2026-08-01"}],
        facets=[{"axis": "unit", "ref": "dcid:USDollar", "basis": "declared_default"}],
    )
    assert result == expected


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
    once from the live-shaped lineage at write time
    (`derive_frame_content_version`). The second call's own input
    resolution reads that stored value (`input_content_version`), so it
    succeeds even though nothing here re-sanitises anything for it.
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
