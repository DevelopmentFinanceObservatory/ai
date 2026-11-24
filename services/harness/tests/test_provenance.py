"""Tests for `harness.provenance.ProvenanceStore`: the per-conversation
lineage cache and its two bounds -- an LRU cap on frames within one
conversation, and an LRU cap on the number of conversation buckets held at
once. The second is the one that matters for real use: the studio CLI mints
a fresh conversation id per run, so ordinary repeated use grows the number
of buckets, not the size of any one of them. Also `alias`, which writes a
derived frame's lineage from the retrievals under it, and `merged_manifest`,
which folds a lineage into one manifest-shaped dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from harness.provenance import (
    _MAX_CONVERSATIONS,
    _MAX_MANIFESTS_PER_CONVERSATION,
    ProvenanceInput,
    ProvenanceStore,
    merged_manifest,
)


def _manifest(tag: str) -> dict[str, object]:
    return {"citation": {"text": tag}}


def test_stat_var_count_is_what_alias_recorded_and_none_for_a_retrieval() -> None:
    store = ProvenanceStore()
    store.put(conversation="c1", spec_id="spec_a", content_version="v1", manifest=_manifest("a"))
    store.alias(
        conversation="c1",
        spec_id="ops_x",
        content_version="ocv_x",
        sources=[("spec_a", "v1")],
        stat_var_count=2,
    )

    assert store.stat_var_count(conversation="c1", spec_id="ops_x", content_version="ocv_x") == 2
    assert store.stat_var_count(conversation="c1", spec_id="spec_a", content_version="v1") is None
    assert store.stat_var_count(conversation="c1", spec_id="ghost", content_version="v1") is None
    # The lineage reads independently of the count.
    assert store.get(conversation="c1", spec_id="ops_x", content_version="ocv_x") == (
        ProvenanceInput("spec_a", "v1", _manifest("a")),
    )


_FRANCE_EDUCATION_GRANTS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "mint_trace_france_education_grants.json").read_text()
)


def _traced(refs: dict[str, int | None], *, summed: int | None = None) -> dict[str, object]:
    """A `get_data` result whose trace lists `refs`, each with its row count
    (`None` leaves `observation_count` out), summing `summed` StatVars."""
    entries = [
        {"ref": ref} if rows is None else {"ref": ref, "observation_count": rows}
        for ref, rows in refs.items()
    ]
    return {"trace": {"resolved_stat_vars": entries}, "summed_stat_var_count": summed}


def _carried_count(
    store: ProvenanceStore,
    carried: list[str],
    declared: int | None,
    sources: list[str] | None = None,
) -> int | None:
    store.alias(
        conversation="c1",
        spec_id="ops_x",
        content_version="ocv_x",
        sources=[(spec_id, "v1") for spec_id in sources or carried],
        stat_var_count=declared,
        carried=[(spec_id, "v1") for spec_id in carried],
    )
    return store.stat_var_count(conversation="c1", spec_id="ops_x", content_version="ocv_x")


def test_the_recorded_trace_lists_with_rows_exactly_the_stat_vars_dfo_apis_summed() -> None:
    """France's education grants, summed on dev apis: the trace lists 183
    members, the 68 with rows are the 68 the `stat_vars_summed` warning names,
    and a frame carrying the sum counts them."""
    entries = _FRANCE_EDUCATION_GRANTS["trace"]["resolved_stat_vars"]
    (warning,) = _FRANCE_EDUCATION_GRANTS["warnings"]
    with_rows = [entry for entry in entries if entry["observation_count"] > 0]
    assert (len(entries), len(with_rows)) == (183, 68)
    assert "68 StatVars were summed" in warning["message"]

    store = ProvenanceStore()
    _put(
        store, "spec_sum", {"trace": _FRANCE_EDUCATION_GRANTS["trace"], "summed_stat_var_count": 68}
    )

    assert _carried_count(store, ["spec_sum"], 1) == 68


def test_a_single_summed_family_counts_the_stat_vars_it_sums() -> None:
    store = ProvenanceStore()
    _put(store, "spec_abc", _traced({"A": 4, "B": 2, "C": 1, "Z": 0}, summed=3))

    assert _carried_count(store, ["spec_abc"], 1) == 3
    assert _carried_count(store, ["spec_abc"], None) is None


def test_two_summed_families_sharing_members_count_each_stat_var_once() -> None:
    """A join of the families {A, B, C} and {B, C}: three StatVars, not five."""
    store = ProvenanceStore()
    _put(store, "spec_abc", _traced({"A": 4, "B": 2, "C": 1}, summed=3))
    _put(store, "spec_bc", _traced({"B": 2, "C": 1, "Z": 0}, summed=2))

    assert _carried_count(store, ["spec_abc", "spec_bc"], 2) == 3


def test_disjoint_summed_families_add() -> None:
    store = ProvenanceStore()
    _put(store, "spec_ab", _traced({"A": 4, "B": 2}, summed=2))
    _put(store, "spec_cde", _traced({"C": 1, "D": 1, "E": 3}, summed=3))

    assert _carried_count(store, ["spec_ab", "spec_cde"], 2) == 5


def test_a_summed_family_beside_a_single_stat_var_counts_the_union() -> None:
    """Joined with one of its own members the family counts its three; joined
    with an outside StatVar, four."""
    store = ProvenanceStore()
    _put(store, "spec_abc", _traced({"A": 4, "B": 2, "C": 1}, summed=3))
    _put(store, "spec_b", _traced({"B": 2}))
    _put(store, "spec_d", _traced({"D": 5}))

    assert _carried_count(store, ["spec_abc", "spec_b"], 2) == 3
    assert _carried_count(store, ["spec_abc", "spec_d"], 2) == 4


def test_a_summed_retrieval_the_frame_does_not_carry_keeps_the_declared_count() -> None:
    """A retrieval the frame rests on without carrying its StatVars leaves the
    declared count alone."""
    store = ProvenanceStore()
    _put(store, "spec_abc", _traced({"A": 4, "B": 2, "C": 1}, summed=3))
    _put(store, "spec_d", _traced({"D": 5}))

    assert _carried_count(store, ["spec_d"], 1, sources=["spec_d", "spec_abc"]) == 1
    assert _carried_count(store, ["spec_d"], 1) == 1


def test_families_whose_contributors_cannot_be_named_add_only_when_disjoint() -> None:
    """With no row counts in the traces, each family counts as its summed
    count: disjoint member lists add, shared members give the larger alone."""
    store = ProvenanceStore()
    _put(store, "spec_abc", _traced({"A": None, "B": None, "C": None, "Z": None}, summed=3))
    _put(store, "spec_bc", _traced({"B": None, "C": None}, summed=2))
    _put(store, "spec_de", _traced({"D": None, "E": None, "Y": None}, summed=2))

    assert _carried_count(store, ["spec_abc"], 1) == 3
    assert _carried_count(store, ["spec_abc", "spec_bc"], 2) == 3
    assert _carried_count(store, ["spec_abc", "spec_de"], 2) == 5


def test_a_family_with_no_trace_never_adds_to_another_part() -> None:
    """With no member list, nothing shows two parts are disjoint."""
    store = ProvenanceStore()
    _put(store, "spec_sum", {**_manifest("sum"), "summed_stat_var_count": 74})
    _put(store, "spec_one", _manifest("one"))

    assert _carried_count(store, ["spec_sum"], 1) == 74
    assert _carried_count(store, ["spec_sum", "spec_one"], 2) == 74
    assert _carried_count(store, ["spec_one"], 1) == 1


def test_get_returns_none_for_a_key_never_written() -> None:
    store = ProvenanceStore()
    assert store.get(conversation="c1", spec_id="spec_a", content_version="v1") is None
    assert not store.was_evicted(conversation="c1", spec_id="spec_a", content_version="v1")


def test_put_then_get_round_trips_within_a_conversation() -> None:
    store = ProvenanceStore()
    store.put(conversation="c1", spec_id="spec_a", content_version="v1", manifest=_manifest("a"))
    got = store.get(conversation="c1", spec_id="spec_a", content_version="v1")
    assert got == (ProvenanceInput("spec_a", "v1", _manifest("a")),)


def test_a_conversation_never_holds_more_than_the_per_bucket_cap() -> None:
    store = ProvenanceStore()
    conversation = "c1"
    for index in range(_MAX_MANIFESTS_PER_CONVERSATION * 3):
        store.put(
            conversation=conversation,
            spec_id=f"spec_{index}",
            content_version="v1",
            manifest=_manifest(str(index)),
        )
        bucket = store._by_conversation[conversation]
        assert len(bucket) <= _MAX_MANIFESTS_PER_CONVERSATION

    # Least-recently-written entries are the ones gone; the newest survive.
    newest = _MAX_MANIFESTS_PER_CONVERSATION * 3 - 1
    assert store.get(conversation=conversation, spec_id=f"spec_{newest}", content_version="v1")
    assert store.get(conversation=conversation, spec_id="spec_0", content_version="v1") is None


def test_a_read_hit_protects_a_manifest_from_lru_eviction() -> None:
    store = ProvenanceStore()
    conversation = "c1"
    for index in range(_MAX_MANIFESTS_PER_CONVERSATION):
        store.put(
            conversation=conversation,
            spec_id=f"spec_{index}",
            content_version="v1",
            manifest=_manifest(str(index)),
        )
    # Touch the oldest entry so it is no longer the least-recently-used one.
    store.get(conversation=conversation, spec_id="spec_0", content_version="v1")
    store.put(
        conversation=conversation,
        spec_id="spec_new",
        content_version="v1",
        manifest=_manifest("new"),
    )
    assert store.get(conversation=conversation, spec_id="spec_0", content_version="v1")
    assert store.get(conversation=conversation, spec_id="spec_1", content_version="v1") is None


def test_the_store_never_holds_more_conversations_than_the_global_cap() -> None:
    """Drives far more conversations through the store than the cap allows,
    the shape the studio CLI's one-conversation-per-run usage produces, and
    checks the bound at every write -- not only once the run ends, which
    would pass even if the store spiked over the cap mid-run and only
    trimmed back down at the last write."""
    store = ProvenanceStore()
    total_conversations = _MAX_CONVERSATIONS * 3
    for index in range(total_conversations):
        store.put(
            conversation=f"conv_{index}",
            spec_id="spec_a",
            content_version="v1",
            manifest=_manifest(str(index)),
        )
        assert len(store._by_conversation) <= _MAX_CONVERSATIONS
        total_manifests = sum(len(bucket) for bucket in store._by_conversation.values())
        assert total_manifests <= _MAX_CONVERSATIONS * _MAX_MANIFESTS_PER_CONVERSATION

    # The newest conversations survive; the earliest are gone entirely.
    newest = f"conv_{total_conversations - 1}"
    assert store.get(conversation=newest, spec_id="spec_a", content_version="v1")
    assert store.get(conversation="conv_0", spec_id="spec_a", content_version="v1") is None


def test_a_read_on_a_conversation_protects_its_bucket_from_eviction() -> None:
    store = ProvenanceStore()
    for index in range(_MAX_CONVERSATIONS):
        store.put(
            conversation=f"conv_{index}",
            spec_id="spec_a",
            content_version="v1",
            manifest=_manifest(str(index)),
        )
    # Touch the oldest conversation so it is no longer least-recently-used.
    store.get(conversation="conv_0", spec_id="spec_a", content_version="v1")
    store.put(
        conversation="conv_new", spec_id="spec_a", content_version="v1", manifest=_manifest("new")
    )
    assert store.get(conversation="conv_0", spec_id="spec_a", content_version="v1")
    assert store.get(conversation="conv_1", spec_id="spec_a", content_version="v1") is None


def test_an_evicted_manifest_is_distinguishable_from_one_never_written() -> None:
    store = ProvenanceStore()
    store.put(conversation="c1", spec_id="spec_a", content_version="v1", manifest=_manifest("a"))
    for index in range(_MAX_MANIFESTS_PER_CONVERSATION):
        store.put(
            conversation="c1",
            spec_id=f"spec_fill_{index}",
            content_version="v1",
            manifest=_manifest(str(index)),
        )

    assert store.get(conversation="c1", spec_id="spec_a", content_version="v1") is None
    assert store.was_evicted(conversation="c1", spec_id="spec_a", content_version="v1")
    # A key that was genuinely never written stays merely missing.
    assert store.get(conversation="c1", spec_id="spec_ghost", content_version="v1") is None
    assert not store.was_evicted(conversation="c1", spec_id="spec_ghost", content_version="v1")


def test_a_whole_conversation_dropped_by_the_global_cap_is_recorded_as_evicted() -> None:
    store = ProvenanceStore()
    store.put(
        conversation="conv_0", spec_id="spec_a", content_version="v1", manifest=_manifest("a")
    )
    for index in range(1, _MAX_CONVERSATIONS + 1):
        store.put(
            conversation=f"conv_{index}",
            spec_id="spec_a",
            content_version="v1",
            manifest=_manifest(str(index)),
        )

    assert store.get(conversation="conv_0", spec_id="spec_a", content_version="v1") is None
    assert store.was_evicted(conversation="conv_0", spec_id="spec_a", content_version="v1")


def _put(store: ProvenanceStore, spec_id: str, manifest: dict[str, object]) -> None:
    store.put(conversation="c1", spec_id=spec_id, content_version="v1", manifest=manifest)


def test_alias_writes_the_sources_lineage_in_the_order_given() -> None:
    store = ProvenanceStore()
    _put(store, "spec_a", _manifest("a"))
    _put(store, "spec_b", _manifest("b"))

    linked = store.alias(
        conversation="c1",
        spec_id="ops_x",
        content_version="ocv_x",
        sources=[("spec_b", "v1"), ("spec_a", "v1")],
    )

    assert linked
    lineage = store.get(conversation="c1", spec_id="ops_x", content_version="ocv_x")
    assert lineage == (
        ProvenanceInput("spec_b", "v1", _manifest("b")),
        ProvenanceInput("spec_a", "v1", _manifest("a")),
    )


def test_alias_is_all_or_nothing_when_a_source_has_no_lineage() -> None:
    store = ProvenanceStore()
    _put(store, "spec_a", _manifest("a"))

    linked = store.alias(
        conversation="c1",
        spec_id="ops_x",
        content_version="ocv_x",
        sources=[("spec_a", "v1"), ("spec_ghost", "v1")],
    )

    assert not linked
    assert store.get(conversation="c1", spec_id="ops_x", content_version="ocv_x") is None


def test_alias_with_no_sources_writes_nothing() -> None:
    store = ProvenanceStore()

    linked = store.alias(conversation="c1", spec_id="ops_x", content_version="ocv_x", sources=[])

    assert not linked
    assert store.get(conversation="c1", spec_id="ops_x", content_version="ocv_x") is None


def test_alias_flattens_an_aliased_source_and_keeps_a_repeated_retrieval_once() -> None:
    store = ProvenanceStore()
    _put(store, "spec_a", _manifest("a"))
    _put(store, "spec_b", _manifest("b"))
    store.alias(
        conversation="c1",
        spec_id="ops_ab",
        content_version="ocv_ab",
        sources=[("spec_a", "v1"), ("spec_b", "v1")],
    )

    store.alias(
        conversation="c1",
        spec_id="ops_top",
        content_version="ocv_top",
        sources=[("ops_ab", "ocv_ab"), ("spec_a", "v1")],
    )

    lineage = store.get(conversation="c1", spec_id="ops_top", content_version="ocv_top")
    assert lineage is not None
    assert [member.spec_id for member in lineage] == ["spec_a", "spec_b"]


def test_an_aliased_frame_is_dropped_like_any_other_key_past_the_per_bucket_cap() -> None:
    store = ProvenanceStore()
    _put(store, "spec_a", _manifest("a"))
    store.alias(
        conversation="c1", spec_id="ops_x", content_version="ocv_x", sources=[("spec_a", "v1")]
    )
    for index in range(_MAX_MANIFESTS_PER_CONVERSATION):
        _put(store, f"spec_fill_{index}", _manifest(str(index)))

    assert store.get(conversation="c1", spec_id="ops_x", content_version="ocv_x") is None
    assert store.was_evicted(conversation="c1", spec_id="ops_x", content_version="ocv_x")


def _stated(
    spec_id: str,
    *,
    text: str,
    ref: str,
    licence: str | None = None,
    created_at: str | None = "2020-01-01T00:00:00+00:00",
    end_year: object = 2024,
    warnings: list[dict[str, str]] | None = None,
    stat_vars: list[str] | None = None,
) -> ProvenanceInput:
    manifest: dict[str, object] = {
        "citation": {
            "text": text,
            "sources": [
                {"ref": ref, "label": ref.upper(), **({"license": licence} if licence else {})}
            ],
        },
        "coverage": {"window": {"end_year": end_year}},
        "artifacts": [{"created_at": created_at}] if created_at else [],
        "trace": {"resolved_stat_vars": [{"ref": var} for var in stat_vars or []]},
        "warnings": warnings or [],
    }
    return ProvenanceInput(spec_id, "v1", manifest)


def test_a_one_input_lineage_merges_to_that_inputs_own_manifest() -> None:
    only = _stated("spec_a", text="A", ref="a")

    assert merged_manifest((only,)) is only.manifest


def test_merging_unions_sources_by_ref_and_joins_distinct_citations_in_lineage_order() -> None:
    first = _stated("spec_a", text="Source A", ref="a", licence="CC BY 4.0")
    second = _stated("spec_b", text="Source B", ref="b")
    repeat = _stated("spec_c", text="Source A", ref="a")

    merged = merged_manifest((first, second, repeat))

    assert merged["citation"]["text"] == "Source A; Source B"
    assert [source["ref"] for source in merged["citation"]["sources"]] == ["a", "b"]
    assert [source.get("license") for source in merged["citation"]["sources"]] == [
        "CC BY 4.0",
        None,
    ]


def test_merging_takes_the_oldest_creation_date_and_the_earliest_end_year() -> None:
    recent = _stated(
        "spec_a", text="A", ref="a", created_at="2024-03-09T10:00:00+00:00", end_year=2024
    )
    stale = _stated(
        "spec_b", text="B", ref="b", created_at="2019-05-01T00:00:00+00:00", end_year="2021"
    )

    merged = merged_manifest((recent, stale))

    assert merged["artifacts"] == [{"created_at": "2019-05-01"}]
    assert merged["coverage"] == {"window": {"end_year": 2021}}


@pytest.mark.parametrize("unreadable", [None, "", "not a year"])
def test_merging_leaves_an_end_year_out_when_any_input_lacks_a_readable_one(
    unreadable: object,
) -> None:
    known = _stated("spec_a", text="A", ref="a", end_year=2024)
    unknown = _stated("spec_b", text="B", ref="b", end_year=unreadable)

    assert "coverage" not in merged_manifest((known, unknown))


@pytest.mark.parametrize("created_at", [None, "yesterday-ish"])
def test_merging_leaves_the_creation_date_out_when_any_input_lacks_a_readable_one(
    created_at: str | None,
) -> None:
    known = _stated("spec_a", text="A", ref="a")
    unknown = _stated("spec_b", text="B", ref="b", created_at=created_at)

    assert "artifacts" not in merged_manifest((known, unknown))


def test_merging_tags_each_warning_with_its_retrievals_spec_id() -> None:
    warning = {"code": "provisional", "severity": "warn", "message": "latest year is provisional"}
    first = _stated("spec_a", text="A", ref="a", warnings=[warning])
    second = _stated("spec_b", text="B", ref="b", warnings=[warning])

    merged = merged_manifest((first, second))

    assert merged["warnings"] == [
        {**warning, "spec_id": "spec_a"},
        {**warning, "spec_id": "spec_b"},
    ]


def test_merging_unions_resolved_stat_vars_by_ref() -> None:
    first = _stated("spec_a", text="A", ref="a", stat_vars=["dcid:Debt", "dcid:Gdp"])
    second = _stated("spec_b", text="B", ref="b", stat_vars=["dcid:Gdp", "dcid:Population"])

    merged = merged_manifest((first, second))

    assert [var["ref"] for var in merged["trace"]["resolved_stat_vars"]] == [
        "dcid:Debt",
        "dcid:Gdp",
        "dcid:Population",
    ]
