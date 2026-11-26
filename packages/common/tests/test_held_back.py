"""Tests for `common.held_back`."""

from __future__ import annotations

import logging

import pytest
from common.held_back import held_back_of

_PARTIAL = {
    "kind": "partial",
    "source": "dfo/crs",
    "source_name": "OECD CRS",
    "start_year": 2025,
    "end_year": 2025,
    "reported_share": {"basis": "series", "share": 0.12, "against": 2024},
}
_PROJECTED = {
    "kind": "projected",
    "source": "dfo/ids",
    "source_name": "World Bank IDS",
    "start_year": 2026,
    "end_year": 2030,
    "reported_share": None,
    "served": True,
    "unseen": 1,
}


def test_a_search_coverage_block_with_both_kinds_parses_to_values() -> None:
    partial, projected = held_back_of({"kind": "exact", "held_back": [_PARTIAL, _PROJECTED]})

    assert (partial.kind, partial.source_name, partial.start_year) == ("partial", "OECD CRS", 2025)
    assert (partial.served, projected.served) == (False, True)
    assert partial.reported_share is not None
    assert (partial.reported_share.basis, partial.reported_share.share) == ("series", 0.12)
    assert (projected.kind, projected.end_year, projected.reported_share) == (
        "projected",
        2030,
        None,
    )


def test_a_coverage_block_without_held_back_reads_empty() -> None:
    assert held_back_of({"kind": "exact"}) == ()
    assert held_back_of({"held_back": []}) == ()


@pytest.mark.parametrize(
    "bad",
    [{**_PARTIAL, "kind": "late"}, {**_PARTIAL, "source": None}, "2025"],
    ids=["an-unknown-kind", "a-null-source", "not-a-mapping"],
)
def test_a_malformed_entry_is_dropped_with_a_warning_and_the_rest_kept(
    bad: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="common.held_back"):
        runs = held_back_of({"held_back": [bad, _PROJECTED]}, label="spec_crs")

    assert [(run.kind, run.start_year) for run in runs] == [("projected", 2026)]
    assert [record.args[:3] for record in caplog.records] == [("held_back", 0, "spec_crs")]
