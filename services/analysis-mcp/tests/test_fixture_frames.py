"""Fixture frames validate through `Frame.build`, cover the shapes the
verbs need, and register cleanly into the materialisation store behind the
`seed_fixture_frames` setting."""

from __future__ import annotations

from analysis_mcp.fixtures import (
    SeededFixtures,
    join_compatible_frame_pair,
    multi_unit_frame,
    plain_single_unit_frame,
    seed_fixture_store,
    sparse_coverage_frame,
    time_series_with_gap_year_frame,
)
from common.materialisation import FrameRef, InMemoryMaterialisationStore
from dfo_frame_grammar import Frame


def test_plain_single_unit_frame_is_a_real_frame_with_one_unit() -> None:
    frame = plain_single_unit_frame()

    assert isinstance(frame, Frame)
    assert len(frame.rows) >= 10
    unit_index = frame.columns.index("unit")
    assert {row[unit_index] for row in frame.rows} == {"dcid:USDollar"}


def test_multi_unit_frame_carries_heterogeneous_units() -> None:
    frame = multi_unit_frame()

    unit_index = frame.columns.index("unit")
    units = {row[unit_index] for row in frame.rows}
    assert units == {"dcid:USDollar", "dcid:Percent"}


def test_sparse_coverage_frame_has_fewer_rows_than_the_full_grid() -> None:
    frame = sparse_coverage_frame()

    recipient_index = frame.columns.index("recipient")
    time_index = frame.columns.index("time_period")
    recipients = {row[recipient_index] for row in frame.rows}
    years = {row[time_index] for row in frame.rows}
    full_grid_size = len(recipients) * len(years)

    assert len(frame.rows) < full_grid_size


def test_join_compatible_frame_pair_shares_reporter_and_time_period_grain() -> None:
    oda, gni = join_compatible_frame_pair()

    def grain_keys(frame: Frame) -> set[tuple[object, object]]:
        reporter_index = frame.columns.index("reporter")
        time_index = frame.columns.index("time_period")
        return {(row[reporter_index], row[time_index]) for row in frame.rows}

    oda_keys = grain_keys(oda)
    gni_keys = grain_keys(gni)
    assert oda_keys & gni_keys
    assert oda_keys >= gni_keys


def test_time_series_with_gap_year_frame_skips_exactly_one_year() -> None:
    frame = time_series_with_gap_year_frame()

    time_index = frame.columns.index("time_period")
    years = sorted(int(row[time_index]) for row in frame.rows)
    full_span = range(years[0], years[-1] + 1)
    missing = sorted(set(full_span) - set(years))

    assert missing == [2021]


async def test_seed_fixture_store_writes_every_fixture_under_a_valid_pair_and_is_readable() -> None:
    store = InMemoryMaterialisationStore()

    seeded = await seed_fixture_store(store)

    assert isinstance(seeded, SeededFixtures)
    all_pairs = (
        seeded.plain_single_unit,
        seeded.multi_unit,
        seeded.sparse_coverage,
        *seeded.join_pair,
        seeded.time_series_with_gap,
        *seeded.sum_pair,
    )
    distinct_pairs = {(pair["spec_id"], pair["content_version"]) for pair in all_pairs}
    assert len(distinct_pairs) == len(all_pairs), "every fixture gets its own pair"
    for pair in all_pairs:
        # Raises if the pair's shape doesn't satisfy FrameRef's own patterns.
        FrameRef(**pair)

    plain_result = await store.get(FrameRef(**seeded.plain_single_unit))
    oda_pair, gni_pair = seeded.join_pair
    oda_result = await store.get(FrameRef(**oda_pair))
    gni_result = await store.get(FrameRef(**gni_pair))

    assert plain_result is not None
    assert oda_result is not None
    assert gni_result is not None
