"""The `rank` tool end to end: every refusal path, the coverage floor, the
tie policy through the registered tool, one real analyst question, and the
replay + execution-identity-perturbation determinism proofs.
"""

from __future__ import annotations

import json

import pytest
from analysis_mcp import rank_tool
from analysis_mcp import server as server_module
from analysis_mcp.artifact import Interval
from analysis_mcp.engine.rank import run_rank
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.frame_ref_resolution import build_interval, derive_frame_content_version
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import (
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import MAX_RANK_LIMIT, AggregateOpSpec
from analysis_mcp.rank_tool import rank
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp import Client
from fastmcp.tools import ToolResult
from rolling_support import anchored, second_version
from runner_replay import frame_digest


@pytest.fixture
async def seeded():
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)
    set_store(store)
    yield store, handles
    set_store(None)


def _minimal_trace() -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
        group_by=(),
        aggregation="sum",
    )
    return Trace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": "1"}],
            "facets": [],
            "conversion": None,
        },
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash_for(op_spec),
        execution_record=ExecutionRecord(content="{}"),
        applied_policy=AppliedPolicy(),
        content_version="ocv_" + "c" * 32,
        input_content_versions=["cv_" + "b" * 32],
        execution_identity=current_execution_identity(),
    )


async def _seed_custom_frame(store: InMemoryMaterialisationStore, frame: Frame) -> dict[str, str]:
    digest = frame_digest(frame)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


# Refusals ----------------------------------------------------------------------


async def test_op_spec_invalid_refuses(seeded) -> None:
    _store, _handles = seeded
    result = await rank(frame_ref="not-a-valid-pair", rank_by=("reporter",))
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_empty_rank_by_refuses_op_spec_invalid(seeded) -> None:
    _store, handles = seeded
    result = await rank(frame_ref=handles.plain_single_unit, rank_by=())
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_one_sided_window_refuses_op_spec_invalid(seeded) -> None:
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit, rank_by=("reporter",), window_start="2021"
    )
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_unregistered_pair_refuses_frame_not_held(seeded) -> None:
    _store, _handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await rank(frame_ref=never_registered, rank_by=("reporter",))
    assert result.structured_content["code"] == "frame_not_held"


async def test_unsupported_aggregation_refuses_operation_uncovered(seeded) -> None:
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit, rank_by=("reporter",), aggregation="median"
    )
    assert result.structured_content["code"] == "operation_uncovered"


async def test_unknown_rank_by_column_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await rank(frame_ref=handles.plain_single_unit, rank_by=("donor",))
    assert result.structured_content["code"] == "role_ambiguous"


async def test_reserved_column_in_rank_by_refuses_role_ambiguous(seeded) -> None:
    """`time_period` (and `stat_var`) are reserved -- naming either as a
    `rank_by` column collides inside `Frame.build` with a raw `FrameError`
    unless caught first at validation. `time_series_with_gap` carries one
    reporter, one row per period, so grouping by `time_period` alone never
    trips `RankAmbiguousEntityError` first (unlike `plain_single_unit`,
    whose several reporters share each period and would reach that refusal
    regardless of whether `time_period` were ever disallowed as a
    `rank_by` column) -- only `validate_entity_columns`'s own widened
    disallowed set stands between this call and the `Frame.build` crash.
    """
    _store, handles = seeded
    result = await rank(frame_ref=handles.time_series_with_gap, rank_by=("time_period",))
    assert result.structured_content["code"] == "role_ambiguous"
    assert result.structured_content["fix"]["columns"] == ["time_period"]


async def test_duplicate_rank_by_column_refuses_op_spec_invalid_rather_than_crashing(
    seeded,
) -> None:
    """A duplicate `rank_by` entry passes op-spec parsing but
    `op_spec_hash_for` raises `OpSpecIdentityError` -- this must reach the
    tool boundary as a typed refusal, not an unhandled exception."""
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter", "reporter"),
        aggregation="sum",
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_ambiguous_entity_without_aggregation_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",))
    assert result.structured_content["code"] == "role_ambiguous"


async def test_no_rows_in_window_refuses_stat_var_unresolved(seeded) -> None:
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2099",
        window_end="2100",
    )
    assert result.structured_content["code"] == "stat_var_unresolved"


async def test_heterogeneous_unit_frame_refuses(seeded) -> None:
    _store, handles = seeded
    result = await rank(frame_ref=handles.multi_unit, rank_by=("reporter",), aggregation="sum")
    assert result.structured_content["code"] == "unit_heterogeneous"


async def test_coverage_below_the_floor_refuses(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, handles = seeded
    monkeypatch.setattr(rank_tool, "COVERAGE_FLOOR", 0.9)  # sparse_coverage_frame realises 0.75
    result = await rank(
        frame_ref=handles.sparse_coverage, rank_by=("recipient",), aggregation="sum"
    )
    assert result.structured_content["code"] == "coverage_insufficient"


# max / min -----------------------------------------------------------------------


async def test_max_aggregation_reduces_each_entitys_rows_before_ranking(seeded) -> None:
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        aggregation="max",
        window_start="2021",
        window_end="2023",
    )
    assert result.is_error is False
    assert "rank: reporter by max, window 2021..2023" in result.content[0].text


async def test_a_one_year_window_is_named_as_that_year(seeded) -> None:
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        window_start="2021",
        window_end="2021",
    )
    assert result.is_error is False
    assert "rank: reporter by (raw value), window 2021\n" in result.content[0].text


# Tie policy, through the registered tool ----------------------------------------


async def test_tied_entities_share_a_rank_number_and_a_deterministic_order(seeded) -> None:
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:ZZZ", "time_period": "2020", "value": 100.0},
            {"stat_var": "dcid:X", "reporter": "dcid:AAA", "time_period": "2020", "value": 100.0},
            {"stat_var": "dcid:X", "reporter": "dcid:MMM", "time_period": "2020", "value": 50.0},
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await rank(frame_ref=custom_ref, rank_by=("reporter",))
    digest = result.content[0].text
    aaa_index = digest.index("dcid:AAA")
    zzz_index = digest.index("dcid:ZZZ")
    mmm_index = digest.index("dcid:MMM")
    assert aaa_index < zzz_index < mmm_index
    assert "#1 reporter=dcid:AAA" in digest
    assert "#1 reporter=dcid:ZZZ" in digest
    assert "#3 reporter=dcid:MMM" in digest


# Per-period mode, through the registered tool -----------------------------------


async def test_per_period_rank_digest_lists_one_block_per_period(seeded) -> None:
    _store, handles = seeded
    result = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert result.is_error is False
    digest = result.content[0].text
    assert "per_period=True" in digest
    assert "2021:" in digest
    assert "2022:" in digest
    assert "2023:" in digest
    assert "#1 reporter=dcid:USA" in digest


async def test_per_period_rank_structured_content_row_count_is_entities_times_periods(
    seeded,
) -> None:
    """Five donors, three years, every cell present, so the per-period output
    frame has 15 rows. `rows` reflects that count. The collapsing mode's
    frame has 5."""
    _store, handles = seeded
    result = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert result.is_error is False
    assert result.structured_content["rows"] == 15


async def test_per_period_rank_coverage_floor_check_is_unaffected(
    seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coverage floor is checked identically in either mode -- it reads
    `RankResult.overall_coverage_fraction`, which `per_period` doesn't
    change the computation of."""
    _store, handles = seeded
    monkeypatch.setattr(rank_tool, "COVERAGE_FLOOR", 0.9)  # sparse_coverage_frame realises 0.75
    result = await rank(frame_ref=handles.sparse_coverage, rank_by=("recipient",), per_period=True)
    assert result.structured_content["code"] == "coverage_insufficient"


# Rank positions as input ---------------------------------------------------------


@pytest.mark.parametrize(
    "second",
    [
        {"per_period": True, "limit": 1},
        {"per_period": True},
        {"aggregation": "mean", "limit": 1},
        {"aggregation": "mean"},
    ],
)
async def test_a_rank_over_rank_positions_refuses_derived_input_ordinal(
    seeded, second: dict[str, object]
) -> None:
    """AAA disburses 100 and BBB 10, so an unlimited per-period rank writes
    positions 1 and 2. A highest-first rank over those positions would put
    BBB first, at "value" 2, so every rank over them refuses."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": reporter,
                "time_period": "2020",
                "value": value,
                "unit": "dcid:USDollar",
            }
            for reporter, value in (("dcid:AAA", 100.0), ("dcid:BBB", 10.0))
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    positions = await rank(frame_ref=custom_ref, rank_by=("reporter",), per_period=True)
    assert positions.is_error is False
    positions_ref = {
        "spec_id": positions.structured_content["spec_id"],
        "content_version": positions.structured_content["content_version"],
    }

    result = await rank(frame_ref=positions_ref, rank_by=("reporter",), **second)

    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert "limit" in result.structured_content["message"]


async def test_a_limited_per_period_rank_keeps_the_entity_with_the_largest_value(seeded) -> None:
    """The call the refusal above advises: the limit on the rank that reads
    the measured values keeps AAA at 100."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": reporter,
                "time_period": "2020",
                "value": value,
                "unit": "dcid:USDollar",
            }
            for reporter, value in (("dcid:AAA", 100.0), ("dcid:BBB", 10.0))
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)

    result = await rank(frame_ref=custom_ref, rank_by=("reporter",), per_period=True, limit=1)

    assert result.is_error is False
    kept, _trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    rows = kept.table.to_pylist()
    assert [(row["reporter"], row["value"]) for row in rows] == [("dcid:AAA", 100.0)]


# Limit, through the registered tool --------------------------------------------


@pytest.mark.parametrize("limit", [0, 1001])
async def test_limit_out_of_bounds_refuses_op_spec_invalid(seeded, limit: int) -> None:
    _store, handles = seeded
    result = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), limit=limit)
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_the_advertised_limit_bounds_match_the_op_spec() -> None:
    """A caller reading `tools/list` sees the same 1..1000 the op-spec
    enforces, so a limit past it fails the advertised schema first."""
    async with Client(server_module.mcp) as client:
        tools = await client.list_tools()
    (rank_tool,) = [tool for tool in tools if tool.name == "rank"]
    limit_schema = rank_tool.input_schema["properties"]["limit"]
    assert (limit_schema["type"], limit_schema["minimum"], limit_schema["maximum"]) == (
        "integer",
        1,
        MAX_RANK_LIMIT,
    )


async def test_limit_moves_the_op_spec_hash_and_the_content_version(seeded) -> None:
    _store, handles = seeded
    arguments = {
        "frame_ref": handles.plain_single_unit,
        "rank_by": ("reporter",),
        "aggregation": "sum",
        "window_start": "2021",
        "window_end": "2023",
    }
    unlimited = await rank(**arguments)
    limited = await rank(**arguments, limit=2)

    assert limited.structured_content["spec_id"] != unlimited.structured_content["spec_id"]
    assert unlimited.structured_content["ranked_before_limit"] is None
    assert (
        limited.structured_content["content_version"]
        != unlimited.structured_content["content_version"]
    )


async def test_limit_keeps_the_top_entities_and_reports_the_rest(seeded) -> None:
    """Summed over 2021-2023 the USA and Germany lead; Japan, the UK and
    France are cut, three years of rows each."""
    store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2021",
        window_end="2023",
        limit=2,
    )

    assert result.is_error is False
    assert result.structured_content["rows"] == 2
    assert result.structured_content["ranked_before_limit"] == 5
    digest = result.content[0].text
    assert "limit 2 (kept 2 of 5 ranked entities)" in digest
    assert "beyond_limit (ranked beyond the limit): 9 row(s)" in digest
    assert "2 ranked entit(y/ies):" in digest
    assert [line.split()[0] for line in digest.splitlines() if line.startswith("  #")] == [
        "#1",
        "#2",
    ]

    stored_frame, stored_trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    column = stored_frame.columns.index("reporter")
    assert sorted(row[column] for row in stored_frame.rows) == ["dcid:DEU", "dcid:USA"]
    (exclusion,) = [e for e in stored_trace.exclusions if e.reason == "beyond_limit"]
    assert exclusion.rows == 9
    assert exclusion.entities == {"reporter": ("dcid:FRA", "dcid:GBR", "dcid:JPN")}
    detail = stored_trace.applied_policy.detail
    assert sorted(detail["per_entity_coverage"]) == ["reporter=dcid:DEU", "reporter=dcid:USA"]
    assert detail["ranked_population"]["entity_count"] == 5


async def test_ascending_limit_keeps_the_lowest(seeded) -> None:
    store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2021",
        window_end="2023",
        ascending=True,
        limit=2,
    )

    stored_frame, _trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    column = stored_frame.columns.index("reporter")
    assert sorted(row[column] for row in stored_frame.rows) == ["dcid:FRA", "dcid:GBR"]


async def test_per_period_limit_keeps_the_top_of_each_period(seeded) -> None:
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True, limit=1
    )

    assert result.is_error is False
    assert result.structured_content["rows"] == 3
    assert result.structured_content["ranked_before_limit"] == 15
    assert "limit 1 per period (kept 3 of 15 ranked rows)" in result.content[0].text


async def test_a_limited_per_period_rank_stores_measured_values_and_no_ordinal_marker(
    seeded,
) -> None:
    """AAA leads 2020 and CCC leads 2021: the stored frame holds their
    amounts in their unit, the trace carries no ordinal marker, and the
    exclusion names only BBB, cut from both periods."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": reporter,
                "time_period": year,
                "value": value,
                "unit": "dcid:USDollar",
            }
            for reporter, year, value in (
                ("dcid:AAA", "2020", 100.0),
                ("dcid:BBB", "2020", 50.0),
                ("dcid:CCC", "2020", 10.0),
                ("dcid:AAA", "2021", 10.0),
                ("dcid:BBB", "2021", 20.0),
                ("dcid:CCC", "2021", 30.0),
            )
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)
    result = await rank(frame_ref=custom_ref, rank_by=("reporter",), per_period=True, limit=1)

    assert result.is_error is False
    stored_frame, stored_trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    columns = stored_frame.columns
    assert sorted(
        (row[columns.index("reporter")], row[columns.index("value")], row[columns.index("unit")])
        for row in stored_frame.rows
    ) == [("dcid:AAA", 100.0, "dcid:USDollar"), ("dcid:CCC", 30.0, "dcid:USDollar")]
    assert "derived" not in stored_trace.applied_policy.detail
    (exclusion,) = [e for e in stored_trace.exclusions if e.reason == "beyond_limit"]
    assert exclusion.rows == 4
    assert exclusion.entities == {"reporter": ("dcid:BBB",)}
    assert "beyond_limit (ranked beyond the limit): 4 row(s) [reporter=dcid:BBB]" in (
        result.content[0].text
    )


async def _stored(store: InMemoryMaterialisationStore, result: ToolResult) -> tuple[Frame, Trace]:
    """The frame and trace `result` names in `store`."""
    assert result.structured_content is not None
    stored = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    assert stored is not None
    frame, trace = stored
    assert isinstance(trace, Trace)
    return frame, trace


@pytest.mark.parametrize(
    ("window", "limit", "label", "interval"),
    [
        (("2021", "2023"), None, "2021/2023", ("2021", "2023")),
        (("2021", "2023"), 2, "2021/2023", ("2021", "2023")),
        (("2021", "2025"), None, "2021/2023", ("2021", "2023")),
        (("2023", "2024"), None, "2023", None),
    ],
)
async def test_a_collapsed_rank_labels_and_times_the_windows_observed_extent(
    seeded,
    monkeypatch: pytest.MonkeyPatch,
    window: tuple[str, str],
    limit: int | None,
    label: str,
    interval: tuple[str, str] | None,
) -> None:
    """AAA and BBB each have one 2023 row; CCC has 2021 and 2022. Every
    kept row carries the extent of the periods the window holds, and the
    trace records that extent as an interval, even when a limit cuts CCC.
    A window holding one period labels the point and records an
    observation."""
    store, _handles = seeded
    monkeypatch.setattr(rank_tool, "COVERAGE_FLOOR", 0)
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": reporter, "time_period": year, "value": value}
            for reporter, year, value in (
                ("dcid:AAA", "2023", 100.0),
                ("dcid:BBB", "2023", 90.0),
                ("dcid:CCC", "2021", 1.0),
                ("dcid:CCC", "2022", 1.0),
            )
        ],
    )
    custom_ref = await _seed_custom_frame(store, frame)

    result = await rank(
        frame_ref=custom_ref,
        rank_by=("reporter",),
        aggregation="sum",
        window_start=window[0],
        window_end=window[1],
        limit=limit,
    )

    assert result.is_error is False
    stored_frame, trace = await _stored(store, result)
    time_periods = stored_frame.columns.index("time_period")
    assert {row[time_periods] for row in stored_frame.rows} == {label}
    if interval is None:
        assert trace.time.kind == "observation"
    else:
        assert isinstance(trace.time, Interval)
        assert (trace.time.start, trace.time.end) == interval
        assert (
            f"time: rows are labelled {label}; real periods span {interval[0]} to {interval[1]}"
            in result.content[0].text
        )


async def test_a_rank_over_a_span_frame_keeps_every_entity_and_inherits_its_interval(
    seeded,
) -> None:
    """A growth result carries one span on every row and an interval in its
    trace. Ranking it keeps every entity, stamps the span unchanged, and
    passes the input's interval through, gaps included."""
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": reporter, "time_period": "2018/2022", "value": value}
            for reporter, value in (("dcid:AAA", 3.0), ("dcid:BBB", 2.0), ("dcid:CCC", 1.0))
        ],
    )
    span_interval = build_interval(start="2018", end="2022", observed=["2018", "2020", "2022"])
    digest = frame_digest(frame)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(
        ref,
        frame=frame,
        trace=_minimal_trace().model_copy(update={"time": span_interval}),
        writer="test",
    )

    for per_period in (False, True):
        result = await rank(
            frame_ref={"spec_id": ref.spec_id, "content_version": ref.content_version},
            rank_by=("reporter",),
            window_start="2018",
            window_end="2022",
            per_period=per_period,
            limit=3,
        )

        assert result.is_error is False
        stored_frame, trace = await _stored(store, result)
        columns = stored_frame.columns
        assert sorted(
            (row[columns.index("reporter")], row[columns.index("time_period")])
            for row in stored_frame.rows
        ) == [
            ("dcid:AAA", "2018/2022"),
            ("dcid:BBB", "2018/2022"),
            ("dcid:CCC", "2018/2022"),
        ]
        assert trace.time == span_interval


@pytest.mark.parametrize(
    ("window", "kept"),
    [
        pytest.param((None, None), ["dcid:AAA", "dcid:BBB"], id="no-window"),
        pytest.param(("2018", "2022"), ["dcid:AAA", "dcid:BBB"], id="the-growth-window"),
        pytest.param(("2015", "2024"), ["dcid:AAA", "dcid:BBB"], id="a-wider-window"),
        pytest.param(("2020", "2022"), None, id="a-window-cutting-the-span"),
    ],
)
async def test_growth_then_rank_by_the_change_keeps_every_entity_the_window_covers(
    seeded, window: tuple[str | None, str | None], kept: list[str] | None
) -> None:
    """Answers "which donors grew aid most from 2018 to 2022": growth over 2018-2022,
    then a limited rank of its result. A window holding the whole span
    keeps every entity, labelled 2018/2022; one that cuts the span holds no
    row, so the rank refuses."""
    store, _handles = seeded
    source = await _seed_custom_frame(
        store,
        Frame.build(
            role_keys=["reporter"],
            rows=[
                {
                    "stat_var": "dcid:X",
                    "reporter": reporter,
                    "time_period": str(year),
                    "value": base + step * (year - 2018),
                }
                for reporter, base, step in (("dcid:AAA", 10.0, 5.0), ("dcid:BBB", 10.0, 1.0))
                for year in range(2018, 2023)
            ],
        ),
    )
    grown = await growth(
        frame_ref=source,
        window_start="2018",
        window_end="2022",
        entity_by=("reporter",),
        change="absolute",
    )
    assert grown.is_error is False
    window_start, window_end = window

    result = await rank(
        frame_ref={
            "spec_id": grown.structured_content["spec_id"],
            "content_version": grown.structured_content["content_version"],
        },
        rank_by=("reporter",),
        window_start=window_start,
        window_end=window_end,
        limit=2,
    )

    if kept is None:
        assert result.structured_content["code"] == "stat_var_unresolved"
        return
    assert result.is_error is False
    stored_frame, _trace = await _stored(store, result)
    columns = stored_frame.columns
    assert [
        (row[columns.index("reporter")], row[columns.index("time_period")])
        for row in stored_frame.rows
    ] == [(reporter, "2018/2022") for reporter in kept]


async def test_a_limited_rank_stores_a_trace_sized_by_the_entities_it_keeps(seeded) -> None:
    """A top-1 rank over 2,000 entities stores the trace a top-1 rank over
    21 stores, give or take the digits of its counts: coverage names the
    one entity kept, and the ranked population is three numbers."""
    store, _handles = seeded

    async def stored_trace(entities: int) -> Trace:
        frame = Frame.build(
            role_keys=["reporter"],
            rows=[
                {
                    "stat_var": "dcid:X",
                    "reporter": f"dcid:E{entity:04d}",
                    "time_period": "2020",
                    "value": float(entity),
                }
                for entity in range(entities)
            ],
        )
        result = await rank(
            frame_ref=await _seed_custom_frame(store, frame), rank_by=("reporter",), limit=1
        )
        _frame, trace = await store.get(
            FrameRef(
                spec_id=result.structured_content["spec_id"],
                content_version=result.structured_content["content_version"],
            )
        )
        return trace

    small, large = await stored_trace(21), await stored_trace(2_000)

    assert len(large.model_dump_json()) - len(small.model_dump_json()) < 16
    assert large.applied_policy.detail["per_entity_coverage"] == {"reporter=dcid:E1999": 1.0}
    assert large.applied_policy.detail["ranked_population"] == {
        "entity_count": 2_000,
        "min_coverage_fraction": 1.0,
        "max_coverage_fraction": 1.0,
    }


# Success: one real analyst question end to end, through the registered tool ----


async def test_real_analyst_question_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Rank donors by total ODA disbursed, 2021-2023" via the live FastMCP
    `Client`, exercising the real registered tool."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "rank",
            {
                "frame_ref": handles.plain_single_unit,
                "rank_by": ["reporter"],
                "aggregation": "sum",
                "window_start": "2021",
                "window_end": "2023",
            },
        )

        assert result.is_error is False
        digest = result.content[0].text
        assert result.structured_content["spec_id"].startswith("ops_")
        assert result.structured_content["content_version"].startswith("ocv_")
        assert "#1 reporter=dcid:USA" in digest
        assert "166683000000" in digest

        limited = await client.call_tool(
            "rank",
            {
                "frame_ref": handles.plain_single_unit,
                "rank_by": ["reporter"],
                "aggregation": "sum",
                "limit": 1,
            },
            raise_on_error=False,
        )
        assert limited.is_error is False
        assert limited.structured_content["rows"] == 1


# Replay + execution-identity perturbation ---------------------------------------


async def test_replay_from_execution_record_content_is_bit_identical(seeded) -> None:
    store, handles = seeded

    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2021",
        window_end="2023",
    )
    assert result.is_error is False
    output_ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    stored_frame, stored_trace = await store.get(output_ref)

    canonical_bytes = stored_trace.execution_record.content.encode()
    assert recompute_op_spec_hash(canonical_bytes) == stored_trace.op_spec_hash

    parameters = json.loads(stored_trace.execution_record.content)["parameters"]
    input_frame, _input_trace = await store.get(FrameRef(**handles.plain_single_unit))
    replayed = run_rank(
        input_frame,
        rank_by=tuple(parameters["rank_by"]),
        aggregation=parameters["aggregation"],
        window_start=parameters["window_start"],
        window_end=parameters["window_end"],
        ascending=parameters["ascending"],
        pinned_unit=parameters["unit"],
        per_period=parameters["per_period"],
        # An unset `limit` is absent from the canonical op-spec.
        limit=parameters.get("limit"),
    )
    assert replayed.frame == stored_frame

    # Distinct from the recompute-from-parameters replay above. Proves the
    # *stored* frame_content_version was derived over the stored output
    # frame, not the input frame's bytes.
    assert derive_frame_content_version(frame=stored_frame) == stored_trace.frame_content_version

    recomputed_content_version = recompute_toolkit_content_version(
        op_spec_hash=stored_trace.op_spec_hash,
        input_content_versions=list(stored_trace.input_content_versions),
        execution_identity=stored_trace.execution_identity,
    )
    assert recomputed_content_version == stored_trace.content_version


async def test_execution_identity_perturbation_moves_the_content_version(seeded) -> None:
    store, handles = seeded

    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        aggregation="sum",
        window_start="2021",
        window_end="2023",
    )
    output_ref = FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )
    _stored_frame, stored_trace = await store.get(output_ref)

    bumped_identity = ExecutionIdentity(
        engine_version=stored_trace.execution_identity.engine_version,
        frame_grammar_version=stored_trace.execution_identity.frame_grammar_version,
        reduction_policy_version="9.9.9",
    )
    moved_content_version = mint_toolkit_content_version(
        op_spec_hash=stored_trace.op_spec_hash,
        input_content_versions=list(stored_trace.input_content_versions),
        execution_identity=bumped_identity,
    )
    assert moved_content_version != stored_trace.content_version


# Rolling overlay ----------------------------------------------------------------

_LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}
_ROOT = "spec_" + "c" * 32
_ANCHOR_KEY = "rak_" + "e" * 32


async def test_a_rolling_rank_computes_its_fixed_twins_rows_under_another_pair(seeded) -> None:
    store, handles = seeded
    arguments = {
        "frame_ref": handles.plain_single_unit,
        "rank_by": ("reporter",),
        "aggregation": "sum",
        "window_start": "2021",
        "window_end": "2023",
    }
    root = handles.plain_single_unit
    anchor = await anchored(root)
    wider = await anchored(root, await second_version(store, root))
    assert anchor.period == "2023"
    rolling = {
        "window_start": {**_LATEST, "offset": -2},
        "window_end": _LATEST,
        "population": anchor.populations,
    }

    fixed = await rank(**arguments)
    rolled = await rank(**arguments, rolling=rolling, anchor_key=anchor.anchor_key)
    rekeyed = await rank(**arguments, rolling=rolling, anchor_key=wider.anchor_key)

    fixed_frame, fixed_trace = await store.get(FrameRef(**_pair(fixed)))
    rolled_frame, rolled_trace = await store.get(FrameRef(**_pair(rolled)))
    assert rolled_frame.table.equals(fixed_frame.table)
    assert rolled_trace.frame_content_version == fixed_trace.frame_content_version
    assert rolled_trace.execution_identity == fixed_trace.execution_identity
    assert rolled_trace.applied_policy == fixed_trace.applied_policy
    assert _pair(rolled)["spec_id"] != _pair(fixed)["spec_id"]
    assert _pair(rolled)["content_version"] != _pair(fixed)["content_version"]
    assert json.loads(rolled_trace.execution_record.content)["parameters"]["rolling"] == rolling
    assert wider.anchor_key != anchor.anchor_key
    assert _pair(rekeyed) == _pair(rolled)


async def test_a_rolling_bound_on_a_rank_without_a_window_refuses(seeded) -> None:
    _store, handles = seeded
    result = await rank(
        frame_ref=handles.plain_single_unit,
        rank_by=("reporter",),
        aggregation="sum",
        rolling={"window_end": _LATEST, "population": [{"spec_id": _ROOT}]},
        anchor_key=_ANCHOR_KEY,
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


def _pair(result) -> dict[str, str]:
    content = result.structured_content
    return {"spec_id": content["spec_id"], "content_version": content["content_version"]}
