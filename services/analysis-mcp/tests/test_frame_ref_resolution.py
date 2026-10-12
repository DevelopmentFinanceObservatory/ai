"""`frame_ref_resolution`'s own pure functions: the derived-ness marker
reader (`read_derived_marker`, `DerivedMarker`), the period admission rule
(`_period_refusal`) and what an admitted input says about its periods
(`AdmittedFrame`).
End-to-end producer/consumer coverage lives in `test_derived_metadata.py`;
this file is each function's own contract in isolation.

Also covers this module's own import-time side effect: registering
`derive_frame_content_version` into `common.materialisation.content_version`'s
shared registry, mirroring `analysis_mcp.trace`'s own `register_trace_codec`
call and its own `test_trace_round_trips_through_the_common_trace_codec`.
"""

from __future__ import annotations

import time
from datetime import date

import pytest
from analysis_mcp.artifact import Interval, Observation, Time
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.fixtures import live_shaped_trace
from analysis_mcp.frame_ref_resolution import (
    AdmittedFrame,
    DerivedMarker,
    _one_span,
    _period_refusal,
    build_interval,
    derive_frame_content_version,
    derived_marker_detail,
    read_derived_marker,
)
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.tool_errors import ToolErrorCode
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, get_content_version_deriver
from dfo_frame_grammar import Frame

_SNAPSHOT_REF = {"spec_id": "spec_" + "a" * 32, "content_version": "cv_" + "b" * 32}


def _trace_with_detail(detail: dict[str, object]) -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(**_SNAPSHOT_REF), group_by=("reporter",), aggregation="sum"
    )
    return Trace(
        upstream_trace={"graph_revision": "rev1", "resolved_stat_vars": []},
        frame_content_version="cv_" + "d" * 32,
        op_spec_hash=op_spec_hash_for(op_spec),
        execution_record=ExecutionRecord(kind="op_spec_canonical", content="{}"),
        applied_policy=AppliedPolicy(detail=detail),
        content_version="ocv_" + "c" * 32,
        input_content_versions=[_SNAPSHOT_REF["content_version"]],
        execution_identity=current_execution_identity(),
    )


def test_read_derived_marker_is_none_when_the_input_carries_no_marker() -> None:
    assert read_derived_marker(_trace_with_detail({})) is None


def test_read_derived_marker_parses_every_field() -> None:
    trace = _trace_with_detail(
        {
            "derived": {
                "dimensionless": True,
                "inherited_stat_var": True,
                "ordinal": True,
                "reason": "rank positions",
            }
        }
    )
    assert read_derived_marker(trace) == DerivedMarker(
        dimensionless=True, inherited_stat_var=True, ordinal=True, reason="rank positions"
    )


def test_read_derived_marker_defaults_absent_fields_to_false() -> None:
    marker = read_derived_marker(_trace_with_detail({"derived": {"dimensionless": True}}))
    assert marker == DerivedMarker(dimensionless=True)


def test_read_derived_marker_reads_a_stored_trace_carrying_retired_keys() -> None:
    """A trace stored before `spans_periods`/`rank_relabelled` retired still
    reads, keeping the flags this toolkit still knows."""
    trace = _trace_with_detail(
        {
            "derived": {
                "spans_periods": True,
                "rank_relabelled": True,
                "inherited_stat_var": True,
                "reason": "spans 2021..2023",
            }
        }
    )
    assert read_derived_marker(trace) == DerivedMarker(
        inherited_stat_var=True, reason="spans 2021..2023"
    )


def test_derived_marker_detail_writes_no_retired_key_from_an_old_input() -> None:
    """An input stored with the retired span flags carries only the flags
    `DerivedMarker` still names onward."""
    inherited = read_derived_marker(
        _trace_with_detail({"derived": {"spans_periods": True, "rank_relabelled": True}})
    )
    assert derived_marker_detail(inherited=inherited) is None
    assert derived_marker_detail(inherited=inherited, ordinal=True) == {"ordinal": True}


@pytest.mark.parametrize(
    "periods",
    [
        pytest.param(["2018", "2019", "2022"], id="points-at-one-grain"),
        pytest.param(["2018-01", "2018-03"], id="month-points"),
        pytest.param(["2018/2022"], id="one-span"),
        pytest.param(["2018-01/2018-06"], id="one-month-span"),
        pytest.param(["2022"], id="one-point"),
        pytest.param([], id="no-rows"),
    ],
)
def test_period_admission_admits_points_at_one_grain_or_one_span(periods: list[str]) -> None:
    assert _period_refusal(periods) is None


@pytest.mark.parametrize(
    ("periods", "grains", "named"),
    [
        pytest.param(["2021-01", "2020"], ["year", "month"], ["2020", "2021-01"], id="grain-mix"),
        pytest.param(
            ["2018/2022", "2023", "2024"], ["year"], ["2018/2022", "2023"], id="span-beside-points"
        ),
        pytest.param(
            ["2019/2023", "2018/2022"], ["year"], ["2018/2022", "2019/2023"], id="two-spans"
        ),
        pytest.param(
            ["2018/2022", "2018-01"],
            ["year", "month"],
            ["2018/2022", "2018-01"],
            id="grain-and-span",
        ),
    ],
)
def test_period_admission_refuses_a_mix_naming_the_periods(
    periods: list[str], grains: list[str], named: list[str]
) -> None:
    refusal = _period_refusal(periods)
    assert refusal is not None
    assert refusal.code is ToolErrorCode.MIXED_TIME_GRAIN
    assert refusal.fix == {"grains": grains, "periods": named}


def _frame_with_periods(*periods: str) -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": period, "value": 1.0}
            for period in periods
        ],
    )


@pytest.mark.parametrize(
    ("periods", "span", "time"),
    [
        pytest.param(
            ("2018/2022",),
            "2018/2022",
            Interval(start="2018", end="2022", contiguous=None),
            id="one-span",
        ),
        pytest.param(
            ("2018-01/2018-06",),
            "2018-01/2018-06",
            Interval(start="2018-01", end="2018-06", contiguous=None),
            id="one-month-span",
        ),
        pytest.param(("2022",), None, Observation(), id="one-point"),
        pytest.param(("2018", "2022"), None, Observation(), id="points"),
    ],
)
def test_an_admitted_raw_frame_reads_its_time_from_its_cells(
    periods: tuple[str, ...], span: str | None, time: Time
) -> None:
    """A raw materialisation's trace records no time kind, so the one span
    its cells carry is an interval whose contiguity no trace states."""
    frame = _frame_with_periods(*periods)
    admitted = AdmittedFrame(
        frame=frame, trace=live_shaped_trace("raw", frame), span=_one_span(list(periods))
    )
    assert (admitted.span, admitted.time) == (span, time)


def test_build_interval_at_the_day_grain_terminal_date_does_not_overflow() -> None:
    """A single-day span ending exactly at `date.max` (`9999-12-31`) must
    not raise -- the day branch checks the loop bound before incrementing,
    so it never tries to construct a date past the terminal one."""
    interval = build_interval(start="9999-12-31", end="9999-12-31", observed=["9999-12-31"])
    assert interval.contiguous is True
    assert interval.absent == ()
    assert interval.absent_count == 0


def test_build_interval_day_grain_gap_reaching_the_terminal_date_does_not_overflow() -> None:
    interval = build_interval(start="9999-12-29", end="9999-12-31", observed=["9999-12-29"])
    assert interval.contiguous is False
    assert interval.absent == ("9999-12-30", "9999-12-31")
    assert interval.absent_count == 2


def test_build_interval_degrades_to_unknown_contiguity_on_a_calendar_invalid_day() -> None:
    """`dfo_frame_grammar.grammar` deliberately admits a shape-valid but
    calendar-invalid day (e.g. 30 February) from a source -- `build_interval`
    must not crash computing `date.fromisoformat` on it, and must not claim
    a contiguity it can't determine. `contiguous` degrades to `None`
    (unknown), with `absent`/`absent_count` left at their defaults, rather
    than refusing or raising."""
    interval = build_interval(
        start="2024-02-28", end="2024-02-30", observed=["2024-02-28", "2024-02-30"]
    )
    assert interval.contiguous is None
    assert interval.absent == ()
    assert interval.absent_count == 0


def test_build_interval_degrades_to_unknown_contiguity_on_year_zero() -> None:
    """Year `0000` is shape-valid per the frame grammar's own regex but
    outside `datetime.date`'s representable range (`MINYEAR == 1`) -- the
    identical degrade as a calendar-invalid day."""
    interval = build_interval(
        start="0000-01-01", end="0000-01-02", observed=["0000-01-01", "0000-01-02"]
    )
    assert interval.contiguous is None
    assert interval.absent == ()
    assert interval.absent_count == 0


def test_build_interval_over_a_wide_day_grain_span_stays_fast() -> None:
    """A day-grain span covering the entire representable date range, with
    only its two endpoints observed: `build_interval` must answer by
    walking at most `len(observed) + _ABSENT_CAP` periods, not by
    materialising the ~3.6 million days between them."""
    total_days = date(9999, 12, 31).toordinal() - date(1, 1, 1).toordinal() + 1
    started = time.monotonic()
    interval = build_interval(
        start="0001-01-01", end="9999-12-31", observed=["0001-01-01", "9999-12-31"]
    )
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert interval.contiguous is False
    assert interval.absent == tuple(f"0001-01-{day:02d}" for day in range(2, 12))
    assert interval.absent_count == total_days - 2


def test_derive_frame_content_version_is_registered_as_the_stores_deriver() -> None:
    """Importing this module (transitively, as every verb tool already does) registers
    `derive_frame_content_version` into `common.materialisation.content_version`'s
    registry -- `common.materialisation.build_store` resolves this same registered
    deriver when a service asks for `verify_content_version_on_read`, regardless of
    which service's own lifespan builds the store."""
    registered = get_content_version_deriver()
    assert registered is not None

    frame = Frame.build(
        role_keys=[],
        rows=[
            {
                "stat_var": "dcid:OdaDisbursedAmount",
                "time_period": "2020",
                "value": 100.0,
                "unit": None,
            }
        ],
    )

    assert registered(frame) == derive_frame_content_version(frame=frame)
