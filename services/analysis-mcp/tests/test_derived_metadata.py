"""The derived-ness marker (`frame_ref_resolution.DerivedMarker`) end to
end: producers (`rank`, `growth`, `share_of`, `compare`) write
`applied_policy.detail["derived"]` on their own output's trace, every verb
carries an input's marker onward, and the verbs that cannot read a rank
position refuse `derived_input_ordinal`. Also covers `Trace.time`, the
interval `aggregate` and `rank` record when they collapse periods.
"""

from __future__ import annotations

import dataclasses

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.artifact import Interval, Observation
from analysis_mcp.compare_tool import compare
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import (
    canonicalize_op_spec,
    mint_toolkit_content_version,
    op_spec_hash_for,
)
from analysis_mcp.join_tool import join
from analysis_mcp.op_spec import AggregateOpSpec, CategoryMatchPredicate, OrderedThresholdPredicate
from analysis_mcp.rank_tool import rank
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from dfo_frame_grammar.content_version import frame_sha256, mint_content_version
from fastmcp.tools import ToolResult


@pytest.fixture
async def seeded():
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)
    set_store(store)
    yield store, handles
    set_store(None)


def _pair(result: ToolResult) -> dict[str, str]:
    return {
        "spec_id": result.structured_content["spec_id"],
        "content_version": result.structured_content["content_version"],
    }


def _frame_ref(result: ToolResult) -> FrameRef:
    return FrameRef(**_pair(result))


async def _seed_handle(
    store: InMemoryMaterialisationStore,
    frame: Frame,
    *,
    suffix: str = "1",
    policy_detail: dict[str, object] | None = None,
    engine_version: str | None = None,
) -> dict[str, str]:
    """Register a hand-built `frame` under a deterministic pair, with a
    minimal but valid `Trace` -- for tests needing a coverage shape (each
    entity genuine at a different single year) none of `fixtures.py`'s own
    frame builders carries, and no combination of `filter` predicates can
    produce (predicates are column-wise and conjunctive; "USA only at 2021,
    DEU only at 2023" is an OR across two columns). Mirrors `fixtures.py`'s
    own `_fixture_trace`. `suffix` distinguishes two hand-built frames
    seeded into the same store; `policy_detail` is the trace's
    `applied_policy.detail`; `engine_version`, when given, replaces the
    current engine's in the trace's `execution_identity`.
    """
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(spec_id="spec_" + "0" * 32, content_version="cv_" + "0" * 32),
        group_by=(),
        aggregation="sum",
    )
    op_spec_hash = op_spec_hash_for(op_spec)
    execution_identity = current_execution_identity()
    if engine_version is not None:
        execution_identity = dataclasses.replace(execution_identity, engine_version=engine_version)
    input_cv = mint_content_version(
        frame_sha256=frame_sha256(frame),
        graph_revision="rev_test",
        source_revisions=[],
        facets=[],
    )
    trace = Trace(
        upstream_trace={"graph_revision": "rev_test"},
        frame_content_version=input_cv,
        op_spec_hash=op_spec_hash,
        execution_record=ExecutionRecord(content=canonicalize_op_spec(op_spec).decode()),
        applied_policy=AppliedPolicy(detail=policy_detail or {}),
        content_version=mint_toolkit_content_version(
            op_spec_hash=op_spec_hash,
            input_content_versions=[input_cv],
            execution_identity=execution_identity,
        ),
        input_content_versions=[input_cv],
        execution_identity=execution_identity,
    )
    ref = FrameRef(spec_id="spec_" + suffix * 32, content_version="cv_" + suffix * 32)
    await store.put(ref, frame=frame, trace=trace, writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


# `Trace.time`: an interval exactly when a call collapses periods ---------


async def test_aggregate_grouped_by_time_period_writes_no_derived_marker(seeded) -> None:
    """`time_period` in `group_by` means every output row's own `time_period`
    is the real grouping value, not a borrowed span label -- no marker."""
    store, handles = seeded
    result = await aggregate(
        frame_ref=handles.plain_single_unit,
        aggregation="sum",
        group_by=("reporter", "time_period"),
    )
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    assert "derived" not in trace.applied_policy.detail


async def test_time_kind_is_an_interval_exactly_when_aggregate_collapses_periods(
    seeded,
) -> None:
    """`aggregate(sum, group_by=('reporter',))` over `plain_single_unit`'s
    three years, 2021-2023, records that interval; the same sum grouped by
    `time_period` collapses nothing and records an observation.
    """
    store, handles = seeded

    spanning = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="sum", group_by=("reporter",)
    )
    assert spanning.is_error is False
    _frame, spanning_trace = await store.get(_frame_ref(spanning))
    assert isinstance(spanning_trace.time, Interval)
    assert spanning_trace.time.start == "2021"
    assert spanning_trace.time.end == "2023"
    assert spanning.structured_content["time"] == {
        "kind": "interval",
        "start": "2021",
        "end": "2023",
        "contiguous": True,
        "absent": [],
        "absent_count": 0,
    }

    genuine = await aggregate(
        frame_ref=handles.plain_single_unit,
        aggregation="sum",
        group_by=("reporter", "time_period"),
    )
    assert genuine.is_error is False
    _frame, genuine_trace = await store.get(_frame_ref(genuine))
    assert genuine_trace.time == Observation()
    assert genuine.structured_content["time"] == {"kind": "observation"}


async def test_aggregate_gapped_mean_reports_the_interval_as_not_contiguous(seeded) -> None:
    """A mean over two years with a missing year between them: the trace's
    own `Interval` still names the full 2021-2023 span (the union across the
    one collapsed group), but says plainly that the span is not contiguous
    and names 2022 as the year no collapsed row carried.
    """
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2021", "value": 10.0},
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2023", "value": 30.0},
        ],
    )
    ref = await _seed_handle(store, frame)
    result = await aggregate(frame_ref=ref, aggregation="mean", group_by=())
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    assert isinstance(trace.time, Interval)
    assert trace.time.start == "2021"
    assert trace.time.end == "2023"
    assert trace.time.contiguous is False
    assert trace.time.absent == ("2022",)
    assert trace.time.absent_count == 1


async def test_aggregate_wide_gap_reports_the_true_absent_count_past_the_sample(seeded) -> None:
    """A 2000-2020 span with only the two endpoints observed: 19 years are
    missing, but `Interval.absent` samples only the first 10 -- the trace
    and the digest must both say 19, not let the 10-entry sample read as
    exhaustive.
    """
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2000", "value": 10.0},
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 30.0},
        ],
    )
    ref = await _seed_handle(store, frame)
    result = await aggregate(frame_ref=ref, aggregation="mean", group_by=())
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    assert isinstance(trace.time, Interval)
    assert trace.time.contiguous is False
    assert trace.time.absent == tuple(str(year) for year in range(2001, 2011))
    assert trace.time.absent_count == 19
    assert "(19 total)" in result.content[0].text


# Two entities each genuine at a different single year: nothing collapses,
# and the result reads downstream. -----------------------------------------


async def test_rank_two_entities_at_different_single_years_reads_downstream(
    seeded,
) -> None:
    """USA's only row is a 2021 observation; DEU's only row is a 2023
    observation. Ranking both with no window and no aggregation labels both
    rows with the extent 2021/2023 and writes no derived marker, and
    `compare` reads the result.
    """
    store, _handles = seeded
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:OfficialDevelopmentAssistanceNet_Disbursement",
                "reporter": "dcid:USA",
                "time_period": "2021",
                "value": 47_805_000_000.0,
                "unit": "dcid:USDollar",
            },
            {
                "stat_var": "dcid:OfficialDevelopmentAssistanceNet_Disbursement",
                "reporter": "dcid:DEU",
                "time_period": "2023",
                "value": 36_680_000_000.0,
                "unit": "dcid:USDollar",
            },
        ],
    )
    ref = await _seed_handle(store, frame)

    ranked = await rank(frame_ref=ref, rank_by=("reporter",))
    assert ranked.is_error is False
    ranked_ref = _pair(ranked)
    _frame, trace = await store.get(_frame_ref(ranked))
    assert "derived" not in trace.applied_policy.detail

    result = await compare(
        frame_ref=ranked_ref,
        left_entity={"reporter": "dcid:USA"},
        right_frame_ref=ranked_ref,
        right_entity={"reporter": "dcid:USA"},
    )
    assert result.is_error is False


# A result computed before engine 0.3.0 carries a borrowed point label. ---


@pytest.mark.parametrize(
    ("engine_version", "policy_detail", "refused"),
    [
        pytest.param(
            "0.2.0",
            {"derived": {"spans_periods": True, "reason": "sum over 2018-2022"}},
            True,
            id="0.2.0-with-span-marker",
        ),
        pytest.param("0.2.0", {}, True, id="0.2.0-growth-without-marker"),
        pytest.param("0.1.4", {}, True, id="0.1.4"),
        pytest.param("0.3.0", {}, False, id="0.3.0"),
        pytest.param("0.10.0", {}, False, id="0.10.0-compared-as-a-version"),
    ],
)
async def test_a_result_from_an_engine_before_0_3_0_refuses_every_verb(
    seeded, engine_version: str, policy_detail: dict[str, object], refused: bool
) -> None:
    """Engines before 0.3.0 labelled a collapsed row with one point, so its
    `2022` may total 2018 to 2022, and a growth row with its end year without
    any marker. Resolution refuses every such input before a verb reads it,
    by engine version; a later engine's result is admitted."""
    store, _handles = seeded
    frame = _span_frame("dcid:X", "2022", {"dcid:USA": 10.0, "dcid:DEU": 5.0})
    ref = await _seed_handle(
        store, frame, policy_detail=policy_detail, engine_version=engine_version
    )

    result = await aggregate(frame_ref=ref, group_by=(), aggregation="sum")

    if refused:
        assert result.structured_content["code"] == "derived_input_unrecoverable"
        assert result.structured_content["fix"] == {"engine_version": engine_version}
    else:
        assert result.is_error is False


# A span reads downstream as its own string: the same span on both sides
# matches, any other pairing finds no common period. ---------------------


def _span_frame(stat_var: str, period: str, values: dict[str, float]) -> Frame:
    """One row per reporter, every row stamped `period` -- the shape an
    `aggregate` or `rank` over several years produces."""
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": stat_var,
                "reporter": reporter,
                "time_period": period,
                "value": value,
                "unit": "dcid:USDollar",
            }
            for reporter, value in values.items()
        ],
    )


_USA = {"reporter": "dcid:USA"}

# (left period, right period, refusal code or None when the sides match)
_SPAN_PAIRINGS = [
    pytest.param("2018/2022", "2018/2022", None, id="same-span"),
    pytest.param("2018/2022", "2019/2023", "disjoint", id="different-spans"),
    pytest.param("2018/2022", "2022", "disjoint", id="span-against-point"),
]
_REFUSAL_CODES = {
    "compare": "comparison_periods_disjoint",
    "share_of": "share_of_grain_mismatch",
}


async def _seed_pair(
    store: InMemoryMaterialisationStore, left_period: str, right_period: str
) -> tuple[dict[str, str], dict[str, str]]:
    left = _span_frame("dcid:X", left_period, {"dcid:USA": 10.0, "dcid:DEU": 5.0})
    right = _span_frame("dcid:Y", right_period, {"dcid:USA": 4.0, "dcid:DEU": 5.0})
    return (
        await _seed_handle(store, left, suffix="1"),
        await _seed_handle(store, right, suffix="2"),
    )


@pytest.mark.parametrize(("left_period", "right_period", "refusal"), _SPAN_PAIRINGS)
async def test_compare_matches_spans_on_their_string(
    seeded, left_period: str, right_period: str, refusal: str | None
) -> None:
    store, _handles = seeded
    left_ref, right_ref = await _seed_pair(store, left_period, right_period)

    result = await compare(
        frame_ref=left_ref,
        left_entity=_USA,
        right_frame_ref=right_ref,
        right_entity=_USA,
    )

    if refusal is None:
        assert result.is_error is False
        frame, _trace = await store.get(_frame_ref(result))
        assert frame.table.column("value").to_pylist() == [-6.0]
        assert frame.table.column("time_period").to_pylist() == [left_period]
    else:
        assert result.structured_content["code"] == _REFUSAL_CODES["compare"]
        assert result.structured_content["fix"]["left_periods"] == [left_period]
        assert result.structured_content["fix"]["right_periods"] == [right_period]


@pytest.mark.parametrize(("left_period", "right_period", "refusal"), _SPAN_PAIRINGS)
async def test_share_of_matches_spans_on_their_string(
    seeded, left_period: str, right_period: str, refusal: str | None
) -> None:
    store, _handles = seeded
    left_ref, right_ref = await _seed_pair(store, left_period, right_period)

    result = await share_of(
        frame_ref=left_ref,
        numerator_entity=_USA,
        denominator_frame_ref=right_ref,
        denominator_entity=_USA,
    )

    if refusal is None:
        assert result.is_error is False
        frame, _trace = await store.get(_frame_ref(result))
        assert frame.table.column("value").to_pylist() == [2.5]
        assert frame.table.column("time_period").to_pylist() == [left_period]
    else:
        assert result.structured_content["code"] == _REFUSAL_CODES["share_of"]
        assert result.structured_content["fix"]["numerator_periods"] == [left_period]
        assert result.structured_content["fix"]["denominator_periods"] == [right_period]
        assert result.structured_content["fix"]["matched_periods"] == []


@pytest.mark.parametrize(("left_period", "right_period", "refusal"), _SPAN_PAIRINGS)
async def test_join_matches_spans_on_their_string(
    seeded, left_period: str, right_period: str, refusal: str | None
) -> None:
    """A join of mismatched periods is an empty result, with both sides'
    rows reported as unmatched."""
    store, _handles = seeded
    left_ref, right_ref = await _seed_pair(store, left_period, right_period)

    result = await join(frame_ref=left_ref, right_frame_ref=right_ref)

    assert result.is_error is False
    assert result.structured_content["rows"] == (4 if refusal is None else 0)
    if refusal is not None:
        unmatched = {e["side"]: e["rows"] for e in result.structured_content["exclusions"]}
        assert unmatched == {"left": 2, "right": 2}


# The same chain through real verb output: a sum over several years reads
# against another sum over the same years. -------------------------------


async def test_aggregate_spans_chain_into_compare_share_of_and_join(seeded) -> None:
    store, handles = seeded
    oda_ref, gni_ref = handles.join_pair
    oda = _pair(await aggregate(frame_ref=oda_ref, aggregation="sum", group_by=("reporter",)))
    gni = _pair(await aggregate(frame_ref=gni_ref, aggregation="sum", group_by=("reporter",)))

    compared = await compare(
        frame_ref=oda, left_entity=_USA, right_frame_ref=gni, right_entity=_USA
    )
    shared = await share_of(
        frame_ref=oda, numerator_entity=_USA, denominator_frame_ref=gni, denominator_entity=_USA
    )
    joined = await join(frame_ref=oda, right_frame_ref=gni)

    assert [r.is_error for r in (compared, shared, joined)] == [False, False, False]
    for result in (compared, shared):
        frame, _trace = await store.get(_frame_ref(result))
        assert frame.table.column("time_period").to_pylist() == ["2022/2023"]
    assert joined.structured_content["rows"] == 6


# The two unconditional, informational-only markers -------------------------


async def test_growth_output_writes_the_dimensionless_derived_marker(seeded) -> None:
    store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap, window_start="2022", window_end="2023"
    )
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    derived = trace.applied_policy.detail["derived"]
    assert derived["dimensionless"] is True


async def test_share_of_output_writes_the_dimensionless_and_inherited_stat_var_marker(
    seeded,
) -> None:
    store, handles = seeded
    result = await share_of(
        frame_ref=handles.plain_single_unit,
        denominator_frame_ref=handles.plain_single_unit,
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={},
    )
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    derived = trace.applied_policy.detail["derived"]
    assert derived["dimensionless"] is True
    assert derived["inherited_stat_var"] is True


# Absolute growth: dimensionless is set only when inherited, never
# unconditionally -- the counterpart to relative growth's own unconditional
# marker above. -------------------------------------------------------------


async def test_absolute_growth_over_a_measured_input_does_not_mark_dimensionless(seeded) -> None:
    """Unlike relative growth (always dimensionless, tested above), absolute
    growth over an ordinary measured input keeps a real unit -- the marker
    must not claim dimensionless here."""
    store, handles = seeded
    result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2022",
        window_end="2023",
        change="absolute",
    )
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    derived = trace.applied_policy.detail.get("derived")
    assert derived is None or "dimensionless" not in derived


async def test_absolute_growth_over_a_dimensionless_input_inherits_the_marker(seeded) -> None:
    """The laundering regression, extended to an absolute growth hop: a
    dimensionless `share_of` output chained into `growth(change="absolute")`
    must carry the `dimensionless` marker onward, not drop it just because
    this producer's own unconditional case (relative mode) doesn't apply."""
    store, handles = seeded
    share = await share_of(
        frame_ref=handles.plain_single_unit,
        denominator_frame_ref=handles.plain_single_unit,
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={},
    )
    assert share.is_error is False

    result = await growth(
        frame_ref=_pair(share), window_start="2021", window_end="2023", change="absolute"
    )
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    assert trace.applied_policy.detail["derived"]["dimensionless"] is True


# `aggregate` and `filter` read a prior aggregate result -------------------


async def test_aggregate_spanning_periods_reaggregated_succeeds(seeded) -> None:
    _store, handles = seeded
    spanning = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="sum", group_by=("reporter",)
    )
    spanning_ref = _pair(spanning)

    result = await aggregate(frame_ref=spanning_ref, aggregation="sum")

    assert result.is_error is False


async def test_aggregate_spanning_periods_filtered_succeeds(seeded) -> None:
    _store, handles = seeded
    spanning = await aggregate(
        frame_ref=handles.plain_single_unit, aggregation="sum", group_by=("reporter",)
    )
    spanning_ref = _pair(spanning)

    result = await filter_verb(
        frame_ref=spanning_ref,
        predicates=(OrderedThresholdPredicate(column="value", op="gte", value=0.0),),
    )

    assert result.is_error is False


async def test_reaggregating_does_not_launder_an_ordinal_marker(seeded) -> None:
    """Two-hop laundering path: per-period rank -> aggregate -> rank. The
    `aggregate` hop must carry the inherited `ordinal` flag onward, or the
    final `rank` would order entities by a sum of rank positions."""
    store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False

    reaggregated = await aggregate(
        frame_ref=_pair(ranked), aggregation="sum", group_by=("reporter",)
    )
    assert reaggregated.is_error is False
    _frame, trace = await store.get(_frame_ref(reaggregated))
    assert trace.applied_policy.detail["derived"]["ordinal"] is True

    result = await rank(frame_ref=_pair(reaggregated), rank_by=("reporter",))
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"


# `growth`'s and `share_of`'s own markers (dimensionless) are informational -
# only -- they never block a downstream verb by themselves ------------------


async def test_growth_output_compared_against_a_real_currency_value_refuses_unit_unbound(
    seeded,
) -> None:
    """`growth`'s own `time_period` is a genuine window endpoint, so this
    refuses purely on the ordinary
    unit-resolution path -- `unit=None` (a growth rate is dimensionless)
    against the real side's `dcid:USDollar`. Growth's own rows leave `unit`
    unbound entirely, with no second resolved unit on that side to disagree
    with, so this raises `cross_frame_unit_unbound`.
    """
    _store, handles = seeded
    growth_result = await growth(
        frame_ref=handles.time_series_with_gap,
        window_start="2022",
        window_end="2023",
    )
    assert growth_result.is_error is False
    growth_ref = _pair(growth_result)

    result = await compare(
        frame_ref=growth_ref,
        right_frame_ref=handles.plain_single_unit,
        right_entity={"reporter": "dcid:USA"},
    )

    assert result.is_error is True
    assert result.structured_content["code"] == "cross_frame_unit_unbound"


async def test_growth_output_compared_against_another_growth_output_succeeds(seeded) -> None:
    """Both sides are `unit=None` for the same reason (a dimensionless growth
    rate). Only a genuine cross-unit pairing blocks this chain."""
    _store, handles = seeded
    first = await growth(
        frame_ref=handles.time_series_with_gap, window_start="2022", window_end="2023"
    )
    second = await growth(
        frame_ref=handles.time_series_with_gap, window_start="2022", window_end="2023"
    )
    assert first.is_error is False
    assert second.is_error is False

    result = await compare(
        frame_ref=_pair(first),
        right_frame_ref=_pair(second),
    )

    assert result.is_error is False
    assert "(delta 0)" in result.content[0].text


async def test_share_of_output_compared_against_itself_succeeds(seeded) -> None:
    """`share_of`'s own marker (`dimensionless`, `inherited_stat_var`) is
    informational only, like `growth`'s, so chaining into `compare` is not
    blocked."""
    _store, handles = seeded
    result = await share_of(
        frame_ref=handles.plain_single_unit,
        denominator_frame_ref=handles.plain_single_unit,
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={},
    )
    assert result.is_error is False
    share_ref = _pair(result)

    compared = await compare(frame_ref=share_ref, right_frame_ref=share_ref)

    assert compared.is_error is False


async def test_compare_across_different_stat_vars_writes_inherited_stat_var(seeded) -> None:
    """The output row's own `stat_var` always carries the left side's
    (`engine/compare.py`), so a GNI-minus-ODA delta ships labelled as ODA --
    `compare` must flag that with its own `inherited_stat_var` marker.
    """
    store, handles = seeded
    oda_ref, gni_ref = handles.join_pair

    result = await compare(
        frame_ref=oda_ref,
        left_entity={"reporter": "dcid:USA"},
        right_frame_ref=gni_ref,
        right_entity={"reporter": "dcid:USA"},
    )

    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    assert trace.applied_policy.detail["derived"]["inherited_stat_var"] is True


async def test_compare_across_same_stat_var_writes_no_inherited_stat_var(seeded) -> None:
    """Both sides resolve to the same `stat_var` -- the output row's
    borrowed label is honest, so no marker is written."""
    store, handles = seeded

    result = await compare(
        frame_ref=handles.plain_single_unit,
        left_entity={"reporter": "dcid:USA"},
        right_frame_ref=handles.plain_single_unit,
        right_entity={"reporter": "dcid:GBR"},
    )

    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    assert "derived" not in trace.applied_policy.detail


# Exclusion accounting: entity narrowing and unmatched periods ------------------


async def test_entity_narrowed_compare_reports_the_dropped_rows_and_entities(seeded) -> None:
    """`left_entity`/`right_entity` each narrow `plain_single_unit`'s five
    reporters down to one -- the other four's own rows on each side must be
    accounted for in the trace and `structuredContent`, but the digest's own
    `exclusions:` block must stay silent about them, since the caller's own
    `entity` binding is what dropped them."""
    store, handles = seeded

    result = await compare(
        frame_ref=handles.plain_single_unit,
        left_entity={"reporter": "dcid:USA"},
        right_frame_ref=handles.plain_single_unit,
        right_entity={"reporter": "dcid:GBR"},
    )

    assert result.is_error is False
    assert "exclusions:" not in result.content[0].text
    by_side = {
        exclusion["side"]: exclusion
        for exclusion in result.structured_content["exclusions"]
        if exclusion["reason"] == "entity_not_selected"
    }
    assert by_side["left"]["rows"] == 12
    assert by_side["right"]["rows"] == 12
    _frame, trace = await store.get(_frame_ref(result))
    assert any(exclusion.reason == "entity_not_selected" for exclusion in trace.exclusions)


async def test_entity_narrowed_share_of_reports_the_dropped_rows_and_entities(seeded) -> None:
    _store, handles = seeded

    result = await share_of(
        frame_ref=handles.plain_single_unit,
        numerator_entity={"reporter": "dcid:USA"},
        denominator_frame_ref=handles.plain_single_unit,
        denominator_entity={"reporter": "dcid:GBR"},
    )

    assert result.is_error is False
    assert "exclusions:" not in result.content[0].text
    by_side = {
        exclusion["side"]: exclusion
        for exclusion in result.structured_content["exclusions"]
        if exclusion["reason"] == "entity_not_selected"
    }
    assert by_side["numerator"]["rows"] == 12
    assert by_side["denominator"]["rows"] == 12


async def test_partial_period_compare_reports_unmatched_periods_but_not_a_second_digest_line(
    seeded,
) -> None:
    """Left offers 2021-2023, right offers 2022-2024: one period on each
    side (left's 2021, right's 2024) falls outside the intersection.
    `compare`'s own coverage line already states the ratio; the exclusion
    accounting must not add a second digest line repeating it."""
    store, _handles = seeded
    left = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": year, "value": 1.0}
            for year in ("2021", "2022", "2023")
        ],
    )
    right = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": year, "value": 2.0}
            for year in ("2022", "2023", "2024")
        ],
    )
    left_ref = await _seed_handle(store, left, suffix="1")
    right_ref = await _seed_handle(store, right, suffix="2")

    result = await compare(frame_ref=left_ref, right_frame_ref=right_ref)

    assert result.is_error is False
    digest = result.content[0].text
    assert "exclusions:" not in digest
    assert "period overlap: left 66.7%, right 66.7%" in digest
    by_side = {
        exclusion["side"]: exclusion
        for exclusion in result.structured_content["exclusions"]
        if exclusion["reason"] == "period_unmatched"
    }
    assert by_side["left"]["rows"] == 1
    assert by_side["right"]["rows"] == 1


# The ordinal marker: a per-period rank's own output value is a rank
# position, not a measured value -- share_of and a relative growth both
# refuse over it, at the tool as well as in the plan. ------------------


async def test_per_period_rank_writes_the_ordinal_marker(seeded) -> None:
    store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    _frame, trace = await store.get(_frame_ref(ranked))
    assert trace.applied_policy.detail["derived"]["ordinal"] is True


async def test_collapsing_rank_writes_no_ordinal_marker(seeded) -> None:
    """`per_period=False` (the default) is unaffected -- no `ordinal` key at
    all, not `ordinal: False`, since `derived_marker_detail` omits an unset
    flag rather than writing it `False`. Filtered down to one year first,
    so the collapsing rank writes no other flag either, keeping this test
    about `ordinal` alone."""
    store, handles = seeded
    filtered = await filter_verb(
        frame_ref=handles.plain_single_unit,
        predicates=(CategoryMatchPredicate(column="time_period", values=("2023",)),),
    )
    assert filtered.is_error is False
    ranked = await rank(frame_ref=_pair(filtered), rank_by=("reporter",))
    assert ranked.is_error is False
    _frame, trace = await store.get(_frame_ref(ranked))
    assert trace.applied_policy.detail.get("derived") is None


async def test_relative_growth_over_a_rank_series_is_refused_as_ordinal(seeded) -> None:
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = _pair(ranked)

    result = await growth(
        frame_ref=ranked_ref, window_start="2021", window_end="2023", entity_by=("reporter",)
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
    assert result.structured_content["retry_mode"] == "correct_and_retry"


async def test_absolute_growth_over_a_rank_series_computes_in_places(seeded) -> None:
    """`value_end - value_start` over rank positions, unit `None`, the
    digest stating the move in words rather than a bare signed number."""
    store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = _pair(ranked)

    result = await growth(
        frame_ref=ranked_ref,
        window_start="2021",
        window_end="2023",
        entity_by=("reporter",),
        change="absolute",
    )
    assert result.is_error is False
    _frame, trace = await store.get(_frame_ref(result))
    assert trace.applied_policy.detail["derived"]["ordinal"] is True
    digest = result.content[0].text
    assert "places" in digest
    assert "unit: absolute change (places)" in digest


async def test_share_of_over_a_rank_series_refuses_as_ordinal_on_the_numerator(seeded) -> None:
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = _pair(ranked)

    result = await share_of(
        frame_ref=ranked_ref,
        denominator_frame_ref=handles.plain_single_unit,
        numerator_entity={},
        denominator_entity={},
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
    assert "numerator" in result.structured_content["fix"]["reason"]


async def test_share_of_over_a_rank_series_refuses_as_ordinal_on_the_denominator(seeded) -> None:
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = _pair(ranked)

    result = await share_of(
        frame_ref=handles.plain_single_unit,
        denominator_frame_ref=ranked_ref,
        numerator_entity={},
        denominator_entity={},
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
    assert "denominator" in result.structured_content["fix"]["reason"]


async def test_a_rank_over_a_filtered_rank_series_refuses_as_ordinal(seeded) -> None:
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    filtered = await filter_verb(
        frame_ref=_pair(ranked),
        predicates=(CategoryMatchPredicate(column="time_period", values=("2023",)),),
    )
    assert filtered.is_error is False

    result = await rank(frame_ref=_pair(filtered), rank_by=("reporter",), limit=1)

    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"


async def test_a_rank_over_a_change_in_rank_position_refuses_as_ordinal(seeded) -> None:
    """A growth over a rank series carries `ordinal` onward, and a rank
    reads a change in places as it reads the positions."""
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    places = await growth(
        frame_ref=_pair(ranked),
        window_start="2021",
        window_end="2023",
        entity_by=("reporter",),
        change="absolute",
    )
    assert places.is_error is False

    result = await rank(frame_ref=_pair(places), rank_by=("reporter",))

    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"


async def test_ordinal_marker_survives_a_filter_hop(seeded) -> None:
    """`filter` must carry the inherited `ordinal` flag onward -- it is a
    pure passthrough (see `derived_marker_detail`'s own docstring) -- or a
    relative growth two hops downstream would wrongly compute a percentage
    over rank positions instead of refusing."""
    store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = _pair(ranked)

    filtered = await filter_verb(
        frame_ref=ranked_ref,
        predicates=(CategoryMatchPredicate(column="reporter", values=("dcid:USA",)),),
    )
    assert filtered.is_error is False
    _frame, filtered_trace = await store.get(_frame_ref(filtered))
    assert filtered_trace.applied_policy.detail["derived"]["ordinal"] is True
    filtered_ref = _pair(filtered)

    result = await growth(
        frame_ref=filtered_ref, window_start="2021", window_end="2023", entity_by=("reporter",)
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
