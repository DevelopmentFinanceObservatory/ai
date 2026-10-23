"""The `share_of` tool end to end: every refusal path, one real analyst
question through the registered tool, and the replay + execution-identity-
perturbation determinism proofs.
"""

from __future__ import annotations

import json
import secrets

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.engine.share_of import run_share_of
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.frame_ref_resolution import derive_frame_content_version
from analysis_mcp.identity import (
    mint_toolkit_content_version,
    op_spec_hash_for,
    recompute_op_spec_hash,
    recompute_toolkit_content_version,
)
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.rank_tool import rank
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from fastmcp import Client


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
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


# Refusals ----------------------------------------------------------------------


async def test_op_spec_invalid_refuses(seeded) -> None:
    _store, _handles = seeded
    result = await share_of(frame_ref="not-a-valid-pair")
    assert result.is_error is True
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_unregistered_numerator_pair_refuses_frame_not_held(seeded) -> None:
    _store, _handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await share_of(frame_ref=never_registered)
    assert result.structured_content["code"] == "frame_not_held"


async def test_unregistered_denominator_pair_refuses_frame_not_held(seeded) -> None:
    _store, handles = seeded
    never_registered = {"spec_id": "spec_" + "0" * 32, "content_version": "cv_" + "0" * 32}
    result = await share_of(frame_ref=handles.join_pair[0], denominator_frame_ref=never_registered)
    assert result.structured_content["code"] == "frame_not_held"


async def test_grain_mismatch_refuses_when_no_period_is_shared(seeded) -> None:
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2021", "value": 1.0},
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2022", "value": 2.0},
        ],
    )
    denominator = Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:Y", "reporter": "dcid:USA", "time_period": "2020", "value": 10.0}],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(frame_ref=numerator_ref, denominator_frame_ref=denominator_ref)
    assert result.structured_content["code"] == "share_of_grain_mismatch"
    assert result.structured_content["fix"] == {
        "numerator_periods": ["2021", "2022"],
        "denominator_periods": ["2020"],
        "key": {},
    }


async def test_a_key_lagging_a_pooled_denominator_is_left_out_and_declared(seeded) -> None:
    """Each donor's share of a 2027 total: France reports to 2026 only, so
    it is left out and declared as `quotient_of` declares such a key, and
    the United States and Germany keep their shares."""
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["donor"],
        rows=[
            {"stat_var": "dcid:X", "donor": donor, "time_period": period, "value": 10.0}
            for donor, periods in (
                ("dcid:USA", ("2026", "2027")),
                ("dcid:DEU", ("2027",)),
                ("dcid:FRA", ("2025", "2026")),
            )
            for period in periods
        ],
    )
    denominator = Frame.build(
        role_keys=["donor"],
        rows=[{"stat_var": "dcid:Y", "donor": "dcid:ALL", "time_period": "2027", "value": 100.0}],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(
        frame_ref=numerator_ref, denominator_frame_ref=denominator_ref, numerator_by=["donor"]
    )

    assert result.is_error is False
    assert result.structured_content["dropped_key_count"] == 1
    assert result.structured_content["rows"] == 2
    _frame, trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    assert trace.applied_policy.detail["keys_without_shared_period_count"] == 1
    (step,) = trace.steps
    assert step.left_out[0].model_dump() == {
        "reason": "period_unmatched",
        "count": 1,
        "of": "entities",
    }
    assert (
        "numerator keys left out: 1 sharing no period with the denominator (donor=dcid:FRA)"
        in result.content[0].text
    )


async def test_both_sides_empty_names_the_real_fault_not_two_matching_empty_lists(seeded) -> None:
    """A typo'd role value on *both* `numerator_entity` and
    `denominator_entity` makes both sides' own period sets the empty tuple.
    The generic grain-mismatch message reads "The denominator's periods []
    do not match the numerator's periods []" -- two identical empty lists
    framed as disagreeing, which never names the actual fault: neither
    side's own entity binding matched a single row. This dedicated branch
    must say that instead."""
    _store, handles = seeded
    result = await share_of(
        frame_ref=handles.plain_single_unit,
        numerator_entity={"reporter": "dcid:TYPO_NUMERATOR"},
        denominator_entity={"reporter": "dcid:TYPO_DENOMINATOR"},
    )
    assert result.structured_content["code"] == "share_of_grain_mismatch"
    message = result.content[0].text
    assert "do not match" not in message
    assert "entity binding" in message
    assert "matched" in message


async def test_partial_overlap_divides_as_the_filtered_pair_does(seeded) -> None:
    """A pair sharing some periods divides over those periods with no
    narrowing first: the shares equal those of the same call after `filter`
    narrows the numerator to the shared period, and the period left out is
    recorded on the numerator."""
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["reporter"],
        rows=[
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 1.0},
            {"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2021", "value": 2.0},
        ],
    )
    denominator = Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:Y", "reporter": "dcid:USA", "time_period": "2020", "value": 10.0}],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    direct = await share_of(frame_ref=numerator_ref, denominator_frame_ref=denominator_ref)
    narrowed = await filter_verb(
        frame_ref=numerator_ref,
        predicates=({"kind": "category_match", "column": "time_period", "values": ["2020"]},),
    )
    narrowed_ref = {
        "spec_id": narrowed.structured_content["spec_id"],
        "content_version": narrowed.structured_content["content_version"],
    }
    filtered = await share_of(frame_ref=narrowed_ref, denominator_frame_ref=denominator_ref)

    assert direct.is_error is False
    assert filtered.is_error is False
    (direct_frame, direct_trace), (filtered_frame, _filtered_trace) = [
        await store.get(
            FrameRef(
                spec_id=result.structured_content["spec_id"],
                content_version=result.structured_content["content_version"],
            )
        )
        for result in (direct, filtered)
    ]
    assert direct_frame.table.column("value").to_pylist() == [10.0]
    assert direct_frame.table.column("value").equals(filtered_frame.table.column("value"))
    assert [
        (exclusion.reason, exclusion.side, exclusion.rows) for exclusion in direct_trace.exclusions
    ] == [("period_unmatched", "numerator", 1)]


async def test_zero_denominator_refuses_share_of_zero_denominator(seeded) -> None:
    """A zero-summed denominator is retryable (a different denominator
    source can sum to nonzero), so it gets its own code and
    `correct_and_retry` -- distinct from a genuine overflow, which is
    `non_finite_result`'s `do_not_retry`."""
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:X", "reporter": "dcid:USA", "time_period": "2020", "value": 5.0}],
    )
    denominator = Frame.build(
        role_keys=["reporter"],
        rows=[{"stat_var": "dcid:Y", "reporter": "dcid:USA", "time_period": "2020", "value": 0.0}],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(frame_ref=numerator_ref, denominator_frame_ref=denominator_ref)
    assert result.structured_content["code"] == "share_of_zero_denominator"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"]["time_period"] == "2020"


async def test_heterogeneous_unit_frame_refuses(seeded) -> None:
    _store, handles = seeded
    result = await share_of(
        frame_ref=handles.multi_unit,
        numerator_entity={"reporter": "dcid:USA"},
    )
    assert result.structured_content["code"] == "unit_heterogeneous"


async def test_unknown_entity_column_refuses_role_ambiguous(seeded) -> None:
    _store, handles = seeded
    result = await share_of(
        frame_ref=handles.join_pair[0],
        denominator_frame_ref=handles.join_pair[1],
        numerator_entity={"donor": "dcid:USA"},
        denominator_entity={"reporter": "dcid:USA"},
    )
    assert result.structured_content["code"] == "role_ambiguous"
    assert result.structured_content["fix"]["columns"] == ["donor"]


async def test_numerator_over_a_per_period_rank_refuses_derived_input_ordinal(seeded) -> None:
    """A share over a rank position is meaningless on either side -- the
    tool-level mirror of the plan-level rule. Unlike growth, share_of has
    no retry that fixes this."""
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = {
        "spec_id": ranked.structured_content["spec_id"],
        "content_version": ranked.structured_content["content_version"],
    }
    result = await share_of(
        frame_ref=ranked_ref,
        denominator_frame_ref=handles.plain_single_unit,
        numerator_entity={},
        denominator_entity={},
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
    assert "numerator" in result.structured_content["fix"]["reason"]


async def test_denominator_over_a_per_period_rank_refuses_derived_input_ordinal(seeded) -> None:
    _store, handles = seeded
    ranked = await rank(frame_ref=handles.plain_single_unit, rank_by=("reporter",), per_period=True)
    assert ranked.is_error is False
    ranked_ref = {
        "spec_id": ranked.structured_content["spec_id"],
        "content_version": ranked.structured_content["content_version"],
    }
    result = await share_of(
        frame_ref=handles.plain_single_unit,
        denominator_frame_ref=ranked_ref,
        numerator_entity={},
        denominator_entity={},
    )
    assert result.is_error is True
    assert result.structured_content["code"] == "derived_input_ordinal"
    assert "denominator" in result.structured_content["fix"]["reason"]


async def test_cross_frame_unit_mismatch_refuses(seeded) -> None:
    """Each side resolves cleanly to its own single unit, but the two units
    differ -- distinct from `unit_heterogeneous`, which fires within one
    frame."""
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 5.0,
                "unit": "dcid:USDollar",
            }
        ],
    )
    denominator = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:Y",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
                "unit": "dcid:Percent",
            }
        ],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(frame_ref=numerator_ref, denominator_frame_ref=denominator_ref)
    assert result.structured_content["code"] == "cross_frame_unit_mismatch"
    assert result.structured_content["fix"]["numerator"] == "dcid:USDollar"
    assert result.structured_content["fix"]["denominator"] == "dcid:Percent"


def _sourced_frame(
    *, stat_var: str, provenance: str, value: float, years: range = range(2024, 2025)
) -> Frame:
    return Frame.build(
        role_keys=["debtor"],
        rows=[
            {
                "stat_var": stat_var,
                "debtor": "dcid:country/NGA",
                "time_period": str(year),
                "value": value,
                "unit": "dcid:USDollar",
                "provenance": provenance,
            }
            for year in years
        ],
    )


_IDS = "dcid:provenance/WorldBankInternationalDebtStatistics"
_WDI = "dcid:dc/base/WorldDevelopmentIndicators"


async def test_two_measures_from_two_sources_divide_with_both_sources_on_the_trace(
    seeded,
) -> None:
    """Nigeria's debt service from IDS (1970-2024) as a share of its GDP
    from WDI (1960-2025): two StatVars, so their differing provenance warns
    in full, the share divides over the 55 years both report, the 11 GDP
    years outside them are recorded on the denominator, and the trace
    records each side's own provenance. Every row holds the share in per
    cent under the operation's own StatVar, which the trace describes, and
    the digest states the first and latest share straight after its
    header."""
    store, _handles = seeded
    debt_service, gdp = 5.0, 200.0
    numerator_ref = await _seed_custom_frame(
        store,
        _sourced_frame(
            stat_var="dcid:DebtService",
            provenance=_IDS,
            value=debt_service,
            years=range(1970, 2025),
        ),
    )
    denominator_ref = await _seed_custom_frame(
        store,
        _sourced_frame(
            stat_var="dcid:Amount_EconomicActivity_GDP",
            provenance=_WDI,
            value=gdp,
            years=range(1960, 2026),
        ),
    )

    result = await share_of(frame_ref=numerator_ref, denominator_frame_ref=denominator_ref)

    assert result.is_error is False
    assert result.structured_content["rows"] == 55
    digest = result.content[0].text
    assert digest.startswith(
        "share_of: 55 matched period(s), in per cent (Percent)\n\n"
        "first 1970: 2.500%; latest 2024: 2.500%\n\n"
    )
    assert "periods left out: only the denominator reports 1960" in digest
    assert "(11 total)" in digest
    assert "coverage numerator 100.0%, denominator 83.3%" in digest
    warning_lines = [
        line for line in result.content[0].text.splitlines() if "facet 'provenance'" in line
    ]
    assert len(warning_lines) == 1
    assert _IDS in warning_lines[0]
    assert _WDI in warning_lines[0]
    assert warning_lines[0].endswith("each keeps its own")
    frame, trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    assert frame.table.column("time_period").to_pylist() == [str(y) for y in range(1970, 2025)]
    own_stat_var = f"dcid:{result.structured_content['spec_id']}"
    assert frame.table.column("value").to_pylist() == [debt_service / gdp * 100] * 55
    assert {row[frame.columns.index("unit")] for row in frame.rows} == {"Percent"}
    assert set(frame.table.column("stat_var").to_pylist()) == {own_stat_var}
    assert trace.applied_policy.resolved_unit == "Percent"
    assert trace.applied_policy.detail["numerator_resolved_unit"] == "dcid:USDollar"
    assert trace.applied_policy.detail["derived"] == {"dimensionless": True}
    assert trace.applied_policy.policy_name == "share_of_intersect_periods"
    assert trace.applied_policy.detail["matched_periods"] == [str(y) for y in range(1970, 2025)]
    assert trace.applied_policy.realised_coverage_fraction == pytest.approx(55 / 66)
    assert [
        (exclusion.reason, exclusion.side, exclusion.rows) for exclusion in trace.exclusions
    ] == [("period_unmatched", "denominator", 11)]
    assert trace.applied_policy.resolved_facets == {"provenance": _IDS}
    assert trace.applied_policy.detail["denominator_resolved_facets"] == {"provenance": _WDI}


async def test_one_measure_from_two_sources_refuses_naming_the_measure(seeded) -> None:
    store, _handles = seeded
    numerator_ref = await _seed_custom_frame(
        store, _sourced_frame(stat_var="dcid:DebtService", provenance=_IDS, value=5.0)
    )
    denominator_ref = await _seed_custom_frame(
        store, _sourced_frame(stat_var="dcid:DebtService", provenance=_WDI, value=20.0)
    )

    result = await share_of(frame_ref=numerator_ref, denominator_frame_ref=denominator_ref)

    assert result.structured_content["code"] == "cross_frame_facet_mismatch"
    assert result.structured_content["fix"] == {
        "axis": "provenance",
        "numerator": _IDS,
        "denominator": _WDI,
    }
    assert "dcid:DebtService" in result.structured_content["message"]


async def test_denominator_by_key_not_in_numerator_by_refuses_op_spec_invalid(seeded) -> None:
    _store, handles = seeded
    result = await share_of(
        frame_ref=handles.join_pair[0],
        numerator_by=["donor"],
        denominator_by={"sector": "area"},
    )
    assert result.structured_content["code"] == "op_spec_invalid"


async def test_denominator_by_partial_mapping_refuses_share_of_key_unknown(seeded) -> None:
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["donor", "sector"],
        rows=[
            {
                "stat_var": "dcid:X",
                "donor": "dcid:USA",
                "sector": "dcid:Health",
                "time_period": "2020",
                "value": 10.0,
            }
        ],
    )
    denominator = Frame.build(
        role_keys=["provider"],
        rows=[
            {
                "stat_var": "dcid:Y",
                "provider": "dcid:USA",
                "time_period": "2020",
                "value": 100.0,
            }
        ],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(
        frame_ref=numerator_ref,
        denominator_frame_ref=denominator_ref,
        numerator_by=["donor", "sector"],
        denominator_by={"donor": "provider"},
    )
    assert result.structured_content["code"] == "share_of_key_unknown"
    assert result.structured_content["fix"]["column"] == "sector"


async def test_a_key_with_no_denominator_group_is_left_out_and_declared(seeded) -> None:
    """France has no denominator group, so it is left out and declared as
    `quotient_of` declares such a key, and the United States keeps its
    share."""
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["donor"],
        rows=[
            {"stat_var": "dcid:X", "donor": "dcid:USA", "time_period": "2020", "value": 10.0},
            {"stat_var": "dcid:X", "donor": "dcid:FRA", "time_period": "2020", "value": 5.0},
        ],
    )
    denominator = Frame.build(
        role_keys=["provider"],
        rows=[
            {"stat_var": "dcid:Y", "provider": "dcid:USA", "time_period": "2020", "value": 100.0}
        ],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(
        frame_ref=numerator_ref,
        denominator_frame_ref=denominator_ref,
        numerator_by=["donor"],
        denominator_by={"donor": "provider"},
    )

    assert result.is_error is False
    assert result.structured_content["dropped_key_count"] == 1
    assert result.structured_content["rows"] == 1
    _frame, trace = await store.get(
        FrameRef(
            spec_id=result.structured_content["spec_id"],
            content_version=result.structured_content["content_version"],
        )
    )
    assert trace.applied_policy.detail["unmatched_key_count"] == 1
    assert [exclusion.reason for exclusion in trace.exclusions] == ["unmatched_key"]
    (step,) = trace.steps
    assert [entry.model_dump() for entry in step.left_out] == [
        {"reason": "unmatched_key", "count": 1, "of": "entities"}
    ]
    text = result.content[0].text
    assert "numerator keys left out: 1 with no denominator match (donor=dcid:FRA)" in text
    assert "exclusions:" not in text


async def test_no_key_with_a_denominator_group_refuses_share_of_key_unmatched(seeded) -> None:
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["donor"],
        rows=[
            {"stat_var": "dcid:X", "donor": "dcid:FRA", "time_period": "2020", "value": 5.0},
        ],
    )
    denominator = Frame.build(
        role_keys=["provider"],
        rows=[
            {"stat_var": "dcid:Y", "provider": "dcid:USA", "time_period": "2020", "value": 100.0}
        ],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(
        frame_ref=numerator_ref,
        denominator_frame_ref=denominator_ref,
        numerator_by=["donor"],
        denominator_by={"donor": "provider"},
    )
    assert result.structured_content["code"] == "share_of_key_unmatched"
    assert result.structured_content["fix"] == {
        "numerator_keys": [{"donor": "dcid:FRA"}],
        "denominator_keys": [{"provider": "dcid:USA"}],
        "numerator_key_count": 1,
        "denominator_key_count": 1,
    }


async def test_cross_frame_unit_unbound_refuses_and_is_retryable(seeded) -> None:
    """The denominator's own rows leave `unit` unbound entirely; the
    numerator resolves to a real unit. Distinct from
    `cross_frame_unit_mismatch` -- and, unlike an unanticipated
    `internal_error`, this is a `correct_and_retry` refusal: retry with a
    denominator_frame_ref whose own rows actually declare a unit."""
    store, _handles = seeded
    numerator = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:X",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 5.0,
                "unit": "dcid:USDollar",
            }
        ],
    )
    denominator = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": "dcid:Y",
                "reporter": "dcid:USA",
                "time_period": "2020",
                "value": 10.0,
            }
        ],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    result = await share_of(frame_ref=numerator_ref, denominator_frame_ref=denominator_ref)
    assert result.structured_content["code"] == "cross_frame_unit_unbound"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"]["unbound_side"] == "denominator"
    assert result.structured_content["fix"]["bound_side"] == "numerator"
    assert result.structured_content["fix"]["bound_unit"] == "dcid:USDollar"


# Success: one real analyst question end to end, through the registered tool ----


async def test_real_analyst_question_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "What was the USA's ODA disbursed as a share of its GNI, 2022-2023"
    via the live FastMCP `Client`, exercising the real registered tool."""
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "share_of",
            {
                "frame_ref": handles.join_pair[0],
                "denominator_frame_ref": handles.join_pair[1],
                "numerator_entity": {"reporter": "dcid:USA"},
                "denominator_entity": {"reporter": "dcid:USA"},
            },
        )

        assert result.is_error is False
        digest = result.content[0].text
        assert result.structured_content["spec_id"].startswith("ops_")
        assert result.structured_content["content_version"].startswith("ocv_")
        assert "2 matched period(s)" in digest
        assert "2022:" in digest and "2023:" in digest


async def test_keyed_share_of_end_to_end_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Rank donors by their share of ODA going to health" -- the keyed
    reading, one row per donor per period, through the live FastMCP
    `Client`."""
    store = InMemoryMaterialisationStore()

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    numerator = Frame.build(
        role_keys=["donor"],
        rows=[
            {"stat_var": "dcid:X", "donor": "dcid:USA", "time_period": "2020", "value": 10.0},
            {"stat_var": "dcid:X", "donor": "dcid:GBR", "time_period": "2020", "value": 20.0},
        ],
    )
    denominator = Frame.build(
        role_keys=["provider"],
        rows=[
            {"stat_var": "dcid:Y", "provider": "dcid:USA", "time_period": "2020", "value": 100.0},
            {"stat_var": "dcid:Y", "provider": "dcid:GBR", "time_period": "2020", "value": 200.0},
        ],
    )
    numerator_ref = await _seed_custom_frame(store, numerator)
    denominator_ref = await _seed_custom_frame(store, denominator)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            "share_of",
            {
                "frame_ref": numerator_ref,
                "denominator_frame_ref": denominator_ref,
                "numerator_by": ["donor"],
                "denominator_by": {"donor": "provider"},
            },
        )

        assert result.is_error is False
        assert result.structured_content["role_keys"] == ["donor"]
        assert result.structured_content["rows"] == 2
        digest = result.content[0].text
        assert "2 key(s)" in digest
        assert "dcid:USA" in digest and "dcid:GBR" in digest


# Two slices of one frame_ref, through the registered tool ----------------------


async def test_two_slices_of_one_frame_ref(seeded) -> None:
    _store, handles = seeded
    result = await share_of(
        frame_ref=handles.join_pair[0],
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={"reporter": "dcid:GBR"},
    )
    assert result.is_error is False


# Replay + execution-identity perturbation ---------------------------------------


async def test_replay_from_execution_record_content_is_bit_identical(seeded) -> None:
    store, handles = seeded

    result = await share_of(
        frame_ref=handles.join_pair[0],
        denominator_frame_ref=handles.join_pair[1],
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={"reporter": "dcid:USA"},
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
    numerator_frame, _numerator_trace = await store.get(FrameRef(**handles.join_pair[0]))
    denominator_frame, _denominator_trace = await store.get(FrameRef(**handles.join_pair[1]))
    replayed = run_share_of(
        numerator_frame,
        denominator_frame,
        stat_var=f"dcid:{stored_trace.op_spec_hash}",
        numerator_entity=parameters["numerator_entity"],
        denominator_entity=parameters["denominator_entity"],
        numerator_pinned_unit=parameters["numerator_unit"],
        denominator_pinned_unit=parameters["denominator_unit"],
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

    result = await share_of(
        frame_ref=handles.join_pair[0],
        denominator_frame_ref=handles.join_pair[1],
        numerator_entity={"reporter": "dcid:USA"},
        denominator_entity={"reporter": "dcid:USA"},
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
