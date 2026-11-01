"""The `quotient_of` tool end to end: one per-head question through the
registered tool, the side rules that read a trace, every refusal, a
quotient's composed unit through the other verbs, the declared dropped-key
count, and determinism. `test_derived_metadata.py` holds the marker rule.
"""

from __future__ import annotations

import pytest
from analysis_mcp import server as server_module
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec, OrderedThresholdPredicate
from analysis_mcp.quotient_of_tool import quotient_of
from analysis_mcp.rank_tool import rank
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.sum_of_tool import sum_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame, unit_cell_to_json
from fastmcp import Client
from fastmcp.tools import ToolResult
from runner_replay import ReplayedRunner, frame_digest

GDP = "dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"
POPULATION = "dcid:Count_Person"
AID = "dcid:OfficialDevelopmentAssistanceNet_Disbursement"
USD = "dcid:USDollar"
KEN = "dcid:country/KEN"
GTM = "dcid:country/GTM"
USD_PER_PERSON = {"numerator": USD, "denominator": {"stat_var": POPULATION}}


@pytest.fixture
async def store():
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


def _minimal_trace() -> Trace:
    op_spec = AggregateOpSpec(
        frame_ref=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
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


async def _seed(
    store: InMemoryMaterialisationStore,
    stat_var: str,
    values: dict[str, dict[str, float]],
    *,
    column: str = "observationAbout",
    unit: str | None = None,
) -> dict[str, str]:
    """Store one series per value of `column`: `values` maps each to its periods."""
    frame = Frame.build(
        role_keys=[column],
        rows=[
            {
                "stat_var": stat_var,
                column: entity,
                "time_period": period,
                "value": value,
                "unit": unit,
            }
            for entity, by_period in values.items()
            for period, value in by_period.items()
        ],
    )
    digest = frame_digest(frame)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_minimal_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _pair(result: ToolResult) -> dict[str, str]:
    assert result.is_error is False, result.content[0].text
    return {
        "spec_id": result.structured_content["spec_id"],
        "content_version": result.structured_content["content_version"],
    }


async def _stored(store: InMemoryMaterialisationStore, result: ToolResult) -> tuple[Frame, Trace]:
    return await store.get(FrameRef(**_pair(result)))


def _input_pairs(runner: ReplayedRunner) -> dict[str, dict[str, str]]:
    """The frame pair of each input of the last tree the tools sent the runner."""
    return {
        name: {"spec_id": node["spec_id"], "content_version": node["content_version"]}
        for name, node in runner.requests[-1]["tree"]["inputs"].items()
    }


async def _units(store: InMemoryMaterialisationStore, result: ToolResult) -> list[object]:
    """The distinct `unit` cells of `result`'s stored frame, as JSON."""
    frame, _trace = await _stored(store, result)
    position = frame.columns.index("unit")
    units = [unit_cell_to_json(row[position]) for row in frame.rows]
    return [unit for index, unit in enumerate(units) if unit not in units[:index]]


async def _aid_and_population(store: InMemoryMaterialisationStore) -> tuple[dict, dict]:
    """Aid to Kenya and Guatemala by `recipient`, in US dollars, and both
    countries' unit-less population by `observationAbout`, 2020 to 2022."""
    aid = await _seed(
        store,
        AID,
        {
            KEN: {"2020": 100.0, "2021": 120.0, "2022": 150.0},
            GTM: {"2020": 40.0, "2021": 30.0, "2022": 60.0},
        },
        column="recipient",
        unit=USD,
    )
    population = await _seed(
        store,
        POPULATION,
        {
            KEN: {"2020": 50.0, "2021": 50.0, "2022": 50.0},
            GTM: {"2020": 20.0, "2021": 20.0, "2022": 20.0},
        },
    )
    return aid, population


def _per_head(aid: dict, population: dict, place: str) -> dict[str, object]:
    return {
        "frame_ref": aid,
        "denominator_frame_ref": population,
        "numerator_entity": {"recipient": place},
        "denominator_entity": {"observationAbout": place},
    }


# One per-head question through the registered tool -----------------------------


@pytest.mark.parametrize(
    ("tool", "form"),
    [("quotient_of", {}), ("share_of", {"operation": "quotient"})],
    ids=["quotient_of", "share_of"],
)
async def test_gdp_per_head_through_the_registered_tool(
    monkeypatch: pytest.MonkeyPatch,
    store: InMemoryMaterialisationStore,
    runner: ReplayedRunner,
    tool: str,
    form: dict[str, str],
) -> None:
    """Guatemala's GDP in US dollars (2019-2022) over its unit-less population
    (2020-2023), through the live FastMCP `Client`, as `quotient_of` and as
    `share_of`'s quotient form: one series over 2020-2022 in the unit the two
    sides compose, under the operation's own ref."""
    gdp = await _seed(
        store, GDP, {GTM: {"2019": 770.0, "2020": 780.0, "2021": 860.0, "2022": 950.0}}, unit=USD
    )
    population = await _seed(
        store, POPULATION, {GTM: {"2020": 10.0, "2021": 20.0, "2022": 25.0, "2023": 30.0}}
    )

    async def _stub_build_store(_settings: object) -> InMemoryMaterialisationStore:
        return store

    monkeypatch.setattr(server_module, "_build_store", _stub_build_store)

    async with Client(server_module.mcp) as client:
        result = await client.call_tool(
            tool, {"frame_ref": gdp, "denominator_frame_ref": population, **form}
        )

    assert result.is_error is False
    structured = result.structured_content
    assert structured["rows"] == 3
    assert structured["role_keys"] == []
    assert structured["stat_var_count"] == 1
    assert structured["dropped_key_count"] == 0
    # The reply names the frame the runner stored for this call, computed from
    # the two seeded frames, and the period each side alone reports.
    assert (structured["spec_id"], structured["content_version"]) in runner.stored.values()
    assert _input_pairs(runner) == {"data": gdp, "other": population}
    assert [
        (exclusion["reason"], exclusion["rows"], exclusion["side"])
        for exclusion in structured["exclusions"]
    ] == [
        ("period_unmatched", 1, "numerator"),
        ("period_unmatched", 1, "denominator"),
    ]

    digest = result.content[0].text
    assert digest.startswith("quotient_of: 3 matched period(s), in dcid:USDollar per Count_Person")
    assert (
        f"stat_var dcid:{structured['spec_id']}: {GDP} (unit: dcid:USDollar) divided by "
        f"{POPULATION} (unit: (none declared))"
    ) in digest
    assert (
        "periods left out: only the numerator reports 2019; only the denominator reports 2023"
    ) in digest
    assert "  2020: 780 / 10 = 78\n  2021: 860 / 20 = 43\n  2022: 950 / 25 = 38" in digest
    assert "%" not in digest.split("divided over")[0]
    assert "coverage numerator 75.0%, denominator 75.0%" in digest
    assert "divided over 2020 to 2022" in digest
    # Ahead of the quotients, so a reader keeping only the digest's start has it.
    assert digest.index("periods left out") < digest.index("quotient(s), numerator")


# Side rules that read the trace -----------------------------------------------


async def test_a_share_as_a_side_refuses_as_dimensionless(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, population = await _aid_and_population(store)
    share = await share_of(frame_ref=aid, numerator_entity={"recipient": KEN})
    held = len(store._entries)

    result = await quotient_of(
        frame_ref=_pair(share),
        denominator_frame_ref=population,
        denominator_entity={"observationAbout": KEN},
    )

    assert result.structured_content["code"] == "quotient_of_side_unit_unsupported"
    assert result.structured_content["fix"] == {"side": "numerator", "reason": "dimensionless"}
    assert "numerator" in result.structured_content["message"]
    assert len(store._entries) == held


async def test_a_quotient_as_a_side_refuses_as_composed(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, population = await _aid_and_population(store)
    per_head = await quotient_of(**_per_head(aid, population, KEN))

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=_pair(per_head),
        numerator_entity={"recipient": KEN},
    )

    assert result.structured_content["code"] == "quotient_of_side_unit_unsupported"
    assert result.structured_content["fix"] == {
        "side": "denominator",
        "reason": "composed",
        "unit": USD_PER_PERSON,
    }
    assert "'dcid:USDollar per Count_Person'" in result.structured_content["message"]


async def test_a_unit_less_sum_as_a_side_refuses_as_an_operation_stat_var(
    store: InMemoryMaterialisationStore,
) -> None:
    """A `sum_of` of two unit-less counts declares no unit and carries the
    sum's own ref as its StatVar, which nothing names. The refusal survives
    a `filter` hop, whose trace names only the filter, and stores nothing."""
    aid, population = await _aid_and_population(store)
    entity = {"observationAbout": KEN}
    total = await sum_of(frame_ref=population, left_entity=entity, right_entity=entity)
    filtered = await filter_verb(frame_ref=_pair(total))
    held = len(store._entries)

    for denominator in (_pair(total), _pair(filtered)):
        result = await quotient_of(
            frame_ref=aid, denominator_frame_ref=denominator, numerator_entity={"recipient": KEN}
        )

        assert result.structured_content["code"] == "quotient_of_side_unit_unsupported"
        assert result.structured_content["fix"] == {
            "side": "denominator",
            "reason": "operation_stat_var",
            "stat_var": f"dcid:{total.structured_content['spec_id']}",
        }
    assert len(store._entries) == held


async def test_a_sum_that_declares_a_unit_divides(store: InMemoryMaterialisationStore) -> None:
    """The operation-ref rule reads a side's resolved unit: a `sum_of` result
    in US dollars is a part by its unit, whatever its StatVar."""
    aid, population = await _aid_and_population(store)
    entity = {"recipient": KEN}
    total = await sum_of(frame_ref=aid, left_entity=entity, right_entity=entity)

    result = await quotient_of(
        frame_ref=_pair(total),
        denominator_frame_ref=population,
        denominator_entity={"observationAbout": KEN},
    )

    assert await _units(store, result) == [USD_PER_PERSON]


async def test_share_ofs_quotient_form_is_quotient_of(store: InMemoryMaterialisationStore) -> None:
    aid, population = await _aid_and_population(store)
    arguments = _per_head(aid, population, KEN)

    divided = await quotient_of(**arguments)
    shared = await share_of(**arguments, operation="quotient")

    assert shared.structured_content == divided.structured_content
    assert shared.content == divided.content


async def test_a_unit_less_summed_family_is_the_denominator_part(
    store: InMemoryMaterialisationStore,
) -> None:
    family = "dcid:spec_" + "f" * 32
    aid, _population = await _aid_and_population(store)
    summed = await _seed(store, family, {KEN: {"2020": 50.0, "2021": 60.0, "2022": 75.0}})

    result = await quotient_of(
        frame_ref=aid, denominator_frame_ref=summed, numerator_entity={"recipient": KEN}
    )

    assert await _units(store, result) == [{"numerator": USD, "denominator": {"stat_var": family}}]


@pytest.mark.parametrize("side", ["numerator", "denominator"])
async def test_a_rank_position_as_a_side_refuses_as_ordinal(
    store: InMemoryMaterialisationStore, side: str
) -> None:
    aid, population = await _aid_and_population(store)
    ranked = _pair(await rank(frame_ref=aid, rank_by=("recipient",), per_period=True))
    arguments = (
        {"frame_ref": ranked, "denominator_frame_ref": population}
        if side == "numerator"
        else {"frame_ref": aid, "denominator_frame_ref": ranked}
    )

    result = await quotient_of(**arguments)

    assert result.structured_content["code"] == "derived_input_ordinal"
    assert side in result.structured_content["fix"]["reason"]


# Refusals with nothing left to divide ------------------------------------------


async def test_a_denominator_that_is_zero_in_every_matched_cell_refuses(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, _population = await _aid_and_population(store)
    empty = await _seed(store, POPULATION, {KEN: {"2020": 0.0, "2021": 0.0, "2023": 5.0}})
    held = len(store._entries)

    result = await quotient_of(**_per_head(aid, empty, KEN))

    assert result.structured_content["code"] == "quotient_of_zero_denominator"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"] == {"time_periods": ["2020", "2021"], "cell_count": 2}
    assert "['2020', '2021']" in result.structured_content["message"]
    assert len(store._entries) == held


async def test_a_keyed_call_whose_every_key_has_a_zero_denominator_refuses(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, _population = await _aid_and_population(store)
    empty = await _seed(store, POPULATION, {KEN: {"2020": 0.0}, GTM: {"2021": 0.0, "2022": 0.0}})

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=empty,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert result.structured_content["code"] == "quotient_of_zero_denominator"
    assert result.structured_content["fix"] == {
        "time_periods": ["2020", "2021", "2022"],
        "cell_count": 3,
    }


async def test_a_keyed_call_over_an_empty_numerator_slice_refuses_as_disjoint_periods(
    store: InMemoryMaterialisationStore,
) -> None:
    """No row is bound to Guatemala, so the numerator holds no key. The
    refusal is the un-keyed call's, since no `denominator_by` could pair a
    key the numerator does not hold."""
    frame = Frame.build(
        role_keys=["donor", "recipient"],
        rows=[
            {
                "stat_var": AID,
                "donor": donor,
                "recipient": KEN,
                "time_period": "2020",
                "value": 10.0,
                "unit": USD,
            }
            for donor in ("dcid:country/FRA", "dcid:country/DEU")
        ],
    )
    aid_ref = FrameRef(spec_id="spec_" + "d" * 32, content_version="cv_" + "d" * 32)
    await store.put(aid_ref, frame=frame, trace=_minimal_trace(), writer="test")
    aid = {"spec_id": aid_ref.spec_id, "content_version": aid_ref.content_version}
    population = await _seed(store, POPULATION, {KEN: {"2020": 50.0, "2021": 51.0}})
    arguments = {
        "frame_ref": aid,
        "denominator_frame_ref": population,
        "numerator_entity": {"recipient": GTM},
    }

    keyed = await quotient_of(**arguments, numerator_by=["donor"])
    un_keyed = await quotient_of(**arguments)

    assert keyed.structured_content["code"] == "comparison_periods_disjoint"
    assert keyed.structured_content["fix"] == un_keyed.structured_content["fix"]
    assert keyed.structured_content["fix"]["numerator_periods"] == []
    assert keyed.structured_content["fix"]["denominator_periods"] == ["2020", "2021"]


async def test_a_keyed_call_over_an_empty_denominator_slice_refuses_as_disjoint_periods(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, population = await _aid_and_population(store)
    arguments = {
        "frame_ref": aid,
        "denominator_frame_ref": population,
        "denominator_entity": {"observationAbout": "dcid:country/NOWHERE"},
    }

    keyed = await quotient_of(
        **arguments, numerator_by=["recipient"], denominator_by={"recipient": "observationAbout"}
    )
    un_keyed = await quotient_of(**arguments)

    assert keyed.structured_content["code"] == "comparison_periods_disjoint"
    assert keyed.structured_content["fix"] == un_keyed.structured_content["fix"]
    assert keyed.structured_content["fix"]["numerator_periods"] == ["2020", "2021", "2022"]
    assert keyed.structured_content["fix"]["denominator_periods"] == []


async def test_keys_that_pair_with_no_denominator_group_refuse(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, _population = await _aid_and_population(store)
    elsewhere = await _seed(store, POPULATION, {"dcid:country/UGA": {"2020": 45.0}})
    held = len(store._entries)

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=elsewhere,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert result.structured_content["code"] == "quotient_of_no_key_matched"
    assert result.structured_content["fix"] == {
        "numerator_keys": [{"recipient": GTM}, {"recipient": KEN}],
        "denominator_keys": [{"observationAbout": "dcid:country/UGA"}],
        "numerator_key_count": 2,
        "denominator_key_count": 1,
    }
    message = result.structured_content["message"]
    assert "denominator_by must map each numerator_by column" in message
    assert KEN in message
    assert "dcid:country/UGA" in message
    assert len(store._entries) == held


async def test_sides_sharing_no_period_refuse_under_their_own_labels(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, _population = await _aid_and_population(store)
    later = await _seed(store, POPULATION, {KEN: {"2023": 51.0, "2024": 52.0}})

    result = await quotient_of(**_per_head(aid, later, KEN))

    assert result.structured_content["code"] == "comparison_periods_disjoint"
    assert result.structured_content["fix"] == {
        "numerator_periods": ["2020", "2021", "2022"],
        "denominator_periods": ["2023", "2024"],
    }
    assert "Numerator side has" in result.structured_content["message"]
    assert "denominator side has" in result.structured_content["message"]


async def test_a_quotient_that_overflows_refuses_non_finite_result(
    store: InMemoryMaterialisationStore,
) -> None:
    huge = await _seed(store, GDP, {KEN: {"2020": 1e308}}, unit=USD)
    tiny = await _seed(store, POPULATION, {KEN: {"2020": 1e-308}})

    result = await quotient_of(frame_ref=huge, denominator_frame_ref=tiny)

    assert result.structured_content["code"] == "non_finite_result"
    assert result.structured_content["fix"] == {
        "group_key": {"time_period": "2020"},
        "aggregation": "quotient_of",
    }


@pytest.mark.parametrize(
    ("numerator_by", "denominator_by"),
    [
        pytest.param(["recipient"], {"donor": "observationAbout"}, id="unknown_key"),
        pytest.param(["donor", "recipient"], {"recipient": "observationAbout"}, id="partial"),
    ],
)
async def test_a_denominator_by_that_does_not_pair_numerator_by_refuses_at_the_op_spec(
    store: InMemoryMaterialisationStore, numerator_by: list[str], denominator_by: dict[str, str]
) -> None:
    aid, population = await _aid_and_population(store)

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=population,
        numerator_by=numerator_by,
        denominator_by=denominator_by,
    )

    assert result.structured_content["code"] == "op_spec_invalid"


# A quotient's composed unit through the other verbs ------------------------------


async def test_two_quotients_compare_only_in_the_same_composed_unit(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, population = await _aid_and_population(store)
    gdp = await _seed(store, GDP, {KEN: {"2020": 1000.0, "2021": 1100.0, "2022": 1200.0}}, unit=USD)
    kenya = _pair(await quotient_of(**_per_head(aid, population, KEN)))
    guatemala = _pair(await quotient_of(**_per_head(aid, population, GTM)))
    of_gdp = _pair(await quotient_of(**_per_head(aid, gdp, KEN)))

    same_unit = await compare(frame_ref=kenya, right_frame_ref=guatemala)
    other_unit = await compare(frame_ref=kenya, right_frame_ref=of_gdp)

    assert await _units(store, same_unit) == [USD_PER_PERSON]
    assert other_unit.structured_content["code"] == "cross_frame_unit_mismatch"
    assert other_unit.structured_content["fix"] == {
        "left": USD_PER_PERSON,
        "right": {"numerator": USD, "denominator": USD},
    }
    assert "quotient_of" in other_unit.structured_content["message"]


async def test_a_keyed_quotient_reads_through_every_single_frame_verb(
    store: InMemoryMaterialisationStore,
) -> None:
    """`rank`, an absolute `growth`, a mean `aggregate` and a `filter` on
    `value` keep the composed unit as they keep a unit ref; a relative
    `growth` is a rate and drops it."""
    aid, population = await _aid_and_population(store)
    per_head = _pair(
        await quotient_of(
            frame_ref=aid,
            denominator_frame_ref=population,
            numerator_by=["recipient"],
            denominator_by={"recipient": "observationAbout"},
        )
    )
    window = {"window_start": "2020", "window_end": "2022", "entity_by": ("recipient",)}

    ranked = await rank(frame_ref=per_head, rank_by=("recipient",), aggregation="mean")
    grown = await growth(frame_ref=per_head, **window)
    changed = await growth(frame_ref=per_head, change="absolute", **window)
    averaged = await aggregate(frame_ref=per_head, aggregation="mean", group_by=("recipient",))
    filtered = await filter_verb(
        frame_ref=per_head,
        predicates=(OrderedThresholdPredicate(column="value", op="gt", value=2.0),),
    )

    for kept in (ranked, changed, averaged, filtered):
        assert await _units(store, kept) == [USD_PER_PERSON]
    assert await _units(store, grown) == [None]
    # Kenya's 2, 2.4 and 3 dollars a head pass the filter with Guatemala's 3.
    assert filtered.structured_content["rows"] == 3


async def test_a_share_of_a_keyed_quotient_refuses_the_side_it_would_add_up(
    store: InMemoryMaterialisationStore,
) -> None:
    """Kenya's aid per head over both recipients' adds two per-head figures
    on the denominator; over its own figure each side reads one row."""
    aid, population = await _aid_and_population(store)
    per_head = _pair(
        await quotient_of(
            frame_ref=aid,
            denominator_frame_ref=population,
            numerator_by=["recipient"],
            denominator_by={"recipient": "observationAbout"},
        )
    )
    kenya = {"recipient": KEN}
    held = len(store._entries)

    pooled = await share_of(frame_ref=per_head, numerator_entity=kenya)
    of_itself = await share_of(frame_ref=per_head, numerator_entity=kenya, denominator_entity=kenya)

    assert pooled.structured_content["code"] == "composed_unit_sum_unsupported"
    assert pooled.structured_content["fix"] == {
        "verb": "share_of",
        "side": "denominator",
        "unit": USD_PER_PERSON,
    }
    assert of_itself.is_error is False
    assert len(store._entries) == held + 1


# The declared shape and determinism ----------------------------------------------


async def test_a_keyed_call_declares_the_keys_it_left_out(
    store: InMemoryMaterialisationStore,
) -> None:
    """Uganda has no population group, Chad's periods miss its aid's and
    Tanzania's population is zero in its one matched period, so three of
    five keys are left out and the result says so beside its shape."""
    aid = await _seed(
        store,
        AID,
        {
            KEN: {"2020": 100.0},
            GTM: {"2020": 40.0},
            "dcid:country/UGA": {"2020": 70.0},
            "dcid:country/TCD": {"2019": 30.0},
            "dcid:country/TZA": {"2020": 9.0},
        },
        column="recipient",
        unit=USD,
    )
    population = await _seed(
        store,
        POPULATION,
        {
            KEN: {"2020": 50.0},
            GTM: {"2020": 20.0},
            "dcid:country/TCD": {"2020": 16.0},
            "dcid:country/TZA": {"2020": 0.0},
        },
    )

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert result.structured_content["dropped_key_count"] == 3
    assert result.structured_content["role_keys"] == ["recipient"]
    assert result.structured_content["rows"] == 2
    frame, trace = await _stored(store, result)
    assert trace.applied_policy.detail["unmatched_key_count"] == 1
    assert trace.applied_policy.detail["keys_without_shared_period_count"] == 1
    assert trace.applied_policy.detail["keys_with_zero_denominator_count"] == 1
    # Every numerator key either has a row or is counted as dropped.
    recipient = frame.columns.index("recipient")
    assert len({row[recipient] for row in frame.rows}) + 3 == 5
    digest = result.content[0].text
    assert (
        "numerator keys left out: 1 with no denominator match (recipient=dcid:country/UGA); "
        "1 sharing no period with the denominator (recipient=dcid:country/TCD); "
        "1 with a zero denominator in every matched period (recipient=dcid:country/TZA)"
    ) in digest
    # The left-out line names every unmatched key, so the exclusions block
    # leaves that record out. It keeps the zero-denominator records.
    assert "unmatched_key" not in digest
    assert "zero_denominator" in digest


async def test_keyed_coverage_counts_the_cells_each_key_shares_with_its_group(
    store: InMemoryMaterialisationStore,
) -> None:
    """Each country's aid spans two years and its population one, a
    different year for each, so every period is matched by some key while
    half the numerator's cells have no denominator."""
    aid = await _seed(
        store,
        AID,
        {KEN: {"2020": 100.0, "2021": 120.0}, GTM: {"2020": 40.0, "2021": 30.0}},
        column="recipient",
        unit=USD,
    )
    population = await _seed(store, POPULATION, {KEN: {"2020": 50.0}, GTM: {"2021": 20.0}})

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert result.structured_content["rows"] == 2
    assert result.structured_content["dropped_key_count"] == 0
    _frame, trace = await _stored(store, result)
    assert trace.applied_policy.realised_coverage_fraction == 0.5
    assert trace.applied_policy.detail["numerator_coverage_fraction"] == 0.5
    assert trace.applied_policy.detail["denominator_coverage_fraction"] == 1.0
    assert "numerator 50.0%, denominator 100.0%" in result.content[0].text


async def test_a_keyed_call_names_the_pairs_one_side_alone_reports(
    store: InMemoryMaterialisationStore,
) -> None:
    """Kenya's aid starts in 2021 and its population in 2019, and Guatemala
    matches in every year, so every period is matched by some key and the
    two pairs Kenya lacks are named under it."""
    aid = await _seed(
        store,
        AID,
        {KEN: {"2021": 120.0}, GTM: {"2019": 20.0, "2020": 40.0, "2021": 30.0}},
        column="recipient",
        unit=USD,
    )
    population = await _seed(
        store,
        POPULATION,
        {
            KEN: {"2019": 50.0, "2020": 50.0, "2021": 50.0},
            GTM: {"2019": 20.0, "2020": 20.0, "2021": 20.0},
        },
    )

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert result.structured_content["rows"] == 4
    assert result.structured_content["dropped_key_count"] == 0
    lines = result.content[0].text.splitlines()
    assert f"pairs left out: only the denominator reports recipient={KEN} 2019 to 2020 (2)" in lines
    assert not any(line.startswith("periods left out") for line in lines)


async def test_keys_that_each_lack_a_period_the_other_holds_are_named_as_pairs(
    store: InMemoryMaterialisationStore,
) -> None:
    """Kenya's aid and Guatemala's population both hold 2021, so both sides
    report the period and each key is named on the side it is missing from."""
    aid = await _seed(
        store,
        AID,
        {KEN: {"2020": 100.0, "2021": 120.0}, GTM: {"2020": 40.0}},
        column="recipient",
        unit=USD,
    )
    population = await _seed(
        store, POPULATION, {KEN: {"2020": 50.0}, GTM: {"2020": 20.0, "2021": 20.0}}
    )

    result = await quotient_of(
        frame_ref=aid,
        denominator_frame_ref=population,
        numerator_by=["recipient"],
        denominator_by={"recipient": "observationAbout"},
    )

    assert result.structured_content["rows"] == 2
    lines = result.content[0].text.splitlines()
    assert (
        f"pairs left out: only the numerator reports recipient={KEN} 2021 (1); "
        f"only the denominator reports recipient={GTM} 2021 (1)"
    ) in lines
    assert not any(line.startswith("periods left out") for line in lines)


async def test_a_zero_denominator_cell_is_left_out_and_named(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, _population = await _aid_and_population(store)
    population = await _seed(store, POPULATION, {KEN: {"2020": 50.0, "2021": 0.0, "2022": 50.0}})

    result = await quotient_of(**_per_head(aid, population, KEN))

    assert result.structured_content["rows"] == 2
    assert result.structured_content["dropped_key_count"] == 0
    _frame, trace = await _stored(store, result)
    assert trace.applied_policy.detail["zero_denominator_cell_count"] == 1
    # The whole digest between the label and the frame reference: the
    # zero-denominator line states the cell, and the exclusions block does
    # not repeat it in records that name no key.
    sections = result.content[0].text.split("\n\n")
    assert [sections[0], *sections[2:-1]] == [
        "quotient_of: 2 matched period(s), in dcid:USDollar per Count_Person",
        "zero denominators: 1 matched cell(s) left out, the denominator summing to zero "
        "there, in 2021",
        "2 quotient(s), numerator / denominator, in dcid:USDollar per Count_Person:\n"
        "  2020: 100 / 50 = 2\n"
        "  2022: 150 / 50 = 3",
        "divided over 2020 to 2022, the periods both sides report (numerator offered "
        "2020 to 2022, denominator 2020 to 2022); coverage numerator 100.0%, "
        "denominator 100.0%",
    ]


async def test_the_same_call_returns_the_same_pair_whatever_its_argument_order(
    store: InMemoryMaterialisationStore,
) -> None:
    aid, population = await _aid_and_population(store)

    first = await quotient_of(**_per_head(aid, population, KEN))
    again = await quotient_of(**_per_head(aid, population, KEN))
    reordered = await quotient_of(
        frame_ref=dict(reversed(aid.items())),
        denominator_frame_ref=population,
        denominator_entity={"observationAbout": KEN},
        numerator_entity={"recipient": KEN},
    )

    assert _pair(again) == _pair(first)
    assert _pair(reordered) == _pair(first)
    assert again.structured_content == first.structured_content
    assert again.content[0].text == first.content[0].text
