"""Step records: the step each of the nine verbs stores on its trace, and
the list `common.steps.assemble_steps` joins along a frame's ancestry.

The worked quotient chain is checked against dfo-spec-kit's step schema and
its matching vector, and the growth and the chain through every verb against
the schema. dfo-ai does not pin the kit's descriptions distribution, so where
it is not installed those tests fail on import.
"""

from __future__ import annotations

import json
import secrets
from datetime import date, timedelta
from importlib.util import find_spec
from typing import Any

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.growth_tool import growth
from analysis_mcp.join_tool import join
from analysis_mcp.quotient_of_tool import quotient_of
from analysis_mcp.rank_tool import rank
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.steps import NEW_VARIABLE_OPS, build_step
from analysis_mcp.sum_of_tool import sum_of
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    set_store,
)
from common.steps import assemble_steps
from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult
from pydantic import ValidationError

ODA = "dcid:Amount_OfficialDevelopmentAssistance_Gross"
GNI = "dcid:Amount_EconomicActivity_GrossNationalIncome"
USD = "dcid:USDollar"
YEARS = [str(year) for year in range(2010, 2025)]

_KIT = pytest.mark.xfail(
    find_spec("dfo_descriptions") is None,
    raises=ModuleNotFoundError,
    strict=True,
    reason="dfo-spec-kit-descriptions is not pinned",
)


@pytest.fixture
async def store():
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


async def _root(
    store: InMemoryMaterialisationStore,
    stat_var: str,
    rows: list[dict[str, Any]],
    *,
    roles: tuple[str, ...] = ("recipient",),
) -> dict[str, str]:
    """A root frame of `rows` under `stat_var` in US dollars, with its mint record."""
    frame = Frame.build(
        role_keys=list(roles),
        rows=[{"stat_var": stat_var, "unit": USD, **row} for row in rows],
    )
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    trace = MaterialisationTrace(
        upstream_trace={}, spec_id=ref.spec_id, content_version=ref.content_version
    )
    await store.put(ref, frame=frame, trace=trace, writer="test")
    await store.put_mint_record(MintRecord(frame=ref, dataset=ref, options={}))
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _series(entity: str, periods: list[str], value: float, **roles: str) -> list[dict[str, Any]]:
    return [
        {"recipient": entity, "time_period": period, "value": value, **roles} for period in periods
    ]


def _ref(result: ToolResult) -> FrameRef:
    assert result.is_error is False, result.content[0].text
    return FrameRef(
        spec_id=result.structured_content["spec_id"],
        content_version=result.structured_content["content_version"],
    )


async def _own_step(store: InMemoryMaterialisationStore, result: ToolResult) -> dict[str, Any]:
    trace = await store.get_trace(_ref(result))
    (step,) = trace.steps
    return step.model_dump(mode="json")


async def _written_stat_vars(store: InMemoryMaterialisationStore, result: ToolResult) -> set[str]:
    """The bare dcids of the variables `result`'s rows are written under."""
    held = await store.get(_ref(result))
    assert held is not None
    return {ref.removeprefix("dcid:") for ref in held[0].table.column("stat_var").to_pylist()}


def _version(ref: dict[str, str]) -> dict[str, str]:
    return {"id": ref["spec_id"], "content_version": ref["content_version"]}


async def _worked_quotient(store: InMemoryMaterialisationStore) -> tuple[ToolResult, list[dict]]:
    """The design's worked quotient over 54 recipients: 32 sharing all 15
    years with their income figure, 6 sharing only 2013 to 2024, 9 with no
    income figure, 5 sharing no year with it, and 2 with zero income. Returns
    the result and the two roots."""
    aid: list[dict[str, Any]] = []
    income: list[dict[str, Any]] = []
    for index in range(54):
        recipient = f"dcid:country/R{index:02d}"
        aid += _series(recipient, YEARS, 10.0 + index)
        if index < 32:
            income += _series(recipient, YEARS, 1000.0)
        elif index < 38:
            income += _series(recipient, YEARS[3:], 1000.0)
        elif index < 47:
            continue
        elif index < 52:
            income += _series(recipient, ["2000", "2001"], 1000.0)
        else:
            income += _series(recipient, YEARS, 0.0)
    roots = [await _root(store, ODA, aid), await _root(store, GNI, income)]
    result = await quotient_of(
        frame_ref=roots[0],
        denominator_frame_ref=roots[1],
        numerator_by=["recipient"],
        denominator_by={"recipient": "recipient"},
    )
    return result, roots


def _validate_against_kit(steps: list[dict[str, Any]]) -> None:
    # The kit is no declared dependency, so a missing one fails the caller alone.
    from dfo_descriptions import check_steps, steps_schema  # noqa: PLC0415
    from jsonschema import Draft202012Validator  # noqa: PLC0415

    Draft202012Validator(steps_schema()).validate(steps)
    check_steps(steps)


def _without_labels(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_labels(item) for key, item in value.items() if key != "label"}
    if isinstance(value, list):
        return [_without_labels(item) for item in value]
    return value


@_KIT
async def test_the_worked_quotient_chain_matches_the_kit_vector(store) -> None:
    # The kit is no declared dependency, so a missing one fails this test alone.
    from dfo_descriptions import check_steps, schema_dir, steps_schema  # noqa: PLC0415
    from jsonschema import Draft202012Validator  # noqa: PLC0415

    result, roots = await _worked_quotient(store)
    steps = await assemble_steps(_ref(result), store=store)

    assert steps is not None
    Draft202012Validator(steps_schema()).validate(steps)
    check_steps(steps)
    vector = json.loads(
        (schema_dir().parent / "vectors" / "steps-quotient-chain-known-reasons.json").read_text()
    )
    expected = _without_labels(vector["input"])
    for step, root in zip(expected[:2], roots, strict=True):
        step["input"] = _version(root)
    expected[2]["params"] = {
        "denominator_by": {"recipient": "recipient"},
        "numerator_by": ["recipient"],
    }
    (expected[2]["stat_var"],) = await _written_stat_vars(store, result)
    assert steps == expected


async def test_quotient_counts_the_keys_it_keeps_and_leaves_out_by_reason(store) -> None:
    result, roots = await _worked_quotient(store)

    step = await _own_step(store, result)

    assert step["id"] == _ref(result).spec_id
    assert {step["stat_var"]} == await _written_stat_vars(store, result)
    assert step["inputs"] == {
        "numerator": _version(roots[0]),
        "denominator": _version(roots[1]),
    }
    assert step["kept"] == {"entities": 38, "periods": 15, "entities_with_fewer_periods": 6}
    assert step["left_out"] == [
        {"reason": "unmatched_key", "count": 9, "of": "entities"},
        {"reason": "period_unmatched", "count": 5, "of": "entities"},
        {"reason": "zero_denominator", "count": 2, "of": "entities"},
    ]
    assert step["period_coverage"][:4] == [
        {"period": "2010", "entities": 32},
        {"period": "2011", "entities": 32},
        {"period": "2012", "entities": 32},
        {"period": "2013", "entities": 38},
    ]


async def test_sum_counts_the_periods_it_keeps_and_the_one_sided_periods(store) -> None:
    left = await _root(store, ODA, _series("dcid:country/KEN", YEARS[:5], 1.0))
    right = await _root(store, GNI, _series("dcid:country/KEN", YEARS[2:8], 2.0))

    result = await sum_of(frame_ref=left, right_frame_ref=right)

    (written,) = await _written_stat_vars(store, result)
    assert await _own_step(store, result) == {
        "id": _ref(result).spec_id,
        "op": "sum",
        "stat_var": written,
        "inputs": {"left": _version(left), "right": _version(right)},
        "policy": "sum_of_intersect_periods",
        "params": {},
        "kept": {"periods": 3},
        "left_out": [{"reason": "period_unmatched", "count": 5, "of": "periods"}],
        "unit": {"ref": "USDollar"},
    }


async def test_share_counts_the_entities_each_period_s_total_covers(store) -> None:
    """Each recipient's share of all donors' aid: the total covers two donors
    in 2020 and three in 2021, and a 2022 total no recipient shares is left
    out as values."""
    part = await _root(
        store,
        ODA,
        _series("dcid:country/KEN", ["2020", "2021"], 1.0)
        + _series("dcid:country/GTM", ["2021"], 1.0),
    )
    donors = [
        {"donor": donor, "time_period": period, "value": 10.0}
        for donor, periods in (("d/A", ["2020", "2021", "2022"]), ("d/B", ["2020", "2021"]))
        for period in periods
    ] + [{"donor": "d/C", "time_period": "2021", "value": 10.0}]
    whole = await _root(store, GNI, donors, roles=("donor",))

    result = await share_of(frame_ref=part, denominator_frame_ref=whole, numerator_by=["recipient"])

    step = await _own_step(store, result)
    assert step["op"] == "share"
    assert {step["stat_var"]} == await _written_stat_vars(store, result)
    assert step["policy"] == "share_of_intersect_periods"
    assert step["params"] == {"numerator_by": ["recipient"]}
    assert step["kept"] == {"entities": 2, "periods": 2, "entities_with_fewer_periods": 1}
    assert step["period_coverage"] == [
        {"period": "2020", "entities": 2},
        {"period": "2021", "entities": 3},
    ]
    assert step["left_out"] == [{"reason": "period_unmatched", "count": 1, "of": "entity_periods"}]
    assert step["unit"] == {"ref": "Percent"}


@pytest.mark.parametrize(
    ("group_by", "kept", "pooled"),
    [
        # The donors each recipient's total pools count once each per year.
        (["recipient"], {"groups": 2, "entities": 2, "periods": 2}, 3),
        # One total holds no entity and pools every flow.
        ([], {"groups": 1, "periods": 2}, 4),
        # Every role grouped: each total pools its own flow.
        (["donor", "recipient"], {"groups": 4, "entities": 4, "periods": 2}, 4),
    ],
)
async def test_aggregate_counts_the_entities_holding_a_total_and_those_each_year_pools(
    store, group_by: list[str], kept: dict[str, int], pooled: int
) -> None:
    """Two recipients funded by three donors over two years."""
    rows = [
        {"recipient": recipient, "donor": f"d/{donor}", "time_period": period, "value": 1.0}
        for recipient, donors in (("dcid:country/KEN", "AB"), ("dcid:country/GTM", "BC"))
        for donor in donors
        for period in ("2020", "2021")
    ]
    frame_ref = await _root(store, ODA, rows, roles=("recipient", "donor"))

    result = await aggregate(frame_ref=frame_ref, group_by=group_by, aggregation="sum")

    assert await _own_step(store, result) == {
        "id": _ref(result).spec_id,
        "op": "aggregate",
        "input": _version(frame_ref),
        "policy": "aggregate_skip_absent_grain_cells",
        "params": {"aggregation": "sum", **({"group_by": group_by} if group_by else {})},
        "kept": kept,
        "left_out": [],
        "period_coverage": [
            {"period": "2020", "entities": pooled},
            {"period": "2021", "entities": pooled},
        ],
        "unit": {"ref": "USDollar"},
    }


async def test_a_rank_naming_an_entity_past_the_kit_bound_leaves_its_coverage_out(
    store,
) -> None:
    """Two donor and recipient pairs whose joined refs run past 512
    characters: the step keeps its count without the coverage list."""
    rows = [
        row
        for donor in "AB"
        for row in _series(f"dcid:country/{'K' * 300}", YEARS, 1.0, donor=f"d/{donor * 300}")
    ]
    root = await _root(store, ODA, rows, roles=("recipient", "donor"))

    result = await rank(frame_ref=root, rank_by=("recipient", "donor"), aggregation="sum")

    step = await _own_step(store, result)
    assert step["kept"] == {"entities": 2}
    assert "entity_coverage" not in step


@pytest.mark.parametrize(
    ("coverage", "member"),
    [
        ({"period_entities": [("2020", 1), ("P" * 33, 1)]}, "period_coverage"),
        ({"entity_coverage": [("country/KEN", 1, 1), ("K" * 513, 1, 1)]}, "entity_coverage"),
    ],
)
async def test_a_coverage_entry_named_past_the_kit_bound_leaves_its_list_out(
    store, coverage: dict[str, Any], member: str
) -> None:
    root = await _root(store, ODA, _series("dcid:country/KEN", YEARS, 1.0))
    trace = await store.get_trace(_ref(await aggregate(frame_ref=root, aggregation="sum")))
    ref = FrameRef.model_validate(root)

    step = build_step(
        op="aggregate",
        trace=trace,
        inputs={"input": ref},
        kept={"entities": 2},
        left_out=[],
        unit=None,
        **coverage,
    )
    assert member not in step.model_dump()
    assert step.kept == {"entities": 2}


_HELD = {"donor": "dcid:country/FRA"}


@pytest.mark.parametrize(
    ("right", "written"),
    [
        ({f"role{index}": "dcid:X" for index in range(16)}, "held"),
        ({f"role{index}": "dcid:X" for index in range(17)}, "empty"),
        ({"r" * 64: "dcid:X"}, "held"),
        ({"r" * 65: "dcid:X"}, "empty"),
        ({"donor": "d" * 512}, "held"),
        ({"donor": "d" * 513}, "empty"),
    ],
    ids=["16-columns", "17-columns", "64-name", "65-name", "512-value", "513-value"],
)
async def test_sides_past_the_kit_bound_are_written_empty(
    store, right: dict[str, str], written: str
) -> None:
    """A side past the kit's bounds empties every side, so a reader takes
    the entities as unknown instead of naming the sides by a column that
    is left."""
    left = await _root(store, ODA, _series("dcid:country/KEN", YEARS, 1.0))
    trace = await store.get_trace(_ref(await compare(frame_ref=left, right_frame_ref=left)))
    ref = FrameRef.model_validate(left)

    step = build_step(
        op="compare",
        trace=trace,
        inputs={"left": ref, "right": ref},
        kept={"periods": 1},
        left_out=[],
        unit=None,
        sides={"left": _HELD, "right": right},
    )

    expected = {"left": _HELD, "right": right} if written == "held" else {"left": {}, "right": {}}
    assert step.model_dump()["sides"] == expected


@pytest.mark.parametrize(
    "coverage",
    [
        {"period_entities": [("2020", -1)]},
        {"period_entities": [("", 1)]},
        {"entity_coverage": [("country/KEN", -1, 1)]},
        {"entity_coverage": [("", 1, 1)]},
    ],
)
async def test_a_coverage_entry_invalid_within_the_kit_bound_raises(
    store, coverage: dict[str, Any]
) -> None:
    """Only a name too long for the kit leaves a list out; a negative count
    or an empty name is a fault the step does not hide."""
    root = await _root(store, ODA, _series("dcid:country/KEN", YEARS, 1.0))
    trace = await store.get_trace(_ref(await aggregate(frame_ref=root, aggregation="sum")))

    with pytest.raises(ValidationError):
        build_step(
            op="aggregate",
            trace=trace,
            inputs={"input": FrameRef.model_validate(root)},
            kept={},
            left_out=[],
            unit=None,
            **coverage,
        )


@pytest.mark.parametrize("periods", [500, 501])
@pytest.mark.parametrize("verb", ["aggregate", "share_of"])
async def test_a_coverage_list_over_the_kit_bound_is_left_out_and_the_step_kept(
    store, verb: str, periods: int
) -> None:
    """One recipient's daily values: up to 500 periods the step names each
    period's coverage, and over it the step keeps its counts without it."""
    days = [(date(2000, 1, 1) + timedelta(days=day)).isoformat() for day in range(periods)]
    part = await _root(store, ODA, _series("dcid:country/KEN", days, 1.0))
    if verb == "aggregate":
        result = await aggregate(frame_ref=part, group_by=["recipient"], aggregation="sum")
    else:
        whole = await _root(
            store,
            GNI,
            [{"donor": "d/A", "time_period": day, "value": 10.0} for day in days],
            roles=("donor",),
        )
        result = await share_of(
            frame_ref=part, denominator_frame_ref=whole, numerator_by=["recipient"]
        )

    step = await _own_step(store, result)

    assert step["kept"]["periods"] == periods
    assert len(step.get("period_coverage", ())) == (periods if periods <= 500 else 0)


async def test_a_chain_numbers_each_frame_once_after_the_steps_it_reads(store) -> None:
    """A quotient of one aggregate by another over the same root: the root is
    one retrieve step, each aggregate names it, and the quotient names both."""
    rows = [
        {"recipient": recipient, "donor": "d/A", "time_period": period, "value": 2.0}
        for recipient in ("dcid:country/KEN", "dcid:country/GTM")
        for period in ("2020", "2021")
    ]
    root = await _root(store, ODA, rows, roles=("recipient", "donor"))
    by_recipient = _ref(
        await aggregate(frame_ref=root, group_by=["recipient", "time_period"], aggregation="sum")
    )
    by_period = _ref(await aggregate(frame_ref=root, group_by=["time_period"], aggregation="sum"))

    result = await quotient_of(
        frame_ref=by_recipient.model_dump(),
        denominator_frame_ref=by_period.model_dump(),
        numerator_by=["recipient"],
    )
    steps = await assemble_steps(_ref(result), store=store)

    assert steps is not None
    assert [(step["id"], step["op"]) for step in steps] == [
        ("s1", "retrieve"),
        ("s2", "aggregate"),
        ("s3", "aggregate"),
        ("s4", "quotient"),
    ]
    assert steps[0]["input"] == _version(root)
    assert steps[0]["options"] == {}
    assert steps[1]["input"] == steps[2]["input"] == "s1"
    assert steps[3]["inputs"] == {"numerator": "s2", "denominator": "s3"}


async def test_a_chain_through_a_trace_without_its_step_has_no_steps(store) -> None:
    root = await _root(store, ODA, _series("dcid:country/KEN", YEARS[:3], 1.0))
    result = await aggregate(frame_ref=root, group_by=[], aggregation="sum")
    trace = await store.get_trace(_ref(result))

    steps = await assemble_steps(
        _ref(result), store=store, trace=trace.model_copy(update={"steps": ()})
    )

    assert steps is None


async def _growth_over_45(store: InMemoryMaterialisationStore) -> tuple[ToolResult, dict]:
    """The design's growth over 45 recipients from 2015 to 2024: 20 report
    both years and 25 stop at 2023, so they have no end value."""
    rows: list[dict[str, Any]] = []
    for index in range(45):
        last = 2024 if index < 20 else 2023
        rows += _series(f"dcid:country/R{index:02d}", [str(y) for y in range(2015, last + 1)], 5.0)
    root = await _root(store, ODA, rows)
    result = await growth(
        frame_ref=root, window_start="2015", window_end="2024", entity_by=["recipient"]
    )
    return result, root


async def test_growth_keeps_20_of_45_entities_and_leaves_out_25_with_no_end_value(store) -> None:
    result, root = await _growth_over_45(store)

    assert await _own_step(store, result) == {
        "id": _ref(result).spec_id,
        "op": "growth",
        "input": _version(root),
        "policy": "growth_endpoint_years",
        "params": {
            "change": "relative",
            "end": "2024",
            "entity_by": ["recipient"],
            "start": "2015",
        },
        "kept": {"entities": 20},
        "left_out": [{"reason": "endpoint_missing", "count": 25, "of": "entities"}],
        "unit": {"ref": "dimensionless"},
    }


@_KIT
async def test_the_growth_over_45_entities_meets_the_kit_schema(store) -> None:
    result, _ = await _growth_over_45(store)

    steps = await assemble_steps(_ref(result), store=store)

    assert steps is not None
    _validate_against_kit(steps)


async def test_growth_counts_each_entity_it_cannot_compute_by_reason(store) -> None:
    """Absolute growth keeps the input's unit; relative growth leaves out a
    zero base and an entity with no row inside the window."""
    rows = (
        _series("dcid:country/KEN", ["2020", "2021"], 4.0)
        + _series("dcid:country/GTM", ["2020", "2021"], 0.0)
        + _series("dcid:country/PER", ["2010"], 1.0)
    )
    root = await _root(store, ODA, rows)

    relative = await _own_step(
        store,
        await growth(
            frame_ref=root, window_start="2020", window_end="2021", entity_by=["recipient"]
        ),
    )
    absolute = await _own_step(
        store,
        await growth(
            frame_ref=root,
            window_start="2020",
            window_end="2021",
            entity_by=["recipient"],
            change="absolute",
        ),
    )

    assert relative["kept"] == {"entities": 1}
    assert relative["left_out"] == [
        {"reason": "endpoint_missing", "count": 1, "of": "entities"},
        {"reason": "zero_or_sign_crossing_base", "count": 1, "of": "entities"},
    ]
    assert absolute["kept"] == {"entities": 2}
    assert absolute["unit"] == {"ref": "USDollar"}


async def test_rank_records_how_many_window_periods_each_kept_entity_holds(store) -> None:
    rows = (
        _series("dcid:country/NGA", YEARS, 9.0)
        + _series("dcid:country/ETH", YEARS[1:], 8.0)
        + _series("dcid:country/KEN", YEARS, 1.0)
    )
    root = await _root(store, ODA, rows)

    result = await rank(
        frame_ref=root,
        rank_by=("recipient",),
        aggregation="sum",
        window_start="2010",
        window_end="2024",
        limit=2,
    )

    assert await _own_step(store, result) == {
        "id": _ref(result).spec_id,
        "op": "rank",
        "input": _version(root),
        "policy": "rank_what_is_present",
        "params": {
            "aggregation": "sum",
            "end": "2024",
            "limit": 2,
            "order": "descending",
            "per_period": False,
            "rank_by": ["recipient"],
            "start": "2010",
        },
        "kept": {"entities": 2},
        "left_out": [],
        "entity_coverage": [
            {"entity": {"ref": "country/NGA"}, "periods": 15, "of": 15},
            {"entity": {"ref": "country/ETH"}, "periods": 14, "of": 15},
        ],
        "unit": {"ref": "USDollar"},
    }


async def test_filter_counts_the_values_it_keeps_and_leaves_nothing_out(store) -> None:
    root = await _root(
        store,
        ODA,
        _series("dcid:country/KEN", YEARS[:3], 1.0) + _series("dcid:country/GTM", YEARS[:3], 5.0),
    )

    result = await filter_verb(
        frame_ref=root,
        predicates=[{"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 2.0}],
    )

    step = await _own_step(store, result)
    assert step["op"] == "filter"
    assert step["policy"] == "filter_row_value_predicates"
    assert step["params"] == {"predicates": [{"column": "value", "op": "gt", "operand": 2.0}]}
    assert step["kept"] == {"entity_periods": 3}
    assert step["left_out"] == []
    assert step["unit"] == {"ref": "USDollar"}


@_KIT
async def test_growth_rank_and_filter_name_their_params_as_the_kit_vectors_do(store) -> None:
    """Every param name a kit vector gives growth, rank or filter, and every
    member it gives a filter predicate, is a name the step writes."""
    from dfo_descriptions import schema_dir  # noqa: PLC0415

    root = await _root(store, ODA, _series("dcid:country/KEN", YEARS, 2.0))
    written = {
        "growth": await growth(
            frame_ref=root, window_start="2015", window_end="2024", entity_by=["recipient"]
        ),
        "rank": await rank(frame_ref=root, rank_by=("recipient",), aggregation="sum", limit=1),
        "filter": await filter_verb(
            frame_ref=root,
            predicates=[{"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 0}],
        ),
    }
    params = {op: (await _own_step(store, result))["params"] for op, result in written.items()}
    vector_steps = [
        step
        for path in sorted((schema_dir().parent / "vectors").glob("steps-*.json"))
        if isinstance(steps := json.loads(path.read_text())["input"], list)
        for step in steps
        if step.get("op") in written
    ]

    assert {step["op"] for step in vector_steps} == set(written)
    for step in vector_steps:
        named = step.get("params", {})
        assert named.keys() <= params[step["op"]].keys(), step["op"]
        for predicate in named.get("predicates", []):
            assert predicate.keys() == params["filter"]["predicates"][0].keys()


async def test_compare_counts_the_periods_it_keeps_and_the_one_sided_periods(store) -> None:
    left = await _root(store, ODA, _series("dcid:country/KEN", YEARS[:5], 1.0))
    right = await _root(store, GNI, _series("dcid:country/KEN", YEARS[2:8], 2.0))

    result = await compare(frame_ref=left, right_frame_ref=right)

    assert await _own_step(store, result) == {
        "id": _ref(result).spec_id,
        "op": "compare",
        "inputs": {"left": _version(left), "right": _version(right)},
        "policy": "compare_intersect_periods",
        "params": {},
        "kept": {"periods": 3},
        "left_out": [{"reason": "period_unmatched", "count": 5, "of": "periods"}],
        "unit": {"ref": "USDollar"},
        "stat_var": _ref(result).spec_id,
        "sides": {
            "left": {"recipient": "dcid:country/KEN"},
            "right": {"recipient": "dcid:country/KEN"},
        },
    }


async def test_a_compare_of_two_retrievals_names_each_side_by_the_entities_it_holds(
    store,
) -> None:
    """France's and the United Kingdom's aid, retrieved apart and compared
    with no binding: each side holds its donor, and the digest names the
    difference France minus the United Kingdom."""
    uk, france = [
        await _root(
            store,
            ODA,
            _series("dcid:country/KEN", YEARS[:3], value, donor=donor),
            roles=("donor", "recipient"),
        )
        for donor, value in (("dcid:country/GBR", 1.0), ("dcid:country/FRA", 2.0))
    ]

    result = await compare(frame_ref=uk, right_frame_ref=france)

    assert (await _own_step(store, result))["sides"] == {
        "left": {"donor": "dcid:country/GBR", "recipient": "dcid:country/KEN"},
        "right": {"donor": "dcid:country/FRA", "recipient": "dcid:country/KEN"},
    }
    assert (
        f"stat_var dcid:{_ref(result).spec_id}: {ODA}, dcid:country/FRA minus dcid:country/GBR\n"
        in result.content[0].text
    )


async def test_join_counts_the_matched_and_unmatched_values_of_both_sides(store) -> None:
    left = await _root(
        store,
        ODA,
        _series("dcid:country/KEN", YEARS[:4], 1.0) + _series("dcid:country/GTM", YEARS[:2], 1.0),
    )
    right = await _root(store, GNI, _series("dcid:country/KEN", YEARS[1:4], 2.0))

    result = await join(frame_ref=left, right_frame_ref=right)

    assert await _own_step(store, result) == {
        "id": _ref(result).spec_id,
        "op": "join",
        "inputs": {"left": _version(left), "right": _version(right)},
        "policy": "join_inner",
        "params": {},
        "kept": {"entity_periods": 3},
        "left_out": [{"reason": "unmatched_key", "count": 3, "of": "entity_periods"}],
        "unit": {"ref": "USDollar"},
    }


@pytest.mark.parametrize(
    ("verb", "arguments"),
    [
        # Two relative growths are each a ratio.
        (compare, {}),
        (join, {}),
        (sum_of, {}),
        # One relative growth is a ratio: its total, mean and ranking are too.
        (aggregate, {"group_by": ["recipient"], "aggregation": "mean"}),
        (rank, {"rank_by": ("recipient",), "aggregation": "mean"}),
    ],
)
async def test_a_verb_over_ratios_names_the_ratio_unit(
    store, verb, arguments: dict[str, Any]
) -> None:
    sides = [
        _ref(
            await growth(
                frame_ref=await _root(store, stat_var, _series("dcid:country/KEN", YEARS, value)),
                window_start=YEARS[0],
                window_end=YEARS[-1],
                entity_by=["recipient"],
            )
        ).model_dump()
        for stat_var, value in ((ODA, 1.0), (GNI, 2.0))
    ]

    if verb in (aggregate, rank):
        result = await verb(frame_ref=sides[0], **arguments)
    else:
        result = await verb(frame_ref=sides[0], right_frame_ref=sides[1], **arguments)

    assert (await _own_step(store, result))["unit"] == {"ref": "dimensionless"}


async def test_a_chain_through_every_verb_is_complete(store) -> None:
    """One frame whose ancestry runs through all nine verbs: a quotient of
    aggregated aid by a compare of a sum, grown and filtered, joined with a
    rank of each recipient's share."""
    aid = await _root(
        store,
        ODA,
        [
            {"recipient": recipient, "donor": donor, "time_period": period, "value": base * step}
            for recipient, base in (("dcid:country/KEN", 3.0), ("dcid:country/GTM", 1.0))
            for donor in ("d/A", "d/B")
            for step, period in enumerate(YEARS[:5], start=1)
        ],
        roles=("recipient", "donor"),
    )
    income = await _root(
        store,
        GNI,
        _series("dcid:country/KEN", YEARS[:5], 100.0)
        + _series("dcid:country/GTM", YEARS[:5], 50.0),
    )
    single = await _root(store, ODA, _series("dcid:country/KEN", YEARS[:5], 7.0))
    kenya = {"recipient": "dcid:country/KEN"}

    totals = _ref(
        await aggregate(frame_ref=aid, group_by=["recipient", "time_period"], aggregation="sum")
    ).model_dump()
    summed = _ref(
        await sum_of(
            frame_ref=income, right_frame_ref=single, left_entity=kenya, right_entity=kenya
        )
    ).model_dump()
    compared = _ref(
        await compare(frame_ref=summed, right_frame_ref=single, right_entity=kenya)
    ).model_dump()
    per_gap = _ref(
        await quotient_of(
            frame_ref=totals, denominator_frame_ref=compared, numerator_by=["recipient"]
        )
    ).model_dump()
    grown = _ref(
        await growth(
            frame_ref=per_gap, window_start="2010", window_end="2014", entity_by=["recipient"]
        )
    ).model_dump()
    kept = _ref(
        await filter_verb(
            frame_ref=grown,
            predicates=[
                {"kind": "category_match", "column": "recipient", "values": ["dcid:country/KEN"]}
            ],
        )
    ).model_dump()
    shares = _ref(
        await share_of(frame_ref=totals, denominator_frame_ref=income, numerator_by=["recipient"])
    ).model_dump()
    ranked = _ref(
        await rank(frame_ref=shares, rank_by=("recipient",), aggregation="mean")
    ).model_dump()

    result = await join(frame_ref=kept, right_frame_ref=ranked)
    steps = await assemble_steps(_ref(result), store=store)

    assert steps is not None
    assert sorted({step["op"] for step in steps}) == [
        "aggregate",
        "compare",
        "filter",
        "growth",
        "join",
        "quotient",
        "rank",
        "retrieve",
        "share",
        "sum",
    ]
    by_op = {step["op"]: step for step in steps}
    # Only the operations that write a new variable name one, so renumbering
    # keeps each step's `stat_var` as its own trace stored it.
    assert {step["op"] for step in steps if "stat_var" in step} == NEW_VARIABLE_OPS
    assert by_op["compare"]["stat_var"] == compared["spec_id"]
    assert by_op["quotient"]["stat_var"] == per_gap["spec_id"]
    assert by_op["share"]["stat_var"] == shares["spec_id"]
    assert by_op["sum"]["stat_var"] == summed["spec_id"]
    # A filter over a relative growth keeps its dimensionless ratio.
    assert by_op["filter"]["unit"] == by_op["growth"]["unit"] == {"ref": "dimensionless"}
    for step in steps:
        if step["op"] != "retrieve":
            assert {"policy", "params", "kept", "left_out"} <= step.keys(), step
    if find_spec("dfo_descriptions") is not None:
        _validate_against_kit(steps)


@_KIT
def test_the_new_variable_operations_are_the_kit_s() -> None:
    from dfo_descriptions import schema_dir  # noqa: PLC0415

    operations = json.loads((schema_dir() / "operations.json").read_text())["operations"]
    assert {entry["op"] for entry in operations if entry["new_variable"]} == NEW_VARIABLE_OPS


_PROVIDER = "financingProvider"
_RECIPIENT = "financingRecipient"
_FLOW_ROLES = (_RECIPIENT, _PROVIDER, f"{_RECIPIENT}_party_kind", f"{_PROVIDER}_party_kind")


def _flows(
    providers: dict[str, str],
    recipients: tuple[str, ...],
    periods: tuple[str, ...],
    *,
    recipient_kinds: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Every provider's flow to every recipient in every period, each
    provider of its kind in `providers` and each recipient of its kind in
    `recipient_kinds`, a party by default."""
    return [
        {
            _RECIPIENT: recipient,
            f"{_RECIPIENT}_party_kind": (recipient_kinds or {}).get(recipient, "party"),
            _PROVIDER: provider,
            f"{_PROVIDER}_party_kind": kind,
            "time_period": period,
            "value": 1.0,
        }
        for recipient in recipients
        for provider, kind in providers.items()
        for period in periods
    ]


_TWO_DONORS_AND_DAC = {"d/A": "party", "d/B": "party", "d/DAC": "total"}
_KEN_GTM = ("country/KEN", "country/GTM")


@pytest.mark.parametrize(
    ("verb", "denominator_totals", "left_out"),
    [
        # Both sides of one frame leave out the one provider total: counted once.
        (share_of, None, 1),
        (quotient_of, None, 1),
        # The denominator's own total joins the numerator's: two distinct totals.
        (share_of, {"d/ALL": "total"}, 2),
        # The same total on both frames is one party.
        (share_of, {"d/DAC": "total"}, 1),
    ],
)
async def test_a_two_sided_step_counts_each_pooled_party_total_once(
    store, verb, denominator_totals: dict[str, str] | None, left_out: int
) -> None:
    """Two recipients funded by two donors and the DAC total over two years."""
    frame_ref = await _root(
        store, ODA, _flows(_TWO_DONORS_AND_DAC, _KEN_GTM, ("2020", "2021")), roles=_FLOW_ROLES
    )
    denominator = (
        {}
        if denominator_totals is None
        else {
            "denominator_frame_ref": await _root(
                store,
                ODA,
                _flows({"d/A": "party", **denominator_totals}, _KEN_GTM, ("2020", "2021")),
                roles=_FLOW_ROLES,
            )
        }
    )

    result = await verb(frame_ref=frame_ref, numerator_by=[_RECIPIENT], **denominator)

    step = await _own_step(store, result)
    assert step["kept"]["entities"] == 2
    assert step["left_out"] == [
        {"reason": "party_total", "count": left_out, "of": "pooled_entities", "role": _PROVIDER}
    ]


@pytest.mark.parametrize(
    ("recipient_kinds", "group_by", "kept", "pooled"),
    [
        (None, [_RECIPIENT], {"groups": 2, "entities": 2, "periods": 2}, 2),
        # A recipient's kind describes it: each year still pools the two donors.
        (
            {"country/GTM": "residual"},
            [_RECIPIENT],
            {"groups": 2, "entities": 2, "periods": 2},
            2,
        ),
        # Grouped by kind alone, a total holds no entity and pools every flow.
        (
            {"country/GTM": "residual"},
            [f"{_RECIPIENT}_party_kind"],
            {"groups": 2, "periods": 2},
            4,
        ),
    ],
)
async def test_aggregate_names_the_role_whose_party_totals_it_pools_out(
    store,
    recipient_kinds: dict[str, str] | None,
    group_by: list[str],
    kept: dict[str, int],
    pooled: int,
) -> None:
    frame_ref = await _root(
        store,
        ODA,
        _flows(_TWO_DONORS_AND_DAC, _KEN_GTM, ("2020", "2021"), recipient_kinds=recipient_kinds),
        roles=_FLOW_ROLES,
    )

    result = await aggregate(frame_ref=frame_ref, group_by=group_by, aggregation="sum")

    step = await _own_step(store, result)
    assert step["kept"] == kept
    assert step["period_coverage"] == [
        {"period": "2020", "entities": pooled},
        {"period": "2021", "entities": pooled},
    ]
    assert step["left_out"] == [
        {"reason": "party_total", "count": 1, "of": "pooled_entities", "role": _PROVIDER}
    ]
