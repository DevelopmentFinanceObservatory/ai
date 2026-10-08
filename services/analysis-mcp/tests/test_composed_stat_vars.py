"""`Trace.composed_stat_vars` through the tools: `quotient_of`, `share_of`
and `sum_of` describe the StatVar they mint, and every registered verb's stored trace
describes each such StatVar its frame holds and each one those name.
`test_tool_pipeline.py` holds the rule itself.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from typing import NamedTuple

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.compare_tool import compare
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.growth_tool import growth
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.join_tool import join
from analysis_mcp.op_spec import AggregateOpSpec
from analysis_mcp.quotient_of_tool import quotient_of
from analysis_mcp.rank_tool import rank
from analysis_mcp.server import mcp
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.sum_of_tool import sum_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import (
    ComposedStatVar,
    FrameRef,
    InMemoryMaterialisationStore,
    set_store,
)
from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult

AID = "dcid:OfficialDevelopmentAssistanceNet_Disbursement"
POPULATION = "dcid:Count_Person"
USD = "dcid:USDollar"
KEN = "dcid:country/KEN"
GTM = "dcid:country/GTM"

type _Ref = dict[str, str]


@pytest.fixture
async def store():
    store = InMemoryMaterialisationStore()
    set_store(store)
    yield store
    set_store(None)


def _input_trace() -> Trace:
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
    store: InMemoryMaterialisationStore, stat_var: str, column: str, unit: str | None
) -> _Ref:
    """Kenya's and Guatemala's `stat_var` by `column`, 2020 to 2022."""
    frame = Frame.build(
        role_keys=[column],
        rows=[
            {
                "stat_var": stat_var,
                column: place,
                "time_period": period,
                "value": value * scale,
                "unit": unit,
            }
            for place, scale in ((KEN, 5.0), (GTM, 2.0))
            for period, value in (("2020", 10.0), ("2021", 12.0), ("2022", 15.0))
        ],
    )
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_input_trace(), writer="test")
    return ref.model_dump()


def _pair(result: ToolResult) -> _Ref:
    assert result.is_error is False, result.content[0].text
    return {
        "spec_id": result.structured_content["spec_id"],
        "content_version": result.structured_content["content_version"],
    }


def _stat_var(ref: _Ref) -> str:
    return f"dcid:{ref['spec_id']}"


async def _stored(store: InMemoryMaterialisationStore, ref: _Ref) -> tuple[Frame, Trace]:
    return await store.get(FrameRef(**ref))


class _Inputs(NamedTuple):
    """Aid per head by recipient, the same for Kenya alone and for Guatemala
    alone (three quotients, three StatVars), aid to the two summed, the aid
    frame and the population frame."""

    keyed: _Ref
    kenya: _Ref
    guatemala: _Ref
    total: _Ref
    aid: _Ref
    population: _Ref


async def _inputs(store: InMemoryMaterialisationStore) -> _Inputs:
    aid = await _seed(store, AID, "recipient", USD)
    population = await _seed(store, POPULATION, "observationAbout", None)

    async def per_head(place: str) -> _Ref:
        return _pair(
            await quotient_of(
                frame_ref=aid,
                denominator_frame_ref=population,
                numerator_entity={"recipient": place},
                denominator_entity={"observationAbout": place},
            )
        )

    keyed = _pair(
        await quotient_of(
            frame_ref=aid,
            denominator_frame_ref=population,
            numerator_by=["recipient"],
            denominator_by={"recipient": "observationAbout"},
        )
    )
    total = _pair(
        await sum_of(frame_ref=aid, left_entity={"recipient": KEN}, right_entity={"recipient": GTM})
    )
    return _Inputs(
        keyed=keyed,
        kenya=await per_head(KEN),
        guatemala=await per_head(GTM),
        total=total,
        aid=aid,
        population=population,
    )


def _per_head(ref: _Ref) -> ComposedStatVar:
    return ComposedStatVar(
        stat_var=_stat_var(ref), relation="quotient", first=(AID,), second=(POPULATION,)
    )


async def test_quotient_of_describes_its_own_stat_var(store: InMemoryMaterialisationStore) -> None:
    inputs = await _inputs(store)

    frame, trace = await _stored(store, inputs.kenya)

    assert {row[0] for row in frame.rows} == {_stat_var(inputs.kenya)}
    assert trace.composed_stat_vars == (_per_head(inputs.kenya),)


async def test_sum_of_describes_its_own_stat_var_by_each_sides_stat_vars(
    store: InMemoryMaterialisationStore,
) -> None:
    aid = await _seed(store, AID, "recipient", USD)
    other = await _seed(store, "dcid:OtherOfficialFlows", "recipient", USD)
    kenya = {"recipient": KEN}

    total = _pair(
        await sum_of(frame_ref=aid, right_frame_ref=other, left_entity=kenya, right_entity=kenya)
    )

    _frame, trace = await _stored(store, total)
    assert trace.composed_stat_vars == (
        ComposedStatVar(
            stat_var=_stat_var(total),
            relation="sum",
            first=(AID,),
            second=("dcid:OtherOfficialFlows",),
        ),
    )


async def test_share_of_describes_its_own_stat_var_by_each_sides_stat_var(
    store: InMemoryMaterialisationStore,
) -> None:
    """Aid to Kenya as a share of all aid: two slices of one frame, so both
    sides name the aid StatVar."""
    aid = await _seed(store, AID, "recipient", USD)

    share = _pair(await share_of(frame_ref=aid, numerator_entity={"recipient": KEN}))

    frame, trace = await _stored(store, share)
    assert {row[0] for row in frame.rows} == {_stat_var(share)}
    assert trace.composed_stat_vars == (
        ComposedStatVar(stat_var=_stat_var(share), relation="share", first=(AID,), second=(AID,)),
    )


def _aid_total(ref: _Ref) -> ComposedStatVar:
    return ComposedStatVar(stat_var=_stat_var(ref), relation="sum", first=(AID,), second=(AID,))


class _Carry(NamedTuple):
    """One verb reading an operation's result: the call, and the entries its
    stored trace carries from its inputs, in order. `own` adds the verb's own
    StatVar after them."""

    call: Callable[[_Inputs], Awaitable[ToolResult]]
    carried: Callable[[_Inputs], tuple[ComposedStatVar, ...]]
    own: bool = False


_WINDOW = {"window_start": "2020", "window_end": "2022", "entity_by": ("recipient",)}

_CARRIES: dict[str, _Carry] = {
    "aggregate": _Carry(
        lambda inputs: aggregate(
            frame_ref=inputs.keyed, aggregation="mean", group_by=("recipient",)
        ),
        lambda inputs: (_per_head(inputs.keyed),),
    ),
    "filter": _Carry(
        lambda inputs: filter_verb(frame_ref=inputs.keyed),
        lambda inputs: (_per_head(inputs.keyed),),
    ),
    "growth": _Carry(
        lambda inputs: growth(frame_ref=inputs.keyed, change="absolute", **_WINDOW),
        lambda inputs: (_per_head(inputs.keyed),),
    ),
    "rank": _Carry(
        lambda inputs: rank(frame_ref=inputs.keyed, rank_by=("recipient",), aggregation="mean"),
        lambda inputs: (_per_head(inputs.keyed),),
    ),
    "compare": _Carry(
        lambda inputs: compare(frame_ref=inputs.kenya, right_frame_ref=inputs.guatemala),
        lambda inputs: (_per_head(inputs.kenya),),
    ),
    # Both sides' rows stay, so both entries do, in StatVar order.
    "join": _Carry(
        lambda inputs: join(frame_ref=inputs.kenya, right_frame_ref=inputs.guatemala),
        lambda inputs: tuple(
            sorted(
                (_per_head(inputs.kenya), _per_head(inputs.guatemala)),
                key=lambda entry: entry.stat_var,
            )
        ),
    ),
    "share_of": _Carry(
        lambda inputs: share_of(frame_ref=inputs.kenya, denominator_frame_ref=inputs.guatemala),
        lambda inputs: (_per_head(inputs.kenya), _per_head(inputs.guatemala)),
        own=True,
    ),
    # A sum refuses a quotient as a side, so this one reads a sum.
    "sum_of": _Carry(
        lambda inputs: sum_of(
            frame_ref=inputs.total, right_frame_ref=inputs.aid, right_entity={"recipient": KEN}
        ),
        lambda inputs: (_aid_total(inputs.total),),
        own=True,
    ),
    # A quotient refuses a quotient as a side, so this one reads a sum.
    "quotient_of": _Carry(
        lambda inputs: quotient_of(
            frame_ref=inputs.total,
            denominator_frame_ref=inputs.population,
            denominator_entity={"observationAbout": KEN},
        ),
        lambda inputs: (_aid_total(inputs.total),),
        own=True,
    ),
}


async def test_every_registered_verb_has_a_carry_case() -> None:
    # `resolve_anchor` stores an anchor record and no frame.
    assert {tool.name for tool in await mcp.list_tools()} == set(_CARRIES) | {"resolve_anchor"}


@pytest.mark.parametrize("verb", sorted(_CARRIES))
async def test_a_verb_carries_the_entry_of_each_operation_stat_var_its_frame_keeps(
    store: InMemoryMaterialisationStore, verb: str
) -> None:
    """A two-sided verb whose frame keeps only its first side's StatVar
    (`compare`) carries that side's entry alone."""
    carry = _CARRIES[verb]
    inputs = await _inputs(store)

    result = _pair(await carry.call(inputs))

    frame, trace = await _stored(store, result)
    carried = carry.carried(inputs)
    entries = trace.composed_stat_vars
    assert entries[: len(carried)] == carried
    assert [entry.stat_var for entry in entries[len(carried) :]] == (
        [_stat_var(result)] if carry.own else []
    )
    held = {row[0] for row in frame.rows}
    named = {ref for entry in entries for ref in (*entry.first, *entry.second)}
    assert {entry.stat_var for entry in entries} <= held | named
    assert held & {_stat_var(ref) for ref in inputs[:4]} <= {entry.stat_var for entry in entries}


async def test_a_sum_over_a_sum_describes_it_before_itself(
    store: InMemoryMaterialisationStore,
) -> None:
    inputs = await _inputs(store)

    total = _pair(
        await sum_of(
            frame_ref=inputs.total, right_frame_ref=inputs.aid, right_entity={"recipient": KEN}
        )
    )
    filtered = _pair(await filter_verb(frame_ref=total))

    expected = (
        _aid_total(inputs.total),
        ComposedStatVar(
            stat_var=_stat_var(total),
            relation="sum",
            first=(_stat_var(inputs.total),),
            second=(AID,),
        ),
    )
    for ref in (total, filtered):
        frame, trace = await _stored(store, ref)
        assert {row[0] for row in frame.rows} == {_stat_var(total)}
        assert trace.composed_stat_vars == expected
