"""A frame fetched with a `breakdown` or a `stat_vars` sum is a verb input: its root
names the dataset version apis summed and the `groupby` it was summed with
(`runner_tree.root_node`), through each verb's shape.

The frames are the summed versions `get_data` holds, each under a mint record naming
the dataset version and the groupby, and the runner's answers are replayed from
`fixtures/runner_replies.json`, recorded by dfo-apis' own runner.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from analysis_mcp import verbs
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    set_store,
)
from dfo_frame_grammar import Frame
from fastmcp.tools import ToolResult
from runner_replay import ReplayedRunner, frame_digest

_AXIS = "reporter"
_SUM = {"by": ["time_period"], "op": "sum"}
_BREAKDOWN = {"by": [_AXIS, "time_period"], "op": "sum", "levels": {_AXIS: 1}}
_OPTIONS = {"format": "parquet", "bundle": False, "layout": "frame"}

type Roots = dict[str, dict[str, str]]
type Call = Callable[[Roots], Awaitable[ToolResult]]


@pytest.fixture
async def store():
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


def _grants(stat_var: str, *, members: tuple[str, ...] = ()) -> Frame:
    """A summed frame of grants over 2021 and 2022: one row per period, or one per period
    and member of the breakdown axis."""
    return Frame.build(
        role_keys=[_AXIS] if members else [],
        rows=[
            {
                "stat_var": stat_var,
                "time_period": year,
                "value": float(10 * (index + 1) + int(year) - 2020),
                "unit": "dcid:USDollar",
                **({_AXIS: member} if members else {}),
            }
            for index, member in enumerate(members or (None,))
            for year in ("2021", "2022")
        ],
    )


async def _fetch(
    store: InMemoryMaterialisationStore, frame: Frame, groupby: dict[str, Any]
) -> dict[str, str]:
    """`frame` as `get_data` holds a summed fetch: the dataset version apis summed, named
    by the frame's bytes, and the frame held beside it under the pair the caller gets."""
    digest = frame_digest(frame)
    dataset = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    held = FrameRef(spec_id=dataset.spec_id, content_version=f"cv_{digest[::-1]}")
    trace = MaterialisationTrace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": digest}],
            "facets": [],
            "conversion": None,
        },
        spec_id=held.spec_id,
        content_version=held.content_version,
    )
    for ref in (dataset, held):
        await store.put(ref, frame=frame, trace=trace, writer="test")
    await store.put_mint_record(
        MintRecord(frame=held, dataset=dataset, options={**_OPTIONS, "groupby": groupby})
    )
    return held.model_dump()


@pytest.fixture
async def roots(store: InMemoryMaterialisationStore) -> Roots:
    members = ("dcid:KEN", "dcid:NGA")
    return {
        "breakdown": await _fetch(store, _grants("dcid:Grants", members=members), _BREAKDOWN),
        "loans": await _fetch(store, _grants("dcid:Loans", members=members), _BREAKDOWN),
        "total": await _fetch(store, _grants("dcid:GrantsAndLoans"), _SUM),
    }


# Each verb shape, with the groupby each input of its tree carries.
_CASES: dict[str, tuple[Call, dict[str, dict[str, Any]]]] = {
    "aggregate": (
        lambda r: verbs.aggregate(frame_ref=r["breakdown"], aggregation="sum"),
        {"data": _BREAKDOWN},
    ),
    "rank": (
        lambda r: verbs.rank(frame_ref=r["breakdown"], rank_by=(_AXIS,), aggregation="sum"),
        {"data": _BREAKDOWN},
    ),
    "growth": (
        lambda r: verbs.growth(frame_ref=r["total"], window_start="2021", window_end="2022"),
        {"data": _SUM},
    ),
    "filter": (
        lambda r: verbs.filter_verb(
            frame_ref=r["breakdown"],
            predicates=(
                {"kind": "ordered_threshold", "column": "value", "op": "gte", "value": 20},
            ),
        ),
        {"data": _BREAKDOWN},
    ),
    "compare": (
        lambda r: verbs.compare(
            frame_ref=r["breakdown"], right_frame_ref=r["total"], left_entity={_AXIS: "dcid:KEN"}
        ),
        {"data": _SUM, "other": _BREAKDOWN},
    ),
    "sum_of": (
        lambda r: verbs.sum_of(
            frame_ref=r["breakdown"], right_frame_ref=r["total"], left_entity={_AXIS: "dcid:KEN"}
        ),
        {"data": _BREAKDOWN, "other": _SUM},
    ),
    "share_of": (
        lambda r: verbs.share_of(
            frame_ref=r["breakdown"],
            denominator_frame_ref=r["total"],
            numerator_entity={_AXIS: "dcid:KEN"},
        ),
        {"data": _BREAKDOWN, "other": _SUM},
    ),
    "quotient_of": (
        lambda r: verbs.quotient_of(
            frame_ref=r["breakdown"],
            denominator_frame_ref=r["total"],
            numerator_entity={_AXIS: "dcid:KEN"},
        ),
        {"data": _BREAKDOWN, "other": _SUM},
    ),
    "join": (
        lambda r: verbs.join(
            frame_ref=r["breakdown"], right_frame_ref=r["loans"], join_keys=(_AXIS,)
        ),
        {"data": _BREAKDOWN, "other": _BREAKDOWN},
    ),
}


@pytest.mark.parametrize("verb", _CASES)
async def test_a_breakdown_and_a_sum_are_roots_of_each_verb_shape(
    roots: Roots, runner: ReplayedRunner, verb: str
) -> None:
    call, groupbys = _CASES[verb]

    result = await call(roots)

    assert result.is_error is False, result.content
    inputs = runner.requests[-1]["tree"]["inputs"]
    assert {name: node["rendering"]["groupby"] for name, node in inputs.items()} == groupbys
