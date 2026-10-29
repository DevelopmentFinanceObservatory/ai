"""The frame a call names, as dfo-apis' runner computes it: a bare `spec_id`, a frame
the runner lost, and `place_level`. Each pair of calls here is one tree to the runner
under two spellings, so each asserts both get the same `ops_` ids, which dfo-apis
derives from the recipe's normal form.

The runner's answers are replayed from `fixtures/runner_replies.json`, recorded by
dfo-apis' own runner over the frames these tests seed (`runner_replay`).
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

_GDP = "dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_PLACE_TYPE = "observationAbout_type"
# The mint options the public `get_data` fetches a place frame with.
_FETCHED = {"format": "parquet", "bundle": False, "layout": "frame", "columns": [_PLACE_TYPE]}

# GDP at the world and at three countries: places of several levels.
_PLACES = {
    "Earth": ("Place", 105.0),
    "country/USA": ("Country", 27.0),
    "country/CHN": ("Country", 18.0),
    "country/FIN": ("Country", 0.3),
}


def _gdp() -> Frame:
    return Frame.build(
        role_keys=["observationAbout", _PLACE_TYPE],
        rows=[
            {
                "stat_var": _GDP,
                "observationAbout": place,
                _PLACE_TYPE: types,
                "time_period": year,
                "value": value + (int(year) - 2022),
                "unit": "USDollar",
            }
            for place, (types, value) in _PLACES.items()
            for year in ("2022", "2023")
        ],
    )


def _trace(ref: FrameRef) -> MaterialisationTrace:
    return MaterialisationTrace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": "1"}],
            "facets": [],
            "conversion": None,
        },
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )


@pytest.fixture
def store():
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


async def _fetch(store: InMemoryMaterialisationStore, *, recorded: bool = True) -> dict[str, str]:
    """`_gdp` as the public `get_data` holds it: the dataset version apis serves, and the
    frame of its place-type column beside it under the pair the caller gets, named by
    a mint record. Unless `recorded`, the store has lost the frame and its record, as
    after a flush, and holds the dataset version alone for the runner to read."""
    frame = _gdp()
    digest = frame_digest(frame)
    dataset = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    held = FrameRef(spec_id=dataset.spec_id, content_version=f"cv_{digest[::-1]}")
    await store.put(dataset, frame=frame, trace=_trace(dataset), writer="test")
    if recorded:
        await store.put(held, frame=frame, trace=_trace(held), writer="test")
        await store.put_mint_record(MintRecord(frame=held, dataset=dataset, options=_FETCHED))
    return held.model_dump()


def _ids(result: ToolResult) -> tuple[str, str]:
    assert result.is_error is False, result.content
    content = result.structured_content
    assert content is not None
    return content["spec_id"], content["content_version"]


def _dataset_version(runner: ReplayedRunner) -> str:
    """The version the last call's data root was read at."""
    return runner.requests[0]["tree"]["inputs"]["data"].get("content_version", "")


# A bare spec_id ----------------------------------------------------------------------


async def test_a_bare_spec_id_computes_what_the_frame_get_data_returns_computes(
    store: InMemoryMaterialisationStore, runner: ReplayedRunner
) -> None:
    fetched = await _fetch(store)
    window = {"window_start": "2022", "window_end": "2023", "entity_by": ("observationAbout",)}

    after_get_data = await verbs.growth(frame_ref=fetched, **window)
    dataset_version = _dataset_version(runner)
    bare = await verbs.growth(frame_ref={"spec_id": fetched["spec_id"]}, **window)

    assert runner.requests[-1]["tree"]["inputs"]["data"] == {"spec_id": fetched["spec_id"]}
    assert _ids(bare) == _ids(after_get_data)
    # The digest names the version the Spec was read at.
    assert isinstance(bare.content, list)
    assert dataset_version in bare.content[0].text


async def test_a_frame_lost_with_its_mint_record_is_fetched_again(
    store: InMemoryMaterialisationStore, runner: ReplayedRunner
) -> None:
    lost = await _fetch(store, recorded=False)
    window = {"window_start": "2022", "window_end": "2023", "entity_by": ("observationAbout",)}

    rebuilt = await verbs.growth(frame_ref=lost, **window)
    bare = await verbs.growth(frame_ref={"spec_id": lost["spec_id"]}, **window)

    assert [request["tree"]["inputs"]["data"] for request in runner.requests[:2]] == [
        {**lost, "rendering": {"without_party_totals": False, "columns": []}},
        {"spec_id": lost["spec_id"]},
    ]
    assert _ids(rebuilt) == _ids(bare)


# place_level ----------------------------------------------------------------------------

_COUNTRY = {"kind": "place_level", "column": _PLACE_TYPE, "level": "Country"}

type Verb = Callable[..., Awaitable[ToolResult]]

_VERBS: dict[str, tuple[Verb, dict[str, Any]]] = {
    "rank": (verbs.rank, {"rank_by": ("observationAbout",), "aggregation": "sum"}),
    "aggregate": (verbs.aggregate, {"aggregation": "sum", "group_by": ("time_period",)}),
    "share_of": (verbs.share_of, {"numerator_by": ("observationAbout",)}),
}


@pytest.mark.parametrize("verb", _VERBS)
async def test_place_level_computes_what_a_filter_then_the_verb_computes(
    store: InMemoryMaterialisationStore, verb: str
) -> None:
    fetched = await _fetch(store)
    call, arguments = _VERBS[verb]

    mixed = await call(frame_ref=fetched, **arguments)
    leveled = await call(frame_ref=fetched, place_level="Country", **arguments)
    filtered = await verbs.filter_verb(frame_ref=fetched, predicates=(_COUNTRY,))
    spec_id, content_version = _ids(filtered)
    then = await call(
        frame_ref={"spec_id": spec_id, "content_version": content_version}, **arguments
    )

    assert mixed.structured_content is not None
    assert (mixed.structured_content["code"], mixed.structured_content["fix"]["place_level"]) == (
        "place_levels_mixed",
        "Country",
    )
    assert _ids(leveled) == _ids(then)
