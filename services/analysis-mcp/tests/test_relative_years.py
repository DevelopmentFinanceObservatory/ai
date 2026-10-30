"""A window bound or a `time_period` threshold given as a relative year,
`{anchor: "latest", offset: n}`, reaches the runner as it is given, with no
rolling overlay and no anchor roots, and the digest names the latest year the
runner filled it from (`resolved_periods`). A call over derived inputs with
`rolling_follows_tree`, as the public server makes every call, sends
`rolling: "tree"`, so each input's own rolling years follow the call's one
latest year, and the digest names each input computed again under another year.

The frames are held as `get_data` holds a fetch with no options, and the
runner's answers are replayed from `fixtures/runner_replies.json`, recorded by
dfo-apis' own runner.
"""

from __future__ import annotations

from collections.abc import Iterator
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
from runner_replay import ReplayedRunner, frame_digest

_NINE_BACK = {"anchor": "latest", "offset": -9}
_LATEST = {"anchor": "latest", "offset": 0}


@pytest.fixture
def store() -> Iterator[InMemoryMaterialisationStore]:
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


async def _held(store: InMemoryMaterialisationStore, stat_var: str, years: range) -> dict[str, str]:
    """One series of `stat_var` for Kenya over `years`, held under a pair named by
    its bytes, as `get_data` holds a fetch."""
    frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": stat_var,
                "observationAbout": "dcid:country/KEN",
                "time_period": str(year),
                "value": float(year - 2000),
                "unit": "dcid:USDollar",
            }
            for year in years
        ],
    )
    digest = frame_digest(frame)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_trace(ref), writer="test")
    return ref.model_dump()


def _trace(ref: FrameRef) -> MaterialisationTrace:
    return MaterialisationTrace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": ref.spec_id}],
            "facets": [],
            "conversion": None,
        },
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )


def _pair(structured: dict[str, Any] | None) -> dict[str, str]:
    assert structured is not None
    return {key: structured[key] for key in ("spec_id", "content_version")}


async def test_a_relative_window_reaches_the_runner_as_given_and_the_digest_names_its_years(
    store: InMemoryMaterialisationStore, runner: ReplayedRunner
) -> None:
    oda = await _held(store, "dcid:ODA", range(2010, 2025))

    result = await verbs.growth(frame_ref=oda, window_start=_NINE_BACK, window_end=_LATEST)

    assert result.is_error is False, result.content
    request = runner.requests[-1]
    assert request["tree"]["arguments"]["window_start"] == _NINE_BACK
    assert request["tree"]["arguments"]["window_end"] == _LATEST
    assert "rolling" not in request["tree"]["arguments"]
    assert not {"rolling", "anchor_roots"} & set(request)
    text = result.content[0].text
    assert text.startswith("growth: 2015..2024")
    assert (
        "Rolling: the latest year is 2024, the newest year at least half of the series "
        "counted hold; latest-9 is 2015, latest is 2024."
    ) in text


async def test_a_relative_threshold_reaches_the_runner_as_given(
    store: InMemoryMaterialisationStore, runner: ReplayedRunner
) -> None:
    oda = await _held(store, "dcid:ODA", range(2010, 2025))
    threshold = {"anchor": "latest", "offset": -2}

    result = await verbs.filter_verb(
        frame_ref=oda,
        predicates=(
            {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": threshold},
        ),
    )

    assert result.is_error is False, result.content
    (predicate,) = runner.requests[-1]["tree"]["arguments"]["predicates"]
    assert predicate["value"] == threshold
    assert "rolling" not in predicate
    assert "latest-2 is 2022" in result.content[0].text


async def test_a_combination_of_rolling_frames_computes_both_under_one_latest_year(
    store: InMemoryMaterialisationStore, runner: ReplayedRunner
) -> None:
    # Alone, ODA's latest year is 2024 and GDP's is 2025. Together it is 2024.
    oda = await _held(store, "dcid:ODA", range(2010, 2025))
    gdp = await _held(store, "dcid:GDP", range(2010, 2026))
    oda_growth = await verbs.growth(frame_ref=oda, window_start=_NINE_BACK, window_end=_LATEST)
    gdp_growth = await verbs.growth(frame_ref=gdp, window_start=_NINE_BACK, window_end=_LATEST)
    assert "growth: 2016..2025" in gdp_growth.content[0].text
    gdp_pair = _pair(gdp_growth.structured_content)

    result = await verbs.compare(
        frame_ref=_pair(oda_growth.structured_content),
        right_frame_ref=gdp_pair,
        rolling_follows_tree=True,
    )

    assert result.is_error is False, result.content
    assert runner.requests[-1]["rolling"] == "tree"
    text = result.content[0].text
    assert "Rolling: the latest year is 2024" in text
    assert (
        f"growth {gdp_pair['spec_id']}, {gdp_pair['content_version']} (alone, latest 2025)"
    ) in text
    assert _pair(oda_growth.structured_content)["spec_id"] not in text.split("Rolling:")[1]


_PLACE_TYPE = "observationAbout_type"


async def _world_and_kenya(store: InMemoryMaterialisationStore) -> dict[str, str]:
    """GDP of the world to 2025 and of Kenya to 2024, held as the public `get_data`
    holds a fetch of its place-type column: the dataset version, and the frame
    beside it under a mint record."""
    frame = Frame.build(
        role_keys=["observationAbout", _PLACE_TYPE],
        rows=[
            {
                "stat_var": "dcid:GDP",
                "observationAbout": place,
                _PLACE_TYPE: types,
                "time_period": str(year),
                "value": float(year - 2000),
                "unit": "dcid:USDollar",
            }
            for place, types, last in (("Earth", "Place", 2025), ("country/KEN", "Country", 2024))
            for year in range(2018, last + 1)
        ],
    )
    digest = frame_digest(frame)
    dataset = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    held = FrameRef(spec_id=dataset.spec_id, content_version=f"cv_{digest[::-1]}")
    for ref in (dataset, held):
        await store.put(ref, frame=frame, trace=_trace(ref), writer="test")
    await store.put_mint_record(
        MintRecord(
            frame=held,
            dataset=dataset,
            options={
                "format": "parquet",
                "bundle": False,
                "layout": "frame",
                "columns": [_PLACE_TYPE],
            },
        )
    )
    return held.model_dump()


async def test_a_rolling_rank_at_one_place_level_takes_its_latest_year_after_the_narrowing(
    store: InMemoryMaterialisationStore, runner: ReplayedRunner
) -> None:
    # Over every place the world's 2025 is the latest year one of two series
    # holds; over the countries alone it is Kenya's 2024.
    gdp = await _world_and_kenya(store)

    result = await verbs.rank(
        frame_ref=gdp,
        rank_by=("observationAbout",),
        aggregation="sum",
        window_start={"anchor": "latest", "offset": -1},
        window_end=_LATEST,
        place_level="Country",
        rolling_follows_tree=True,
    )

    assert result.is_error is False, result.content
    tree = runner.requests[-1]["tree"]
    assert tree["arguments"]["window_start"] == {"anchor": "latest", "offset": -1}
    assert tree["arguments"]["window_end"] == _LATEST
    (step,) = tree["inputs"].values()
    ((predicate,),) = [step["arguments"]["predicates"]]
    assert (step["verb"], predicate["kind"], predicate["level"]) == (
        "filter",
        "place_level",
        "Country",
    )
    assert "Rolling: the latest year is 2024" in result.content[0].text
