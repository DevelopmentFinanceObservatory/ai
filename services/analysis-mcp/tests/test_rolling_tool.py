"""`resolve_anchor`, the check `growth`, `rank` and `filter` run on a rolling
call, and a rolling recipe from its roots to the anchor it is published with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.growth_tool import growth
from analysis_mcp.join_tool import join
from analysis_mcp.rank_tool import rank
from analysis_mcp.rolling_tool import resolve_anchor
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    set_store,
)
from common.recipe import RecipeAnchorError, build_recipe
from common.rolling import anchor_key, anchor_populations, canonical_populations
from dfo_frame_grammar import Frame, shift_period
from fastmcp.tools import ToolResult
from rolling_support import LATEST, anchored

_OPTIONS = {"format": "parquet", "bundle": False, "layout": "frame"}
_AID = "spec_" + "a" * 32
_STAT_VAR = "dcid:OfficialDevelopmentAssistanceNet_Disbursement"
_DONORS = ("dcid:AUS", "dcid:BEL", "dcid:CAN", "dcid:DEU")


@pytest.fixture
async def store():
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


def _frame(
    years_by_donor: Mapping[str, Sequence[int]],
    *,
    unit: str | None = "dcid:USD",
    stat_var: str = _STAT_VAR,
) -> Frame:
    return Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": stat_var,
                "reporter": donor,
                "time_period": str(year),
                "value": float(year - 2000),
                "unit": unit,
            }
            for donor, years in years_by_donor.items()
            for year in years
        ],
    )


def _ref(version: str, spec_id: str = _AID) -> FrameRef:
    return FrameRef(spec_id=spec_id, content_version="cv_" + version * 32)


async def _register(
    store: InMemoryMaterialisationStore, ref: FrameRef, frame: Frame
) -> dict[str, str]:
    """`frame` as a root `get_data` registered under `ref`, with its mint record."""
    trace = MaterialisationTrace(
        upstream_trace={
            "graph_revision": "rev_test",
            "source_revisions": [{"source": "test", "revision": ref.content_version}],
            "facets": [],
            "conversion": None,
        },
        spec_id=ref.spec_id,
        content_version=ref.content_version,
    )
    await store.put(ref, frame=frame, trace=trace, writer="test")
    await store.put_mint_record(MintRecord(frame=ref, dataset=ref, options=_OPTIONS))
    return ref.model_dump()


def _pair(result: ToolResult) -> dict[str, str]:
    assert result.is_error is False, result.content
    content = result.structured_content
    return {"spec_id": content["spec_id"], "content_version": content["content_version"]}


def _refused(result: ToolResult) -> tuple[str, str | None]:
    assert result.is_error is True
    content = result.structured_content
    return content["code"], content["fix"].get("detail")


# resolve_anchor -----------------------------------------------------------------


async def test_the_anchor_is_the_newest_year_half_of_the_bound_series_hold(store) -> None:
    # Of the three donors bound, one holds 2023 and two hold 2022. The fourth
    # donor holds 2024 and is no part of the population.
    root = await _register(
        store,
        _ref("1"),
        _frame(
            {
                "dcid:AUS": [2021, 2022, 2023],
                "dcid:BEL": [2021, 2022],
                "dcid:CAN": [2021],
                "dcid:DEU": [2021, 2024],
            }
        ),
    )
    populations = [{"spec_id": _AID, "bindings": {"reporter": list(_DONORS[:3])}}]
    frames_before = await store.list_frames()

    result = await resolve_anchor(populations=populations, roots=[root])

    assert result.is_error is False, result.content
    assert result.structured_content == {
        "anchor_key": anchor_key(populations=populations, roots=[root], rule="half1"),
        "period": "2022",
        "left_out": [
            {
                "root": 0,
                "coordinates": {
                    "stat_var": _STAT_VAR,
                    "reporter": "dcid:CAN",
                    "unit": "dcid:USD",
                },
            }
        ],
    }
    # It stores the record and no frame.
    record = await store.get_anchor_record(result.structured_content["anchor_key"])
    assert record is not None
    assert record.model_dump(mode="json") == {
        "populations": populations,
        "roots": [root],
        "rule": "half1",
        "period": "2022",
        "left_out": result.structured_content["left_out"],
    }
    assert await store.list_frames() == frames_before


async def test_a_series_is_every_column_but_period_value_and_status(store) -> None:
    # One donor under two units is two series, and a column a series holds no
    # value in is left out of its coordinates.
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR,
                "reporter": "dcid:AUS",
                "time_period": year,
                "value": 1.0,
                "unit": unit,
            }
            for unit, years in (("dcid:USD", ["2022", "2023"]), (None, ["2022"]))
            for year in years
        ],
    )
    root = await _register(store, _ref("1"), frame)

    result = await resolve_anchor(populations=[{"spec_id": _AID}], roots=[root])

    assert result.structured_content["period"] == "2023"
    assert result.structured_content["left_out"] == [
        {"root": 0, "coordinates": {"stat_var": _STAT_VAR, "reporter": "dcid:AUS"}}
    ]


async def test_a_left_out_series_names_a_composed_unit_by_its_canonical_text(store) -> None:
    # dfo-apis groups a series under a composed unit by the unit's RFC 8785 text.
    per_head = {"numerator": "dcid:USD", "denominator": {"stat_var": "dcid:Count_Person"}}
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR,
                "reporter": donor,
                "time_period": year,
                "value": 1.0,
                "unit": per_head,
            }
            for donor, years in (("dcid:AUS", ["2022", "2023"]), ("dcid:BEL", ["2022"]))
            for year in years
        ],
    )
    root = await _register(store, _ref("1"), frame)

    result = await resolve_anchor(populations=[{"spec_id": _AID}], roots=[root])

    assert result.structured_content["left_out"] == [
        {
            "root": 0,
            "coordinates": {
                "stat_var": _STAT_VAR,
                "reporter": "dcid:BEL",
                "unit": '{"denominator":{"stat_var":"dcid:Count_Person"},"numerator":"dcid:USD"}',
            },
        }
    ]


async def test_a_population_is_counted_over_every_root_frame_of_its_dataset(store) -> None:
    older = await _register(store, _ref("1"), _frame({"dcid:AUS": [2022], "dcid:BEL": [2022]}))
    newer = await _register(
        store, _ref("2"), _frame({"dcid:AUS": [2022, 2023], "dcid:BEL": [2022, 2023]})
    )
    populations = [{"spec_id": _AID}]

    over_both = await resolve_anchor(populations=populations, roots=[newer, older])
    over_newer = await resolve_anchor(populations=populations, roots=[newer])

    # 2023 is held by no series of the older frame, so it is no anchor over both.
    assert over_both.structured_content["period"] == "2022"
    assert over_newer.structured_content["period"] == "2023"
    assert over_both.structured_content["anchor_key"] != over_newer.structured_content["anchor_key"]


@pytest.mark.parametrize(
    ("populations", "code", "detail"),
    [
        pytest.param(
            [{"spec_id": "spec_" + "b" * 32}, {"spec_id": _AID}],
            "rolling_anchor_unresolved",
            "population_without_root",
            id="population-with-no-root",
        ),
        pytest.param(
            [{"spec_id": _AID, "bindings": {"recipient": ["dcid:GHA"]}}],
            "rolling_anchor_unresolved",
            "binding_column_unknown",
            id="binding-on-a-column-the-root-lacks",
        ),
        pytest.param(
            [{"spec_id": _AID, "bindings": {"time_period": ["2022"]}}],
            "rolling_anchor_unresolved",
            "binding_unusable",
            id="binding-on-the-period",
        ),
        pytest.param(
            [{"spec_id": _AID, "bindings": {"reporter": ["dcid:ZZZ"]}}],
            "rolling_anchor_unresolved",
            "no_qualifying_period",
            id="bindings-keep-no-series",
        ),
        pytest.param(
            [{"spec_id": _AID, "content_version": "cv_" + "1" * 32}],
            "op_spec_invalid",
            None,
            id="population-naming-a-version",
        ),
    ],
)
async def test_resolve_anchor_refuses_what_it_cannot_count(
    store, populations: list[dict[str, Any]], code: str, detail: str | None
) -> None:
    root = await _register(store, _ref("1"), _frame({"dcid:AUS": [2022]}))

    result = await resolve_anchor(populations=populations, roots=[root])

    refused_code, refused_detail = _refused(result)
    assert refused_code == code
    # `op_spec_invalid` states its reason in words.
    assert detail is None or refused_detail == detail
    assert store._anchor_records == {}


async def test_a_root_frame_of_no_population_refuses(store) -> None:
    root = await _register(store, _ref("1"), _frame({"dcid:AUS": [2022]}))
    other = await _register(store, _ref("1", "spec_" + "b" * 32), _frame({"dcid:AUS": [2022]}))

    result = await resolve_anchor(populations=[{"spec_id": _AID}], roots=[root, other])

    assert _refused(result) == ("rolling_anchor_unresolved", "root_without_population")


async def test_a_root_frame_the_store_does_not_hold_refuses_as_any_input_does(store) -> None:
    result = await resolve_anchor(populations=[{"spec_id": _AID}], roots=[_ref("1").model_dump()])

    assert _refused(result)[0] == "frame_not_held"


@pytest.mark.parametrize(
    "periods", [["2022-01", "2022-02"], ["2018/2022"]], ids=["months", "one-span"]
)
async def test_periods_that_are_no_calendar_year_refuse(store, periods: list[str]) -> None:
    frame = Frame.build(
        role_keys=["reporter"],
        rows=[
            {
                "stat_var": _STAT_VAR,
                "reporter": "dcid:AUS",
                "time_period": period,
                "value": 1.0,
                "unit": "dcid:USD",
            }
            for period in periods
        ],
    )
    root = await _register(store, _ref("1"), frame)

    result = await resolve_anchor(populations=[{"spec_id": _AID}], roots=[root])

    assert _refused(result)[0] == "rolling_grain_unsupported"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    assert result.structured_content["fix"]["periods"] == periods


# The check on a rolling verb call ------------------------------------------------


def _growth_call(root: Mapping[str, str], *, end: str, **rolling: Any) -> dict[str, Any]:
    return {
        "frame_ref": root,
        "window_start": shift_period(end, -2),
        "window_end": end,
        "entity_by": ("reporter",),
        **rolling,
    }


def _rank_call(root: Mapping[str, str], *, end: str, **rolling: Any) -> dict[str, Any]:
    return {
        "frame_ref": root,
        "rank_by": ("reporter",),
        "aggregation": "sum",
        "window_start": shift_period(end, -2),
        "window_end": end,
        **rolling,
    }


def _window(populations: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        "rolling": {
            "window_start": {**LATEST, "offset": -2},
            "window_end": LATEST,
            "population": populations,
        },
        "anchor_key": key,
    }


def _filter_call(
    root: Mapping[str, str], *, end: str, populations: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    since = {
        "kind": "ordered_threshold",
        "column": "time_period",
        "op": "gte",
        "value": shift_period(end, -2),
        "rolling": {**LATEST, "offset": -2},
    }
    return {
        "frame_ref": root,
        "predicates": (since,),
        "rolling": {"population": populations},
        "anchor_key": key,
    }


async def _rolling_call(
    verb: str, root: Mapping[str, str], *, end: str, populations: list[dict[str, Any]], key: str
) -> ToolResult:
    if verb == "filter":
        return await filter_verb(**_filter_call(root, end=end, populations=populations, key=key))
    if verb == "growth":
        return await growth(**_growth_call(root, end=end, **_window(populations, key)))
    return await rank(**_rank_call(root, end=end, **_window(populations, key)))


_VERBS = pytest.mark.parametrize("verb", ["growth", "rank", "filter"])


@_VERBS
async def test_a_rolling_call_runs_on_literals_written_from_its_anchor(store, verb: str) -> None:
    handles = await seed_fixture_store(store)
    anchor = await anchored(handles.plain_single_unit)

    result = await _rolling_call(
        verb,
        handles.plain_single_unit,
        end=anchor.period,
        populations=anchor.populations,
        key=anchor.anchor_key,
    )

    assert result.is_error is False, result.content


@_VERBS
@pytest.mark.parametrize(
    "fault",
    ["anchor_record_missing", "populations_differ", "root_not_in_record", "literal_differs"],
)
async def test_a_rolling_call_its_anchor_record_does_not_cover_refuses_and_writes_nothing(
    store, verb: str, fault: str
) -> None:
    handles = await seed_fixture_store(store)
    root = handles.plain_single_unit
    anchor = await anchored(root)
    call = {"end": anchor.period, "populations": anchor.populations, "key": anchor.anchor_key}
    if fault == "anchor_record_missing":
        call["key"] = "rak_" + "e" * 32
    elif fault == "populations_differ":
        call["populations"] = [{"spec_id": root["spec_id"], "bindings": {"reporter": ["dcid:USA"]}}]
    elif fault == "root_not_in_record":
        # The record covers the sparse fixture's dataset under these populations,
        # and the call reads the plain one.
        call["key"] = (await anchored(handles.sparse_coverage)).anchor_key
        call["populations"] = [{"spec_id": handles.sparse_coverage["spec_id"]}]
    else:
        call["end"] = shift_period(anchor.period, -1)
    frames_before = await store.list_frames()

    result = await _rolling_call(verb, root, **call)

    assert _refused(result) == ("rolling_anchor_unresolved", fault)
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert await store.list_frames() == frames_before


async def test_a_fixed_call_reads_no_anchor_record(store, monkeypatch: pytest.MonkeyPatch) -> None:
    handles = await seed_fixture_store(store)
    root = handles.plain_single_unit
    reads: list[str] = []

    async def _read(key: str) -> None:
        reads.append(key)

    monkeypatch.setattr(store, "get_anchor_record", _read)

    since = {"kind": "ordered_threshold", "column": "time_period", "op": "gte", "value": "2022"}
    results = [
        await filter_verb(frame_ref=root, predicates=(since,)),
        await growth(**_growth_call(root, end="2023")),
        await rank(**_rank_call(root, end="2023")),
    ]

    assert [result.is_error for result in results] == [False, False, False]
    assert reads == []


async def test_the_roots_of_a_derived_input_are_found_from_traces_alone(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    handles = await seed_fixture_store(store)
    root = handles.plain_single_unit
    anchor = await anchored(root)
    kept = _pair(
        await filter_verb(
            **_filter_call(
                root, end=anchor.period, populations=anchor.populations, key=anchor.anchor_key
            )
        )
    )
    summed = _pair(
        await aggregate(frame_ref=kept, aggregation="sum", group_by=("reporter", "time_period"))
    )
    traces_read: list[dict[str, str]] = []
    frames_opened: list[dict[str, str]] = []
    get_trace, open_frame = store.get_trace, store.open

    async def _get_trace(ref: FrameRef):
        traces_read.append(ref.model_dump())
        return await get_trace(ref)

    async def _open(ref: FrameRef):
        frames_opened.append(ref.model_dump())
        return await open_frame(ref)

    monkeypatch.setattr(store, "get_trace", _get_trace)
    monkeypatch.setattr(store, "open", _open)

    grown = await growth(
        **_growth_call(summed, end=anchor.period, **_window(anchor.populations, anchor.anchor_key))
    )

    assert grown.is_error is False, grown.content
    # The call's input is read once, with its rows. The filter result under it is
    # read for its trace alone, and the root, known by its id, is not read.
    assert frames_opened == [summed]
    assert traces_read == [kept]


async def test_a_derived_input_over_a_root_outside_the_record_refuses(store) -> None:
    handles = await seed_fixture_store(store)
    anchor = await anchored(handles.sparse_coverage)
    summed = _pair(
        await aggregate(
            frame_ref=handles.plain_single_unit,
            aggregation="sum",
            group_by=("reporter", "time_period"),
        )
    )

    result = await growth(
        **_growth_call(summed, end=anchor.period, **_window(anchor.populations, anchor.anchor_key))
    )

    assert _refused(result) == ("rolling_anchor_unresolved", "root_not_in_record")
    assert result.structured_content["fix"]["root"] == handles.plain_single_unit


# A rolling recipe, from its roots to its anchor ----------------------------------

_SELECTION = {"reporter": list(_DONORS), "recipient": ["dcid:GHA"]}
_KEPT = {
    "kind": "category_match",
    "column": "reporter",
    "values": [*_DONORS, "dcid:ESP"],
}


def _planned_tree(since: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    """The recipe as a caller writes it before any frame exists: growth over a
    filter over one root."""
    kept = {
        "op_spec": {"verb": "filter", "parameters": {"predicates": [_KEPT, since]}},
        "inputs": {"frame_ref": {"root": 0}},
    }
    return {
        "op_spec": {"verb": "growth", "parameters": window},
        "inputs": {"frame_ref": kept},
    }


async def _rolling_recipe(
    store: InMemoryMaterialisationStore, ref: FrameRef, frame: Frame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Registers `frame` as the root `ref`, resolves the anchor, runs a rolling
    filter then a rolling growth on literals written from it, and returns the
    recipe of the result beside what `resolve_anchor` answered."""
    root = await _register(store, ref, frame)
    mint = MintRecord(frame=ref, dataset=ref, options=_OPTIONS).model_dump(mode="json")
    since = {
        "kind": "ordered_threshold",
        "column": "time_period",
        "op": "gte",
        "rolling": {**LATEST, "offset": -3},
    }
    window = {"window_start": {**LATEST, "offset": -1}, "window_end": LATEST}
    populations = anchor_populations(
        _planned_tree(since, {"rolling": window}),
        [mint],
        root_columns=[frame.columns],
        selection=_SELECTION,
    )
    assert populations == [{"spec_id": _AID, "bindings": {"reporter": sorted(_DONORS)}}]

    resolved = await resolve_anchor(populations=populations, roots=[root])
    assert resolved.is_error is False, resolved.content
    anchor = resolved.structured_content
    key, latest = anchor["anchor_key"], anchor["period"]

    kept = await filter_verb(
        frame_ref=root,
        predicates=(_KEPT, {**since, "value": shift_period(latest, -3)}),
        rolling={"population": populations},
        anchor_key=key,
    )
    grown = await growth(
        frame_ref=_pair(kept),
        window_start=shift_period(latest, -1),
        window_end=latest,
        entity_by=("reporter",),
        rolling={**window, "population": populations},
        anchor_key=key,
    )
    recipe = await build_recipe(FrameRef(**_pair(grown)), store=store)

    # The populations the two op-specs hashed are the ones the finished tree gives.
    assert recipe["tree"]["op_spec"]["parameters"]["rolling"]["population"] == populations
    assert (
        anchor_populations(
            recipe["tree"], recipe["roots"], root_columns=[frame.columns], selection=_SELECTION
        )
        == populations
    )
    return recipe, anchor


def _years(*, newest: Mapping[str, int]) -> Frame:
    """Each donor's years from 2018 to its newest. `dcid:ESP` passes the filter
    and is outside the card's selection, and `dcid:FIN` is outside both."""
    return _frame({donor: range(2018, last + 1) for donor, last in newest.items()})


_OUTSIDE = {"dcid:ESP": 2025, "dcid:FIN": 2025}


async def test_a_rolling_recipe_is_published_with_the_anchor_its_calls_were_checked_against(
    store,
) -> None:
    # Three of the four donors selected hold 2022.
    base = _years(
        newest={"dcid:AUS": 2022, "dcid:BEL": 2022, "dcid:CAN": 2022, "dcid:DEU": 2021, **_OUTSIDE}
    )
    # One more year, held by one of the four.
    under_half = _years(
        newest={"dcid:AUS": 2023, "dcid:BEL": 2022, "dcid:CAN": 2022, "dcid:DEU": 2021, **_OUTSIDE}
    )
    # The same year, held by two of the four.
    half = _years(
        newest={"dcid:AUS": 2023, "dcid:BEL": 2023, "dcid:CAN": 2022, "dcid:DEU": 2021, **_OUTSIDE}
    )

    first, first_anchor = await _rolling_recipe(store, _ref("1"), base)
    held, held_anchor = await _rolling_recipe(store, _ref("2"), under_half)
    moved, moved_anchor = await _rolling_recipe(store, _ref("3"), half)

    def _left_out(*donors: str) -> list[dict[str, Any]]:
        return [
            {
                "root": 0,
                "coordinates": {"stat_var": _STAT_VAR, "reporter": donor, "unit": "dcid:USD"},
            }
            for donor in donors
        ]

    assert first["anchor"] == {"period": "2022", "rule": "half1", "left_out": _left_out("dcid:DEU")}
    assert held["anchor"] == first["anchor"]
    assert moved["anchor"] == {
        "period": "2023",
        "rule": "half1",
        "left_out": _left_out("dcid:CAN", "dcid:DEU"),
    }
    for recipe, anchor in ((first, first_anchor), (held, held_anchor), (moved, moved_anchor)):
        assert recipe["anchor"]["period"] == anchor["period"]
        assert recipe["anchor"]["left_out"] == anchor["left_out"]
    # The literals follow the anchor, and the recipe is one recipe throughout.
    assert first["tree"]["op_spec"]["parameters"]["window_end"] == "2022"
    assert moved["tree"]["op_spec"]["parameters"]["window_end"] == "2023"
    assert first["normal_form_hash"] == held["normal_form_hash"] == moved["normal_form_hash"]
    assert len({first["replay_key"], held["replay_key"], moved["replay_key"]}) == 3


# A population held to one level of place ----------------------------------------

_PLACE_TYPE = "observationAbout_type"
_COUNTRY_LEVEL = {"kind": "place_level", "column": _PLACE_TYPE, "level": "Country"}
_REGIONS = {"dcid:WLD": "Place", "dcid:SSA": "UNGeoRegion", "dcid:LIC": "ProvisionalNode"}


def _places(newest: Mapping[str, int], types: Mapping[str, str]) -> Frame:
    """Each place's years from 2018 to its newest, with its place-type cell."""
    return Frame.build(
        role_keys=["observationAbout", _PLACE_TYPE],
        rows=[
            {
                "stat_var": _STAT_VAR,
                "observationAbout": place,
                _PLACE_TYPE: types[place],
                "time_period": str(year),
                "value": float(year - 2000),
                "unit": "dcid:USD",
            }
            for place, last in newest.items()
            for year in range(2018, last + 1)
        ],
    )


def _mixed(*, ghana: str = "Country") -> Frame:
    """Three countries, one of them at 2024, beside three aggregates all at 2024."""
    countries = {"dcid:GHA": ghana, "dcid:KEN": "Country", "dcid:TCD": "Country"}
    return _places(
        {"dcid:GHA": 2024, "dcid:KEN": 2023, "dcid:TCD": 2022, **dict.fromkeys(_REGIONS, 2024)},
        {**countries, **_REGIONS},
    )


async def test_a_level_holds_the_population_to_the_places_the_filter_keeps(store) -> None:
    root = await _register(store, _ref("1"), _mixed())
    levelled = [{"spec_id": _AID, "levels": {_PLACE_TYPE: "Country"}}]

    every_place = await resolve_anchor(populations=[{"spec_id": _AID}], roots=[root])
    countries = await resolve_anchor(populations=levelled, roots=[root])

    # Four of the six places hold 2024, and one of the three countries.
    assert every_place.structured_content["period"] == "2024"
    assert countries.structured_content["period"] == "2023"
    assert [
        series["coordinates"]["observationAbout"]
        for series in countries.structured_content["left_out"]
    ] == ["dcid:TCD"]
    assert (
        countries.structured_content["anchor_key"] != every_place.structured_content["anchor_key"]
    )
    record = await store.get_anchor_record(countries.structured_content["anchor_key"])
    assert record is not None
    assert record.model_dump(mode="json")["populations"] == [
        {"spec_id": _AID, "bindings": {}, "levels": {_PLACE_TYPE: "Country"}}
    ]

    # The series counted are the ones the `filter` step keeps: its result, held
    # as a root of its own, gives the same period and leaves the same series out.
    kept, _trace = await store.get(
        FrameRef(**_pair(await filter_verb(frame_ref=root, predicates=(_COUNTRY_LEVEL,))))
    )
    other = "spec_" + "b" * 32
    kept_root = await _register(store, _ref("1", other), kept)
    over_kept = await resolve_anchor(populations=[{"spec_id": other}], roots=[kept_root])
    assert over_kept.structured_content["period"] == countries.structured_content["period"]
    assert over_kept.structured_content["left_out"] == countries.structured_content["left_out"]


async def test_a_level_on_a_root_without_the_type_column_refuses(store) -> None:
    root = await _register(store, _ref("1"), _frame({"dcid:AUS": [2022]}))

    result = await resolve_anchor(
        populations=[{"spec_id": _AID, "levels": {_PLACE_TYPE: "Country"}}], roots=[root]
    )

    assert _refused(result) == ("rolling_anchor_unresolved", "binding_column_unknown")
    assert result.structured_content["fix"]["columns"] == [_PLACE_TYPE]


async def _levelled_recipe(
    store: InMemoryMaterialisationStore, ref: FrameRef, frame: Frame
) -> dict[str, Any]:
    """The recipe of a rolling rank over the countries of the root `frame`."""
    root = await _register(store, ref, frame)
    mint = MintRecord(frame=ref, dataset=ref, options=_OPTIONS).model_dump(mode="json")
    window = {"window_end": LATEST}
    planned = {
        "op_spec": {"verb": "rank", "parameters": {"rolling": window}},
        "inputs": {
            "frame_ref": {
                "op_spec": {"verb": "filter", "parameters": {"predicates": [_COUNTRY_LEVEL]}},
                "inputs": {"frame_ref": {"root": 0}},
            }
        },
    }
    populations = anchor_populations(planned, [mint], root_columns=[frame.columns])
    assert populations == [{"spec_id": _AID, "bindings": {}, "levels": {_PLACE_TYPE: "Country"}}]

    anchor = (await resolve_anchor(populations=populations, roots=[root])).structured_content
    kept = await filter_verb(frame_ref=root, predicates=(_COUNTRY_LEVEL,))
    ranked = await rank(
        frame_ref=_pair(kept),
        rank_by=("observationAbout",),
        aggregation="sum",
        window_start=shift_period(anchor["period"], -1),
        window_end=anchor["period"],
        rolling={**window, "population": populations},
        anchor_key=anchor["anchor_key"],
    )
    recipe = await build_recipe(FrameRef(**_pair(ranked)), store=store)

    assert recipe["anchor"]["period"] == anchor["period"] == "2023"
    assert recipe["tree"]["op_spec"]["parameters"]["rolling"]["population"] == populations
    assert (
        anchor_populations(recipe["tree"], recipe["roots"], root_columns=[frame.columns])
        == populations
    )
    return recipe


async def test_a_levelled_recipe_is_one_recipe_however_the_data_spells_a_places_types(
    store,
) -> None:
    first = await _levelled_recipe(store, _ref("1"), _mixed())
    # The same data with Ghana's types spelled another way, still a country.
    respelled = await _levelled_recipe(
        store, _ref("2"), _mixed(ghana="AdministrativeArea1,Country")
    )

    assert first["normal_form_hash"] == respelled["normal_form_hash"]
    assert first["tree"]["frame"] != respelled["tree"]["frame"]


# Which roots a rolling period is taken over -----------------------------------

_GNI = "spec_" + "b" * 32
_DEBT = "spec_" + "c" * 32
_SINCE = {"kind": "ordered_threshold", "column": "time_period", "op": "gte"}
_PLANNED_SINCE = {**_SINCE, "rolling": {**LATEST, "offset": -1}}


def _planned_since(root: int) -> dict[str, Any]:
    """A rolling filter keeping the last two years of the root `root`, before any frame exists."""
    return {
        "op_spec": {
            "verb": "filter",
            "parameters": {"predicates": [_PLANNED_SINCE], "rolling": {}},
        },
        "inputs": {"frame_ref": {"root": root}},
    }


def _planned_join(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "op_spec": {"verb": "join", "parameters": {"right_frame_ref": {}}},
        "inputs": {"frame_ref": left, "right_frame_ref": right},
    }


async def _since(
    root: dict[str, str], *, populations: list[dict[str, Any]], anchor: Mapping[str, Any]
) -> dict[str, str]:
    """The rolling filter of `_planned_since` run on `root` against `anchor`."""
    predicate = {**_PLANNED_SINCE, "value": shift_period(anchor["period"], -1)}
    return _pair(
        await filter_verb(
            frame_ref=root,
            predicates=(predicate,),
            rolling={"population": populations},
            anchor_key=anchor["anchor_key"],
        )
    )


async def _roots(store: InMemoryMaterialisationStore, **newest: int) -> list[dict[str, str]]:
    """One root for each of `newest`'s datasets, each donor holding 2018 to that year."""
    specs = {"aid": _AID, "gni": _GNI, "debt": _DEBT}
    return [
        await _register(
            store,
            _ref("1", specs[name]),
            _frame(
                {donor: range(2018, last + 1) for donor in _DONORS},
                stat_var=f"dcid:{name.upper()}",
            ),
        )
        for name, last in newest.items()
    ]


def _mints(roots: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [{"dataset": {"spec_id": root["spec_id"]}} for root in roots]


async def test_a_dataset_joined_beside_a_rolling_step_does_not_move_its_period(store) -> None:
    # join(rolling filter on aid, gni): gni stops two years before aid.
    aid, gni = await _roots(store, aid=2023, gni=2021)
    planned = _planned_join(_planned_since(0), {"root": 1})
    columns = [list(_frame({}).columns)] * 2
    populations = anchor_populations(planned, _mints([aid, gni]), root_columns=columns)
    assert populations == [{"spec_id": _AID, "bindings": {}}]

    resolved = await resolve_anchor(populations=populations, roots=[aid])
    anchor = resolved.structured_content
    joined = _pair(
        await join(
            frame_ref=await _since(aid, populations=populations, anchor=anchor), right_frame_ref=gni
        )
    )
    recipe = await build_recipe(FrameRef(**joined), store=store)

    assert anchor["period"] == "2023"
    assert recipe["anchor"] == {"period": "2023", "rule": "half1", "left_out": []}


async def test_rolling_steps_one_step_joins_stand_on_one_period_over_both_datasets(store) -> None:
    aid, gni = await _roots(store, aid=2023, gni=2021)
    planned = _planned_join(_planned_since(0), _planned_since(1))
    columns = [list(_frame({}).columns)] * 2
    populations = anchor_populations(planned, _mints([aid, gni]), root_columns=columns)
    assert [population["spec_id"] for population in populations] == [_AID, _GNI]

    anchor = (await resolve_anchor(populations=populations, roots=[aid, gni])).structured_content
    joined = _pair(
        await join(
            frame_ref=await _since(aid, populations=populations, anchor=anchor),
            right_frame_ref=await _since(gni, populations=populations, anchor=anchor),
        )
    )
    recipe = await build_recipe(FrameRef(**joined), store=store)

    assert anchor["period"] == "2021"
    assert recipe["anchor"] is not None
    assert recipe["anchor"]["period"] == "2021"


async def test_a_frame_whose_populations_reach_past_its_tree_gives_its_recipe_to_replay(
    store,
) -> None:
    # S1 on aid, S2 on gni and S3 on debt stand on one period, since one join
    # reads S1 and S2 and another S2 and S3. The first join reads aid and gni
    # alone, so its frame has no anchor of its own: the recipe comes back on the
    # error, and its own tree gives the populations a replay takes.
    aid, gni, debt = await _roots(store, aid=2023, gni=2023, debt=2021)
    group = canonical_populations([{"spec_id": _AID}, {"spec_id": _GNI}, {"spec_id": _DEBT}])
    anchor = (await resolve_anchor(populations=group, roots=[aid, gni, debt])).structured_content
    first = _pair(
        await join(
            frame_ref=await _since(aid, populations=group, anchor=anchor),
            right_frame_ref=await _since(gni, populations=group, anchor=anchor),
        )
    )

    with pytest.raises(RecipeAnchorError) as raised:
        await build_recipe(FrameRef(**first), store=store)

    assert raised.value.reason == "populations_outside_reach"
    recipe = raised.value.recipe
    assert recipe is not None
    assert recipe["anchor"] is None
    columns = [list(_frame({}).columns)] * len(recipe["roots"])
    own = anchor_populations(recipe["tree"], recipe["roots"], root_columns=columns)
    assert own == [{"spec_id": _AID, "bindings": {}}, {"spec_id": _GNI, "bindings": {}}]
