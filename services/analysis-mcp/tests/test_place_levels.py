"""Place levels: `place_level` keeps one level of place, and `rank`, `aggregate`,
`share_of` and `quotient_of` refuse `place_levels_mixed` over a frame whose several places
share none.

The frames carry `observationAbout_type` as `get_data(place_types=true)` registers it: every
Data Commons `typeOf` sorted and comma-joined, `Place` for Earth, `ProvisionalNode` for a
World Bank aggregate, and `untyped` for a place the graph holds no type for.
"""

from __future__ import annotations

import secrets

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.engine.common import PlaceLevelsMixedError
from analysis_mcp.engine.filter import (
    FilterPlaceLevelUnknownError,
    FilterPredicateTypeMismatchError,
    run_filter,
)
from analysis_mcp.engine.rank import run_rank
from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.filter_tool import filter_verb
from analysis_mcp.identity import op_spec_hash_for
from analysis_mcp.op_spec import AggregateOpSpec, PlaceLevelPredicate
from analysis_mcp.quotient_of_tool import quotient_of
from analysis_mcp.rank_tool import rank
from analysis_mcp.share_of_tool import share_of
from analysis_mcp.trace import AppliedPolicy, ExecutionRecord, Trace
from common.materialisation import FrameRef, InMemoryMaterialisationStore, set_store
from dfo_frame_grammar import Frame
from pydantic import ValidationError

_GDP = "dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"

# GDP in 2023 at every level of place, as apis serves a mint that names no place.
_EVERY_PLACE = {
    "Earth": ("Place", 105.0),
    "country/HIC": ("ProvisionalNode", 64.0),
    "country/USA": ("Country", 27.0),
    "country/CHN": ("Country", 18.0),
    "country/FIN": ("Country", 0.3),
    "nuts/FI2": ("AdministrativeArea1,Country,EurostatNUTS1,Region", 0.002),
    "country/HKG": ("AdministrativeArea1,City,Country,SpecialAdministrativeRegion", 0.4),
    "country/GUM": ("AdministrativeArea1,Country", 0.007),
    "geoId/06": ("AdministrativeArea1,State", 3.9),
    "undata-geo/G00158000": ("untyped", 2.0),
}


def _frame(
    places: dict[str, tuple[str, float]] = _EVERY_PLACE, years: tuple[str, ...] = ("2023",)
) -> Frame:
    return Frame.build(
        role_keys=["observationAbout", "observationAbout_type"],
        rows=[
            {
                "stat_var": _GDP,
                "observationAbout": place,
                "observationAbout_type": types,
                "time_period": year,
                "value": value,
                "unit": "USDollar",
            }
            for place, (types, value) in places.items()
            for year in years
        ],
    )


# Earth beside two countries the graph also types `Place`: one shared type, no shared level.
_WORLD_AND_COUNTRIES = {
    "Earth": ("Place", 105.0),
    "country/USA": ("Country,Place", 27.0),
    "country/IND": ("Country,Place", 3.6),
}
_COUNTRIES_TYPED_PLACE = {
    place: row for place, row in _WORLD_AND_COUNTRIES.items() if place != "Earth"
}

# World Bank income groups: "Low & middle income" contains "Low income".
_INCOME_GROUPS = {
    "country/LMY": ("ProvisionalNode", 36.0),
    "country/LIC": ("ProvisionalNode", 0.6),
}

# UN regions: Data Commons types Sub-Saharan Africa and the Eastern Africa it contains alike.
_UN_REGIONS = {
    "SubSaharanAfrica": ("UNGeoRegion", 2.0),
    "EasternAfrica": ("UNGeoRegion", 0.5),
}


def _countries_only() -> dict[str, tuple[str, float]]:
    """What apis serves when the full list of places fails: countries alone."""
    return {place: row for place, row in _EVERY_PLACE.items() if row[0] == "Country"}


def _kept(frame: Frame, level: str) -> list[str]:
    result = run_filter(
        frame, predicates=(PlaceLevelPredicate(column="observationAbout_type", level=level),)
    )
    return sorted(result.frame.table.column("observationAbout").to_pylist())


# --- place_level ------------------------------------------------------------------------


def test_a_country_is_typed_country_and_no_eurostat_nuts_level() -> None:
    # Finland's figures include Åland, a NUTS region the graph also types Country;
    # Guam's and Hong Kong's are left out of the United States' and China's.
    assert _kept(_frame(), "Country") == [
        "country/CHN",
        "country/FIN",
        "country/GUM",
        "country/HKG",
        "country/USA",
    ]


def test_place_level_keeps_a_place_at_each_of_its_other_types() -> None:
    assert _kept(_frame(), "AdministrativeArea1") == [
        "country/GUM",
        "country/HKG",
        "geoId/06",
        "nuts/FI2",
    ]
    assert _kept(_frame(), "EurostatNUTS1") == ["nuts/FI2"]
    assert _kept(_frame(), "City") == ["country/HKG"]
    assert _kept(_frame(), "Continent") == []


def test_place_level_reads_a_role_column_and_refuses_the_value() -> None:
    with pytest.raises(FilterPredicateTypeMismatchError):
        run_filter(_frame(), predicates=(PlaceLevelPredicate(column="value", level="Country"),))
    with pytest.raises(FilterPredicateTypeMismatchError):
        run_filter(_frame(), predicates=(PlaceLevelPredicate(column="unit", level="Country"),))


def test_a_place_level_names_one_level_of_the_closed_set() -> None:
    # "Area1" is part of a type's name, and Place and UNGeoRegion are types that name no level.
    for level in ("Area1", "Place", "UNGeoRegion", "Country,City"):
        with pytest.raises(ValidationError):
            PlaceLevelPredicate(column="observationAbout_type", level=level)


def test_a_strict_place_level_refuses_an_untyped_entity() -> None:
    with pytest.raises(
        FilterPlaceLevelUnknownError, match="cannot classify every entity"
    ) as raised:
        run_filter(
            _frame(),
            predicates=(
                PlaceLevelPredicate(column="observationAbout_type", level="Country", strict=True),
            ),
        )

    assert raised.value.column == "observationAbout_type"


def test_a_strict_flow_filter_does_not_read_the_other_partys_type() -> None:
    frame = Frame.build(
        role_keys=[
            "financingProvider",
            "financingProvider_type",
            "financingRecipient",
            "financingRecipient_type",
        ],
        rows=[
            {
                "stat_var": _GDP,
                "financingProvider": "org/Bank",
                "financingProvider_type": "untyped",
                "financingRecipient": "country/KEN",
                "financingRecipient_type": "Country,Place",
                "time_period": "2023",
                "value": 1.0,
                "unit": "USDollar",
            }
        ],
    )

    result = run_filter(
        frame,
        predicates=(
            PlaceLevelPredicate(column="financingRecipient_type", level="Country", strict=True),
        ),
    )

    assert result.kept_row_count == 1


# --- one level ------------------------------------------------------------------------


def test_a_rank_over_every_level_of_place_is_refused_naming_the_types() -> None:
    with pytest.raises(PlaceLevelsMixedError) as raised:
        run_rank(
            _frame(),
            rank_by=("observationAbout",),
            aggregation=None,
            window_start=None,
            window_end=None,
            ascending=False,
        )

    assert raised.value.column == "observationAbout_type"
    assert "Place" in raised.value.types
    assert "ProvisionalNode" in raised.value.types


@pytest.mark.parametrize("places", [_WORLD_AND_COUNTRIES, _INCOME_GROUPS, _UN_REGIONS])
def test_a_rank_over_places_sharing_a_type_that_names_no_level_is_refused(
    places: dict[str, tuple[str, float]],
) -> None:
    with pytest.raises(PlaceLevelsMixedError) as raised:
        run_rank(
            _frame(places),
            rank_by=("observationAbout",),
            aggregation=None,
            window_start=None,
            window_end=None,
            ascending=False,
        )

    assert raised.value.types == tuple(sorted({types for types, _ in places.values()}))
    assert "share no level" in str(raised.value)


@pytest.mark.parametrize(
    "places",
    [
        _countries_only(),
        _COUNTRIES_TYPED_PLACE,
        {"Earth": ("Place", 105.0)},
        {"country/LMY": ("ProvisionalNode", 36.0)},
        {"undata-geo/G00158000": ("untyped", 2.0)},
    ],
)
def test_a_rank_over_one_level_runs_with_no_filter(places: dict[str, tuple[str, float]]) -> None:
    # The countries apis falls back to share Country, and one place alone
    # sits at one level whatever its type.
    result = run_rank(
        _frame(places),
        rank_by=("observationAbout",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
    )

    assert len(result.entity_ranks) == len(places)


# --- the tools --------------------------------------------------------------------------


@pytest.fixture
def store():
    held = InMemoryMaterialisationStore()
    set_store(held)
    yield held
    set_store(None)


def _trace() -> Trace:
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


async def _held(store: InMemoryMaterialisationStore, frame: Frame) -> dict[str, str]:
    digest = secrets.token_hex(16)
    ref = FrameRef(spec_id=f"spec_{digest}", content_version=f"cv_{digest}")
    await store.put(ref, frame=frame, trace=_trace(), writer="test")
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _refused_mixed(result) -> dict[str, object]:
    assert result.is_error is True
    content = result.structured_content
    assert content["code"] == "place_levels_mixed"
    assert content["retry_mode"] == "correct_and_retry"
    assert content["fix"]["column"] == "observationAbout_type"
    return content


async def test_the_tools_refuse_a_rank_a_total_a_share_and_a_quotient_over_every_level(
    store,
) -> None:
    every_place = await _held(store, _frame())
    world = await _held(store, _frame({"Earth": ("Place", 105.0)}))

    ranked = await rank(frame_ref=every_place, rank_by=("observationAbout",))
    totalled = await aggregate(frame_ref=every_place, aggregation="sum", group_by=("time_period",))
    shared = await share_of(
        frame_ref=every_place,
        denominator_frame_ref=world,
        numerator_entity={},
        denominator_entity={},
    )
    divided = await quotient_of(frame_ref=every_place, denominator_frame_ref=world)

    content = _refused_mixed(ranked)
    assert "place_level" in content["message"]
    _refused_mixed(totalled)
    _refused_mixed(shared)
    _refused_mixed(divided)


@pytest.mark.parametrize("places", [_WORLD_AND_COUNTRIES, _INCOME_GROUPS, _UN_REGIONS])
async def test_the_tools_refuse_places_sharing_a_type_that_names_no_level(
    store, places: dict[str, tuple[str, float]]
) -> None:
    held = await _held(store, _frame(places))
    world = await _held(store, _frame({"Earth": ("Place", 105.0)}))

    ranked = await rank(frame_ref=held, rank_by=("observationAbout",))
    totalled = await aggregate(frame_ref=held, aggregation="sum", group_by=("time_period",))
    shared = await share_of(
        frame_ref=held,
        denominator_frame_ref=world,
        numerator_entity={},
        denominator_entity={},
    )
    divided = await quotient_of(frame_ref=held, denominator_frame_ref=world)

    content = _refused_mixed(ranked)
    assert "share no level a filter can keep" in content["message"]
    assert "EurostatNUTS1" in content["message"]
    _refused_mixed(totalled)
    _refused_mixed(shared)
    _refused_mixed(divided)


@pytest.mark.parametrize(
    "places",
    [
        _COUNTRIES_TYPED_PLACE,
        {"Earth": ("Place", 105.0)},
        {"country/LMY": ("ProvisionalNode", 36.0)},
    ],
)
async def test_the_tools_run_over_one_place_or_countries_also_typed_place(
    store, places: dict[str, tuple[str, float]]
) -> None:
    held = await _held(store, _frame(places))
    world = await _held(store, _frame({"Earth": ("Place", 105.0)}))

    ranked = await rank(frame_ref=held, rank_by=("observationAbout",))
    totalled = await aggregate(frame_ref=held, aggregation="sum", group_by=("time_period",))
    shared = await share_of(
        frame_ref=held,
        denominator_frame_ref=world,
        numerator_entity={},
        denominator_entity={},
    )
    divided = await quotient_of(frame_ref=held, denominator_frame_ref=world)

    assert ranked.is_error is False, ranked.structured_content
    assert totalled.is_error is False, totalled.structured_content
    assert shared.is_error is False, shared.structured_content
    assert divided.is_error is False, divided.structured_content


async def test_a_total_per_place_or_per_type_keeps_the_levels_apart(store) -> None:
    every_place = await _held(store, _frame(years=("2022", "2023")))

    per_place = await aggregate(
        frame_ref=every_place, aggregation="mean", group_by=("observationAbout",)
    )
    per_type = await aggregate(
        frame_ref=every_place,
        aggregation="sum",
        group_by=("observationAbout_type", "time_period"),
    )

    assert per_place.is_error is False, per_place.structured_content
    assert per_type.is_error is False, per_type.structured_content


async def test_filtering_one_level_lets_the_rank_and_the_total_run(store) -> None:
    every_place = await _held(store, _frame())

    kept = await filter_verb(
        frame_ref=every_place,
        predicates=(PlaceLevelPredicate(column="observationAbout_type", level="Country"),),
    )
    assert kept.is_error is False, kept.structured_content
    countries = {
        "spec_id": kept.structured_content["spec_id"],
        "content_version": kept.structured_content["content_version"],
    }

    ranked = await rank(frame_ref=countries, rank_by=("observationAbout",), limit=2)
    totalled = await aggregate(frame_ref=countries, aggregation="sum", group_by=("time_period",))

    assert ranked.is_error is False, ranked.structured_content
    assert totalled.is_error is False, totalled.structured_content


async def test_a_total_over_a_country_and_its_nuts_region_is_refused(store) -> None:
    # Both carry Country, and Åland's figures sit inside Finland's.
    finland = await _held(
        store,
        _frame({place: _EVERY_PLACE[place] for place in ("country/FIN", "nuts/FI2")}),
    )

    totalled = await aggregate(frame_ref=finland, aggregation="sum", group_by=("time_period",))

    _refused_mixed(totalled)


async def test_a_share_keyed_by_place_over_every_level_is_refused(store) -> None:
    # The plan refuses a share whose side mixes levels, whether the side sums
    # across its places or keys the shares by them, and so does the tool.
    every_place = await _held(store, _frame())
    countries = await _held(store, _frame(_countries_only()))

    shared = await share_of(
        frame_ref=every_place,
        denominator_frame_ref=countries,
        numerator_by=["observationAbout"],
    )

    _refused_mixed(shared)


async def test_a_quotient_keyed_by_place_over_every_level_is_refused(store) -> None:
    every_place = await _held(store, _frame())
    countries = await _held(store, _frame(_countries_only()))

    divided = await quotient_of(
        frame_ref=every_place,
        denominator_frame_ref=countries,
        numerator_by=["observationAbout"],
        denominator_by={"observationAbout": "observationAbout"},
    )

    _refused_mixed(divided)
