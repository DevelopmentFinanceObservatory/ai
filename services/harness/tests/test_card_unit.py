"""Tests for `harness.card_unit`: which cards can change unit, and how their frame and title move."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from common.live_apis import LiveApisError
from data_mcp.get_data_tool import _SUM_STAT_VARS_GROUPBY, Breakdown
from harness import card_unit, place_options
from harness.card_unit import frame_arguments, retitled, unit_card
from harness.provenance import CardLineage, ProvenanceInput, lineage_from_json, lineage_json
from harness.studio import _read_card_change
from harness.wire import (
    CardChanges,
    PlaceOption,
    PlacesChange,
    PlacesOffer,
    UnitChoice,
    UnitCurrency,
    UnitOffer,
    UnitPrices,
    WindowOffer,
)
from pydantic import ValidationError

_DEFAULT = {"format": "parquet", "bundle": False, "layout": "frame"}
_SPEC = "spec_" + "b" * 32
_CV = "cv_" + "2" * 32


@pytest.mark.parametrize(
    ("rendering", "args"),
    [
        (_DEFAULT, {}),
        (_DEFAULT | {"groupby": _SUM_STAT_VARS_GROUPBY}, {"stat_vars": "sum"}),
        (
            _DEFAULT | {"groupby": Breakdown(code_system="crsPurpose", level=0).groupby()},
            {"breakdown": {"code_system": "crsPurpose", "level": 0}},
        ),
        (
            _DEFAULT
            | {
                "columns": ["party_kind", "party_class", "party_whole"],
                "without_party_totals": True,
            },
            {"without_party_totals": True},
        ),
        (
            _DEFAULT | {"columns": ["observationAbout_type", "party_kind"]},
            {"place_types": True},
        ),
        (_DEFAULT | {"columns": ["party_kind"]}, None),
        (_DEFAULT | {"groupby": {"by": ["time_period", "stat_var"], "op": "sum"}}, None),
        (_DEFAULT | {"format": "csv"}, None),
        (_DEFAULT | {"unheard_of": 1}, None),
        (_DEFAULT | {"metadata": "incomplete"}, {}),
    ],
    ids=[
        "default",
        "sum",
        "breakdown",
        "without-party-totals",
        "place-types",
        "kinds-alone",
        "unknown-groupby",
        "other-format",
        "unknown-option",
        "incomplete-metadata",
    ],
)
def test_a_rendering_maps_back_to_the_get_data_arguments_that_mint_it(
    rendering: dict[str, Any], args: dict[str, Any] | None
) -> None:
    assert frame_arguments(rendering) == args


def _component(spec_id: str = _SPEC) -> dict[str, Any]:
    return {"provenance": {"inputs": [{"spec_id": spec_id, "content_version": _CV}]}}


def _lineage(frame: tuple[str, str], members: int = 1) -> CardLineage:
    manifest = {
        "definition": {"spec_id": _SPEC},
        "content_version": _CV,
        "artifacts": [{"rendering": {"statistics": "series"}}, {"rendering": _DEFAULT}],
    }
    return CardLineage(
        component_id="cmp_1",
        frame=frame,
        lineage=tuple(ProvenanceInput(_SPEC, _CV, manifest) for _ in range(members)),
    )


@pytest.mark.parametrize(
    ("lineage", "spec_id", "readable"),
    [
        (_lineage((_SPEC, _CV)), _SPEC, True),
        (_lineage(("spec_" + "d" * 32, "ocv_1")), _SPEC, False),
        (_lineage((_SPEC, _CV), members=2), _SPEC, False),
        (_lineage((_SPEC, _CV)), "spec_" + "e" * 32, False),
        (None, _SPEC, False),
    ],
    ids=["retrieval-frame", "derived-frame", "several-retrievals", "other-input", "no-lineage"],
)
def test_only_a_card_on_one_retrievals_own_frame_can_change_unit(
    lineage: CardLineage | None, spec_id: str, readable: bool
) -> None:
    card = unit_card(_component(spec_id), lineage)
    assert (card is not None) is readable
    if card is not None:
        assert (card.spec_id, card.args) == (_SPEC, {})


_OFFER = UnitOffer(
    currencies=[
        UnitCurrency(
            code="USD",
            name="US dollars",
            prices=[UnitPrices.CURRENT, UnitPrices.CONSTANT],
            base_year_min=2000,
            base_year_max=2023,
        ),
        UnitCurrency(code="EUR", name="euros", prices=[UnitPrices.CURRENT]),
    ],
    presets=["USD"],
    start=UnitChoice(currency="USD", prices=UnitPrices.CURRENT),
)
_CURRENT_USD = UnitChoice(currency="USD", prices=UnitPrices.CURRENT)
_CONSTANT_2022_USD = UnitChoice(currency="USD", prices=UnitPrices.CONSTANT, base_year=2022)
_CONSTANT_2023_USD = UnitChoice(currency="USD", prices=UnitPrices.CONSTANT, base_year=2023)
_CURRENT_EUR = UnitChoice(currency="EUR", prices=UnitPrices.CURRENT)


@pytest.mark.parametrize(
    ("title", "old", "new", "expected"),
    [
        (
            "Aid to Kenya, current US dollars",
            _CURRENT_USD,
            _CONSTANT_2023_USD,
            "Aid to Kenya, constant 2023 US dollars",
        ),
        (
            "Aid to Kenya (constant 2022 prices)",
            _CONSTANT_2022_USD,
            _CONSTANT_2023_USD,
            "Aid to Kenya (constant 2023 prices)",
        ),
        (
            "Aid to Kenya in 2022 prices",
            _CONSTANT_2022_USD,
            _CURRENT_USD,
            "Aid to Kenya in current prices",
        ),
        (
            "Aid to Kenya at current prices, USD",
            _CURRENT_USD,
            _CONSTANT_2023_USD,
            "Aid to Kenya at constant 2023 prices, USD",
        ),
        ("Aid to Kenya, USD billions", _CURRENT_USD, _CURRENT_EUR, "Aid to Kenya, EUR billions"),
        ("Aid to Kenya in 2022", _CONSTANT_2022_USD, _CONSTANT_2023_USD, "Aid to Kenya in 2022"),
        ("Aid to Kenya", _CURRENT_USD, _CONSTANT_2023_USD, "Aid to Kenya"),
        ("Aid to Kenya in euros", _CURRENT_EUR, _CURRENT_USD, "Aid to Kenya in US dollars"),
        ("Aid to Kenya in US dollars", _CURRENT_USD, _CURRENT_EUR, "Aid to Kenya in euros"),
        ("Euros to Kenya", _CURRENT_EUR, _CURRENT_USD, "US dollars to Kenya"),
        (
            "Aid to Kenya in current USD",
            _CURRENT_USD,
            _CONSTANT_2023_USD,
            "Aid to Kenya in constant 2023 USD",
        ),
        (
            "Aid to Kenya (constant US dollars)",
            _CONSTANT_2023_USD,
            _CURRENT_EUR,
            "Aid to Kenya (current euros)",
        ),
        (
            "Constant 2023 US dollars to Kenya",
            _CONSTANT_2023_USD,
            _CURRENT_EUR,
            "Current euros to Kenya",
        ),
        (
            "Aid to Kenya, constant 2022 USD",
            _CONSTANT_2022_USD,
            _CURRENT_EUR,
            "Aid to Kenya, current EUR",
        ),
        (
            "ODA to Kenya (constant 2022, USD millions)",
            _CONSTANT_2022_USD,
            _CONSTANT_2023_USD,
            "ODA to Kenya (constant 2023, USD millions)",
        ),
        ("ODA to Kenya, constant 2022", _CONSTANT_2022_USD, _CURRENT_USD, "ODA to Kenya, current"),
        (
            "ODA to Kenya in current prices, 2010 prices",
            _CURRENT_USD,
            _CONSTANT_2023_USD,
            "ODA to Kenya in constant 2023 prices, 2010 prices",
        ),
        (
            "ODA to Kenya in current prices, 2010 prices",
            _CONSTANT_2022_USD,
            _CONSTANT_2023_USD,
            "ODA to Kenya in current prices, 2010 prices",
        ),
    ],
    ids=[
        "whole-unit",
        "constant-prices",
        "year-prices-to-current",
        "current-prices",
        "currency-code",
        "a-year-alone-stands",
        "no-unit-words",
        "euros-to-us-dollars",
        "us-dollars-to-euros",
        "name-opens-the-title",
        "current-currency-code",
        "basis-without-year-and-currency",
        "unit-words-open-the-title",
        "basis-and-code-to-another-currency",
        "basis-alone-before-a-currency",
        "basis-alone-to-current",
        "another-year-before-prices-stands",
        "another-basis-stands",
    ],
)
def test_a_title_names_the_new_unit_where_it_named_the_old(
    title: str, old: UnitChoice, new: UnitChoice, expected: str
) -> None:
    assert retitled(title, old=old, new=new, offer=_OFFER) == expected


_BASE = "spec_" + "a" * 32
_BASE_IDENTITY = {
    "stat_var": "dcid:Aid",
    "slots": [{"property": "b", "value": 1}, {"property": "a", "value": 2}],
    "entities": {"destination": ["country/KEN"], "origin": ["country/FRA"]},
    "requested_window": {"kind": "range", "start_year": 2015, "end_year": 2023},
}


@pytest.mark.parametrize(
    ("served", "spec_id", "scope"),
    [
        (
            {
                **_BASE_IDENTITY,
                "slots": [{"property": "a", "value": 2}, {"property": "b", "value": 1}],
                "entities": {"destination": ["country/UGA"], "origin": ["country/FRA"]},
                "requested_window": {"kind": "range", "start_year": 2010, "end_year": 2024},
            },
            _BASE,
            {
                "entities": {"destination": ["dcid:country/UGA"], "origin": ["dcid:country/FRA"]},
                "window": {"kind": "range", "start_year": 2010, "end_year": 2024},
            },
        ),
        ({**_BASE_IDENTITY, "stat_var": "dcid:Other"}, _SPEC, None),
    ],
    ids=["scoped-retrieval-of-the-block", "another-indicator"],
)
def test_a_card_on_a_scoped_retrieval_reads_its_block_with_the_scope(
    served: dict[str, Any], spec_id: str, scope: dict[str, Any] | None
) -> None:
    """A retrieval served under its own spec reads back as the Search block it
    scoped, with the places and window it changed. One that differs from
    every block elsewhere reads its own spec."""
    lineage = _lineage((_SPEC, _CV))
    (member,) = lineage.lineage
    member.manifest["definition"]["identity"] = served
    blocks = {_BASE: {"definition": {"spec_id": _BASE, "identity": _BASE_IDENTITY}}}

    card = unit_card(_component(), lineage, blocks)

    assert card is not None
    assert (card.spec_id, card.scope) == (spec_id, scope)


_OTHER_BASE = "spec_" + "c" * 32


@pytest.mark.parametrize("reversed_blocks", [False, True], ids=["in-order", "reversed"])
@pytest.mark.parametrize(
    ("requested", "spec_id"),
    [(_OTHER_BASE, _OTHER_BASE), (_BASE, _BASE), (None, _SPEC)],
    ids=["asked-for-the-second", "asked-for-the-first", "unrecorded-and-ambiguous"],
)
def test_a_scoped_retrieval_reads_the_spec_it_asked_for(
    requested: str | None, spec_id: str, reversed_blocks: bool
) -> None:
    """Two blocks match the served identity outside its places. The card reads
    the one its retrieval asked for, in whatever order the conversation keeps
    them; a retrieval that recorded none reads its own spec and offers no
    change, since either block could be the one."""
    lineage = _lineage((_SPEC, _CV))
    (member,) = lineage.lineage
    member.manifest["definition"]["identity"] = {
        **_BASE_IDENTITY,
        "entities": {"destination": ["country/TGO"], "origin": ["country/FRA"]},
    }
    lineage = CardLineage(
        component_id="cmp_1",
        frame=lineage.frame,
        lineage=(ProvenanceInput(_SPEC, _CV, member.manifest, requested),),
    )
    blocks = {
        _BASE: {"definition": {"spec_id": _BASE, "identity": _BASE_IDENTITY}},
        _OTHER_BASE: {
            "definition": {
                "spec_id": _OTHER_BASE,
                "identity": {**_BASE_IDENTITY, "entities": {"destination": ["country/UGA"]}},
            }
        },
    }
    if reversed_blocks:
        blocks = dict(reversed(blocks.items()))

    card = unit_card(_component(), lineage, blocks)

    assert card is not None
    assert card.spec_id == spec_id
    assert (card.scope is None) is (requested is None)


def test_a_lineage_keeps_the_spec_its_retrieval_asked_for() -> None:
    lineage = (ProvenanceInput(_SPEC, _CV, {"definition": {"spec_id": _SPEC}}, _BASE),)

    assert lineage_from_json(json.loads(json.dumps(lineage_json(lineage)))) == lineage


def test_a_card_offers_the_places_it_draws_of_its_frame() -> None:
    """The frame holds Kenya and Uganda and the chart draws Kenya: Kenya is
    the place shown, and adding Uganda is a change."""
    block = _france_kenya_block()
    scoped = card_unit.UnitCard(
        spec_id=block["spec_id"],
        args={},
        scope={"entities": {"destination": ["dcid:country/KEN", "dcid:country/UGA"]}},
        labels={"country/KEN": "Kenya", "country/UGA": "Uganda"},
    )
    intent = {"type": "line_chart", "select": {"destination": "dcid:country/KEN"}}
    card = card_unit.encoded(scoped, block, intent=intent)

    offer = card_unit.card_changes(
        card_unit.UnitCard.from_json(card.as_json()) or card,
        block,
        fact_summary=_FETCHED_2016_2024,
        places=True,
    )

    assert offer is not None
    (destination,) = [p for p in offer.places or () if p.role == "destination"]
    assert [o.id for o in destination.chosen] == ["country/KEN"]
    both = PlacesChange(kind="places", role="destination", entities=["country/KEN", "country/UGA"])
    assert card_unit._places_refusal(offer, both) is None


class _Door:
    """Search's places door: lists `listed` places, fails with `fail`, and
    with `stall` waits past the deadline."""

    def __init__(self, listed: int = 2, *, fail: bool = False, stall: bool = False) -> None:
        self.listed, self.fail, self.stall = listed, fail, stall

    async def __call__(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        del identity, role
        if self.stall:
            await asyncio.sleep(60)
        if self.fail:
            raise LiveApisError(status=503, code=None, retryable=True)
        return tuple((f"country/C{n}", f"Place {n}") for n in range(self.listed)), False


@pytest.mark.parametrize(
    "door", [_Door(fail=True), _Door(stall=True)], ids=["door-fails", "door-misses-deadline"]
)
async def test_a_place_read_that_fails_or_misses_its_deadline_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, door: _Door
) -> None:
    """The read is refused, and the warning names the conversation and card."""
    monkeypatch.setattr(place_options, "PLACE_OPTIONS_DEADLINE_S", 0.05)

    with caplog.at_level(logging.WARNING), pytest.raises(place_options.PlacesUnavailableError):
        await place_options.read_role_places(
            door,
            identity=_BASE_IDENTITY,
            role="destination",
            conversation_id="conv_1",
            component_id="cmp_1",
        )

    (warned,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warned.args[:2] == ("conv_1", "cmp_1")


async def test_a_place_read_keeps_the_cap_and_says_it_cut_the_list() -> None:
    read = await place_options.read_role_places(
        _Door(place_options.PLACE_OPTIONS_CAP + 1), identity=_BASE_IDENTITY, role="destination"
    )

    assert (len(read.places), read.truncated) == (place_options.PLACE_OPTIONS_CAP, True)


class _Client:
    """A `LiveApisClient` stand-in: Search's options for any identity, or the
    failure `fails` names, and the doors its capabilities document lists."""

    def __init__(self, doors: frozenset[str] = frozenset({"options"})) -> None:
        self.doors = doors
        self.fails: BaseException | None = None
        self.asked = 0
        # Set, a read waits for it before it answers.
        self.held: asyncio.Event | None = None

    async def read_entity_options(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        del identity, role
        self.asked += 1
        if self.held is not None:
            await self.held.wait()
        if self.fails is not None:
            raise self.fails
        return (("country/UGA", "Uganda"),), False

    async def read_search_doors(self) -> frozenset[str]:
        return self.doors


_NOT_FOUND = LiveApisError(status=404, code="not_found", retryable=False)


def _reader(client: _Client, now: list[float]) -> place_options.ApisPlaceOptions:
    return place_options.ApisPlaceOptions(
        base_url="http://apis.test",
        client=client,  # ty: ignore[invalid-argument-type] -- the two reads the reader makes
        clock=lambda: now[0],
    )


async def _read(reader: place_options.ApisPlaceOptions, role: str = "destination") -> object:
    return await place_options.read_role_places(reader, identity=_BASE_IDENTITY, role=role)


async def test_an_apis_serving_no_options_door_stops_the_offers_until_the_ttl_runs_out() -> None:
    """The first 404 reads the capabilities document. With no door listed the
    offers stop and reads are refused without a call, until the TTL runs out
    and the next read finds the door back."""
    now = [0.0]
    client = _Client(doors=frozenset({"stated_scope"}))
    client.fails = _NOT_FOUND
    reader = _reader(client, now)

    with pytest.raises(place_options.PlacesNotServedError):
        await _read(reader)
    offered_after_404 = place_options.places_offered(reader)
    with pytest.raises(place_options.PlacesNotServedError):
        await _read(reader, "origin")
    asked_while_off = client.asked
    now[0] = place_options.PLACE_OPTIONS_TTL_S + 1
    client.fails, client.doors = None, frozenset({"options"})
    offered_after_ttl = place_options.places_offered(reader)
    read = await _read(reader)

    assert (offered_after_404, asked_while_off, offered_after_ttl) == (False, 1, True)
    assert read == place_options.RolePlaces(places=(("country/UGA", "Uganda"),), truncated=False)


@pytest.mark.parametrize(
    "failure",
    [OSError("connection refused"), LiveApisError(status=503, code=None, retryable=True)],
    ids=["unreachable", "unavailable"],
)
async def test_an_apis_unreachable_at_first_costs_nothing_once_it_is_back(
    failure: BaseException,
) -> None:
    """A read that cannot reach apis is refused to retry and leaves the offers
    on, and the next read once apis is back answers."""
    now = [0.0]
    client = _Client()
    client.fails = failure
    reader = _reader(client, now)

    with pytest.raises(place_options.PlacesUnavailableError) as refused:
        await _read(reader)
    client.fails = None
    read = await _read(reader)

    assert not isinstance(refused.value, place_options.PlacesNotServedError)
    assert place_options.places_offered(reader)
    assert read.places == (("country/UGA", "Uganda"),)


async def test_a_404_with_the_options_door_listed_is_refused_to_retry() -> None:
    client = _Client()
    client.fails = _NOT_FOUND
    reader = _reader(client, [0.0])

    with pytest.raises(place_options.PlacesUnavailableError) as refused:
        await _read(reader)

    assert not isinstance(refused.value, place_options.PlacesNotServedError)
    assert place_options.places_offered(reader)


def test_a_roles_places_are_read_once_a_minute() -> None:
    """A picker's read and the change applied from it make one call."""
    now = [0.0]
    client = _Client()
    reader = _reader(client, now)

    async def read(role: str = "destination") -> object:
        return await reader(identity=_BASE_IDENTITY, role=role)

    async def reads() -> None:
        await read()
        now[0] = place_options.PLACE_OPTIONS_TTL_S - 1
        await read()
        await read("origin")
        now[0] = place_options.PLACE_OPTIONS_TTL_S + 1
        await read()

    asyncio.run(reads())

    assert client.asked == 3


@pytest.mark.parametrize("fails", [False, True], ids=["answers", "fails"])
async def test_concurrent_reads_of_one_role_make_one_call(fails: bool) -> None:
    """Twenty reads of one role while its lookup runs wait on that lookup. A
    failed lookup raises in every one of them and is kept for none, so the
    next read asks again."""
    client = _Client()
    client.held = asyncio.Event()
    client.fails = _NOT_FOUND if fails else None
    client.doors = frozenset({"options"})
    reader = _reader(client, [0.0])

    reads = [
        asyncio.create_task(reader(identity=_BASE_IDENTITY, role="destination")) for _ in range(20)
    ]
    await asyncio.sleep(0)
    client.held.set()
    answers = await asyncio.gather(*reads, return_exceptions=True)

    assert client.asked == 1
    if fails:
        assert all(isinstance(a, LiveApisError) for a in answers)
        client.fails = None
        await reader(identity=_BASE_IDENTITY, role="destination")
        assert client.asked == 2
    else:
        assert answers == [((("country/UGA", "Uganda"),), False)] * 20


@pytest.mark.parametrize(
    ("noun", "nouns"),
    [
        (
            {"singular": "debtor", "plural": "debtors"},
            {"singular": "debtor", "plural": "debtors"},
        ),
        ({"singular": "debtor", "plural": ""}, {"singular": "debtor"}),
        ("debtor", None),
        (None, None),
    ],
    ids=["both", "one-form", "not-a-mapping", "none"],
)
def test_a_roles_noun_is_the_one_its_coverage_states(noun: object, nouns: object) -> None:
    """The forms a coverage entry's noun states, under its axis and its own
    label; a role whose entry states none has no noun."""
    entry: dict[str, object] = {"label": "debtorCountry", "count": 3, "axis": "destination"}
    if noun is not None:
        entry["noun"] = noun
    found = card_unit.role_nouns({"coverage": {"dimensions": [entry, {"label": "years"}]}})

    assert found == ({"debtorCountry": nouns, "destination": nouns} if nouns else None)
    card = card_unit.UnitCard(spec_id="spec_x", args={}, nouns=found)
    assert card_unit.UnitCard.from_json(card.as_json()) == card


def test_a_slot_that_is_no_mapping_reads_as_any_other_slot() -> None:
    base = {**_BASE_IDENTITY, "slots": ["odd", {"property": "a", "value": 2}]}
    served = {**base, "entities": {"destination": ["country/UGA"], "origin": ["country/FRA"]}}

    scope = card_unit._scope_over(base, served)

    assert scope == {
        "entities": {"destination": ["dcid:country/UGA"], "origin": ["dcid:country/FRA"]}
    }


@pytest.mark.parametrize(
    ("entities", "refused"),
    [(["country/UGA", "country/KEN"], True), (["country/KEN", "country/TZA"], False)],
    ids=["same-places-reordered", "other-places"],
)
def test_the_same_places_in_another_order_are_no_change(entities: list[str], refused: bool) -> None:
    offer = CardChanges(
        spec_id=_BASE,
        places=[
            PlacesOffer(
                role="destination",
                multiple=True,
                chosen=[
                    PlaceOption(id="country/KEN", label="Kenya"),
                    PlaceOption(id="country/UGA", label="Uganda"),
                ],
            )
        ],
    )
    change = PlacesChange(kind="places", role="destination", entities=entities)

    assert (card_unit._places_refusal(offer, change) is not None) is refused


def test_a_stored_card_change_mark_that_does_not_validate_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        mark = _read_card_change({"component_id": "cmp_1", "kind": "later_kind"})

    assert mark is None
    assert caplog.records


@pytest.mark.parametrize(
    ("kind", "select", "several"),
    [
        ("line_chart", {"stat_var": "dcid:Aid"}, True),
        ("line_chart", {"stat_var": "dcid:Aid", "recipient": "country/KEN"}, True),
        ("line_chart", {"stat_var": ["dcid:Aid", "dcid:Loans"], "recipient": "country/KEN"}, False),
        ("bar_chart", {"stat_var": "dcid:Aid", "recipient": "country/KEN"}, False),
        ("bar_chart", {"stat_var": "dcid:Aid", "recipient": ["country/KEN"]}, True),
        ("bar_chart", {"stat_var": ["dcid:Aid", "dcid:Loans"], "recipient": "country/KEN"}, False),
        ("table", {"stat_var": ["dcid:Aid", "dcid:Loans"], "recipient": "country/KEN"}, True),
        ("stat_card", {"stat_var": "dcid:Aid", "recipient": "country/KEN"}, False),
    ],
    ids=[
        "line-place-unbound",
        "line-one-place",
        "line-listing-another-dimension",
        "bar-over-time",
        "bar-one-bar-per-place",
        "bar-listing-another-dimension",
        "table",
        "key-number",
    ],
)
def test_a_role_takes_several_places_where_each_draws_its_own_series(
    kind: str, select: dict[str, Any], several: bool
) -> None:
    """dfo-apis' assembler draws a line chart over Kenya and Uganda as two
    series, from a select listing both or leaving the place unbound. It
    refuses a line or bar chart listing two dimensions. It draws a bar chart
    of two places as one bar each at one period, so a bar chart over time
    keeps one."""
    intent = {"type": kind, "select": select}

    assert card_unit.takes_several(intent, ["country/KEN"]) is several


_G7 = ["country/USA", "country/JPN", "country/DEU", "country/FRA"]


@pytest.mark.parametrize(
    ("select", "others", "several"),
    [
        ({"stat_var": "dcid:Aid", "recipient": "country/KEN"}, [_G7], False),
        ({"stat_var": "dcid:Aid", "recipient": "country/KEN"}, [["country/FRA"]], True),
        (
            {"stat_var": "dcid:Aid", "recipient": "country/KEN", "donor": "country/FRA"},
            [["country/FRA"]],
            True,
        ),
        ({"stat_var": "dcid:Aid", "recipient": "country/KEN", "donor": _G7}, [_G7], False),
    ],
    ids=[
        "other-role-unbound-over-several",
        "other-role-one-place",
        "other-role-pinned",
        "other-role-listed",
    ],
)
def test_a_role_beside_one_drawing_several_series_keeps_one_place(
    select: dict[str, Any], others: list[list[str]], several: bool
) -> None:
    """A line chart whose donors draw seven series cannot draw a second
    recipient as well."""
    intent = {"type": "line_chart", "select": select}

    assert card_unit.takes_several(intent, ["country/KEN"], others) is several


def _france_kenya_block() -> dict[str, Any]:
    envelope = json.loads(
        (
            Path(__file__).parents[2]
            / "data-mcp"
            / "tests"
            / "fixtures"
            / "resolve_aid_france_kenya.json"
        ).read_text()
    )
    return envelope["parts"][0]["spec"]


_FETCHED_2016_2024 = {"window": {"start": "2016", "end": "2024"}}


@pytest.mark.parametrize(
    ("intent", "drawn", "window"),
    [
        (
            {"type": "stat_card", "select": {"stat_var": "dcid:Aid"}},
            {"type": "stat_card", "period": "2023"},
            (2023, 2023, True),
        ),
        (
            {"type": "bar_chart", "select": {"destination": ["country/KEN", "country/UGA"]}},
            {"type": "bar_chart", "category_axis": {"mode": "entity", "period": "2021"}},
            (2021, 2021, True),
        ),
        ({"type": "stat_card", "select": {"stat_var": "dcid:Aid"}}, {"type": "stat_card"}, None),
        (
            {"type": "line_chart", "select": {"stat_var": "dcid:Aid"}},
            {"type": "line_chart"},
            (2016, 2024, None),
        ),
    ],
    ids=["key-number", "bar-per-member", "key-number-period-unread", "line-chart"],
)
def test_a_card_showing_one_value_offers_the_year_it_draws(
    intent: dict[str, Any], drawn: dict[str, Any], window: tuple[int, int, bool | None] | None
) -> None:
    """Fetched over 2016 to 2024, a card showing one value per series draws
    one year, which its offer names as both bounds, read off the drawn
    payload, since the platform's fact summary states no window for it. One
    whose drawn year cannot be read offers no years. Other cards offer the
    years their fact summary states."""
    block = _france_kenya_block()
    card = card_unit.encoded(
        card_unit.UnitCard(spec_id=block["spec_id"], args={}),
        block,
        intent=intent,
        component={"payload": {"component": drawn}},
    )

    offer = card_unit.card_changes(card, block, fact_summary=_FETCHED_2016_2024)

    offered = offer.window if offer is not None else None
    got = (offered.start_year, offered.end_year, offered.single) if offered else None
    assert got == window
    if offered is not None and offered.single:
        assert offered.start_year == offered.end_year == card.drawn_year


def test_a_single_window_naming_a_range_is_refused_on_the_wire() -> None:
    with pytest.raises(ValidationError):
        WindowOffer(start_year=2016, end_year=2024, first_year=1970, last_year=2024, single=True)


async def test_a_roles_places_are_listed_by_label() -> None:
    """Search lists places by ref; a reader reads them by name, accents and
    case ignored."""
    listed = (
        ("country/ABW", "Aruba"),
        ("country/AFG", "Afghanistan"),
        ("country/CIV", "Côte d'Ivoire"),
        ("country/TUR", "Türkiye"),
        ("country/TZA", "Tanzania"),
        ("country/ZAF", "south Africa"),
        ("country/CMR", "Cameroon"),
    )

    async def door(**_: Any) -> tuple[tuple[tuple[str, str], ...], bool]:
        return listed, False

    read = await place_options.read_role_places(door, identity={}, role="destination")

    assert [label for _, label in read.places] == [
        "Afghanistan",
        "Aruba",
        "Cameroon",
        "Côte d'Ivoire",
        "south Africa",
        "Tanzania",
        "Türkiye",
    ]


# --- a card on a sum of its one retrieval -----------------------------------

_OPS = ("ops_" + "c" * 32, "ocv_" + "3" * 32)
_SUM = {"aggregation": "sum", "group_by": ["time_period", "financingRecipient"]}


def _sum_node(parameters: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "op_spec": {
            "identity_version": "ops1",
            "verb": "aggregate",
            "frame_ref": {"spec_id": _SPEC, "content_version": _CV},
            "parameters": dict(parameters),
        },
        "inputs": dict(inputs),
    }


def _root(spec_id: str = _SPEC) -> dict[str, Any]:
    pair = {"spec_id": spec_id, "content_version": _CV}
    return {"schema_version": 1, "frame": pair, "dataset": pair, "options": {}, "scope": {}}


def _sum_recipe(**changed: Any) -> dict[str, Any]:
    """The recipe dfo-apis' runner stores for a sum by year of one
    retrieval, with the members `changed` names in place."""
    return {
        "schema_version": 1,
        "tree": _sum_node(_SUM, {"frame_ref": {"root": 0}}),
        "roots": [_root()],
        "anchor": None,
        "left_out": None,
        **changed,
    }


@pytest.mark.parametrize(
    ("recipe", "summed"),
    [
        (_sum_recipe(), True),
        (_sum_recipe(roots=[_root(), _root("spec_" + "f" * 32)]), False),
        (_sum_recipe(anchor={"period": "2023", "rule": "half", "left_out": []}), False),
        (
            _sum_recipe(
                tree=_sum_node(
                    {"aggregation": "sum", "group_by": ["financingRecipient"]},
                    {"frame_ref": {"root": 0}},
                )
            ),
            False,
        ),
        (
            _sum_recipe(tree=_sum_node(_SUM | {"aggregation": "mean"}, {"frame_ref": {"root": 0}})),
            False,
        ),
        (
            _sum_recipe(
                tree=_sum_node(_SUM, {"frame_ref": _sum_node(_SUM, {"frame_ref": {"root": 0}})})
            ),
            False,
        ),
        (
            _sum_recipe(tree=_sum_node(_SUM | {"anchor_key": "k"}, {"frame_ref": {"root": 0}})),
            False,
        ),
        (_sum_recipe(roots=[_root() | {"scope": {"columns": ["party_kind"]}}]), False),
    ],
    ids=[
        "one-sum-by-year",
        "two-roots",
        "anchor",
        "no-period",
        "another-function",
        "two-nodes",
        "unknown-parameter",
        "unknown-root-scope",
    ],
)
async def test_only_one_sum_by_year_of_one_retrieval_can_change(
    monkeypatch: pytest.MonkeyPatch, recipe: dict[str, Any], summed: bool
) -> None:
    """A card on a derived frame reads its recipe once, and changes only when
    the recipe is one `aggregate` sum by year over the card's one retrieval:
    the card then reads the retrieval as a plain card does, keeping the sum,
    and stores it where a release that reads no sum finds no arguments."""
    read: list[object] = []

    async def recipe_of(ref: object) -> dict[str, Any]:
        read.append(ref)
        return recipe

    monkeypatch.setattr(card_unit, "derived_recipe", recipe_of)
    lineage = _lineage(_OPS)

    total = await card_unit.read_summed_frame(lineage)
    card = unit_card(_component(_OPS[0]), lineage, total=total)

    assert len(read) == 1
    assert (card is not None) is summed
    if card is not None:
        assert (card.spec_id, card.args, card.total) == (
            _SPEC,
            {},
            card_unit.SummedFrame(frame=_OPS, root=(_SPEC, _CV), parameters=_SUM),
        )
        stored = card.as_json()
        assert "args" not in stored
        assert card_unit.UnitCard.from_json(stored) == card


async def test_a_card_on_its_retrievals_own_frame_reads_no_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a derived frame resting on one retrieval has its recipe read."""

    async def recipe_of(ref: object) -> dict[str, Any]:
        raise AssertionError(f"read the recipe of {ref}")

    monkeypatch.setattr(card_unit, "derived_recipe", recipe_of)

    assert await card_unit.read_summed_frame(_lineage((_SPEC, _CV))) is None
    assert await card_unit.read_summed_frame(_lineage(_OPS, members=2)) is None


def _summed_card(shown: dict[str, list[str]], *, cut: bool = False) -> card_unit.UnitCard:
    """A line chart on the sum by year and recipient of France to Kenya,
    showing the places `shown` names."""
    block = _france_kenya_block()
    return card_unit.UnitCard(
        spec_id=block["spec_id"],
        args={},
        scope={"entities": {role: [f"dcid:{ref}" for ref in refs] for role, refs in shown.items()}},
        labels={"country/KEN": "Kenya", "country/FRA": "France", "country/DEU": "Germany"},
        roles=dict.fromkeys(shown, True),
        shown=shown,
        most=dict.fromkeys(shown, 10),
        cut=cut,
        years=card_unit.ScopeYears(
            spec_id=block["spec_id"],
            entities=shown,
            first=1978,
            last=2024,
        ),
        total=card_unit.SummedFrame(frame=_OPS, root=(_SPEC, _CV), parameters=_SUM),
    )


@pytest.mark.parametrize(
    ("donors", "offered"),
    [
        (["country/FRA"], {"destination": (True, 10), "origin": (False, 1)}),
        (["country/FRA", "country/DEU"], {"destination": (True, 10)}),
    ],
    ids=["one-donor", "summed-donors"],
)
def test_a_sum_offers_the_places_it_keeps_apart_and_not_the_ones_it_adds_up(
    donors: list[str], offered: dict[str, tuple[bool, int]]
) -> None:
    """The sum keeps each recipient apart and adds up the donors. The
    recipient changes as on a plain card. One donor changes only to one other,
    so the sum still runs over one donor; two donors summed offer no change,
    since others would give another sum under the same title. Its unit
    cannot change."""
    block = _france_kenya_block()
    card = _summed_card({"destination": ["country/KEN"], "origin": donors})

    offer = card_unit.card_changes(card, block, fact_summary=_FETCHED_2016_2024, places=True)

    assert offer is not None and offer.window is not None
    assert {p.role: (p.multiple, p.max) for p in offer.places or ()} == offered
    assert card_unit.card_unit_change(card, block) is None


def test_a_sum_cut_to_its_largest_members_offers_nothing() -> None:
    block = _france_kenya_block()
    card = _summed_card({"destination": ["country/KEN"], "origin": ["country/FRA"]}, cut=True)

    assert card_unit.card_changes(card, block, fact_summary=_FETCHED_2016_2024, places=True) is None
