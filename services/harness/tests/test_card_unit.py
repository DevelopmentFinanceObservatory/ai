"""Tests for `harness.card_unit`: which cards can change unit, and how their frame and title move."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from common.live_apis import LiveApisError
from data_mcp.get_data_tool import _SUM_STAT_VARS_GROUPBY, Breakdown
from harness import place_options
from harness.card_unit import frame_arguments, retitled, unit_card
from harness.provenance import CardLineage, ProvenanceInput
from harness.wire import UnitChoice, UnitCurrency, UnitOffer, UnitPrices

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
    scoped, with the places and window it changed; one that differs from
    every block elsewhere reads its own spec."""
    lineage = _lineage((_SPEC, _CV))
    (member,) = lineage.lineage
    member.manifest["definition"]["identity"] = served
    blocks = {_BASE: {"definition": {"spec_id": _BASE, "identity": _BASE_IDENTITY}}}

    card = unit_card(_component(), lineage, blocks)

    assert card is not None
    assert (card.spec_id, card.scope) == (spec_id, scope)


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
    monkeypatch: pytest.MonkeyPatch, door: _Door
) -> None:
    monkeypatch.setattr(place_options, "PLACE_OPTIONS_DEADLINE_S", 0.05)

    with pytest.raises(place_options.PlacesUnavailableError):
        await place_options.read_role_places(door, identity=_BASE_IDENTITY, role="destination")


async def test_a_place_read_keeps_the_cap_and_says_it_cut_the_list() -> None:
    read = await place_options.read_role_places(
        _Door(place_options.PLACE_OPTIONS_CAP + 1), identity=_BASE_IDENTITY, role="destination"
    )

    assert (len(read.places), read.truncated) == (place_options.PLACE_OPTIONS_CAP, True)
