"""`ConversationState` and `StudioStore`.

The turn's `studio.turn` line and outcome observation, which a Studio write
writes once it commits a close and the turn's cursor write once it commits a
pause (`harness.studio_store.write_pause_line`), are pinned against Postgres
near the bottom of the file, with `common.telemetry.record_turn_outcome`
faked to prove the wiring; its own backend handling is covered by
`packages/common/tests/test_telemetry.py`.

The canvas and conversation digests the models read carry a stated maximum
size (`CANVAS_LISTING_MAX_BYTES`, `CONVERSATION_DIGEST_MAX_BYTES`); the tests
below feed each one oversized input and assert the maximum holds.

The canvas digest entry is pinned here too: what a card shows (its entities
and span) travels in the digest the models read, from the names the
component's own fact summary and stored intent carry.

The store (`StudioStore.create`/`load`/`save`/`close_segment`) and
the segment lease (`claim_segment`/`renew`/`request_cancel`) are exercised
at the bottom of this file against Postgres (the `pg_engine` fixture), each
test in a schema of its own.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import errno
import json
import logging
import ssl
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import harness.turn.store as turn_store_module
import pytest
from common.logging import JSON_FIELDS
from compiler.sources import source_from_spec_block
from fastapi.responses import JSONResponse
from harness import studio_state, studio_store
from harness.approval import ApprovalDecision, ApprovalRecord
from harness.chart_binding import MAX_CHARTS_PER_CALL
from harness.conversation import current_conversation_key
from harness.db import engine as engine_module
from harness.db.commit import transaction_committed
from harness.graph.state import (
    BUDGETS,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
)
from harness.provenance import CardLineage, ProvenanceInput
from harness.studio import _answers_refusal, _resume_answer
from harness.studio_edit import _record_revision, _refresh_block
from harness.studio_state import (
    CANVAS_LISTING_MAX_BYTES,
    CANVAS_SUMMARY_MAX_BYTES,
    CONVERSATION_DIGEST_MAX_BYTES,
    ConversationState,
    DatabaseUnavailable,
    Segment,
    SegmentBusy,
    SegmentLost,
    TurnUsage,
    _digest_entry,
    _next_manifest,
    _PendingInterrupt,
    _PendingQuestion,
    land_component,
    make_recall_part,
    remove_cards,
)
from harness.studio_store import (
    SEQ_BLOCK,
    ConversationStale,
    Renewal,
    SegmentClose,
    StopPending,
    StudioStore,
    TurnMoved,
)
from harness.turn.store import TurnStatus, TurnStore
from harness.turn_records import PlanRecord, RecordedIdentity, SearchReading
from harness.unit_offer import whole_unit_options, whole_units
from harness.wire import (
    AddOp,
    ClientFeature,
    ComponentType,
    CompositionActor,
    CompositionBlock,
    CompositionManifest,
    ElicitationInput,
    ElicitationRequest,
    ErrorCode,
    MessageComplete,
    MutationOp,
    OptionAnswer,
    PlanProposed,
    UnitOffer,
)
from planner.plan import PlanStatus
from planner.workflow import SelectedSource
from sqlalchemy import ARRAY, Text, bindparam, text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from . import drawn_cards
from .resolved_revision import resolved_revision
from .store_clock import StoreClock
from .test_turn_records import _scoped_part, _top_donors
from .test_turn_store import ADMITTED_RUN

CONVERSATION = "conv_01test0000"

_UNIT_OFFER = UnitOffer.model_validate(
    {
        "currencies": [
            {
                "code": "USD",
                "name": "US dollars",
                "prices": ["current", "constant"],
                "base_year_min": 2000,
                "base_year_max": 2023,
            }
        ],
        "presets": ["USD"],
        "start": {"currency": "USD", "prices": "constant", "base_year": 2023},
    }
)

# A unit question as `/interrupt` keeps it: its input, the offer an answer must lie
# inside, and its whole units, which a page sent it as a pick answers with.
_UNIT_QUESTION = _PendingQuestion(
    id="q2",
    question="In which prices?",
    options=dict(whole_unit_options(_UNIT_OFFER)),
    skippable=True,
    input=ElicitationInput.UNIT,
    unit=_UNIT_OFFER,
)


# The same question sent to a client that renders no unit question: a pick
# among the offer's whole units, which keeps the offer and its own form.
_UNIT_AS_PICK = dataclasses.replace(
    _UNIT_QUESTION,
    input=ElicitationInput.SINGLE_SELECT,
    options={"USD-constant-2023": "Constant 2023 US dollars", "USD-current": "Current US dollars"},
    native_input=ElicitationInput.UNIT,
)

_ACCOUNT = "a" * 64
_VISITOR = "ip:" + "b" * 32


def _spy_outcomes(monkeypatch) -> list[dict[str, Any]]:
    """Fake `record_turn_outcome` in `harness.studio_store` and return the
    arguments of every outcome written."""
    written: list[dict[str, Any]] = []
    monkeypatch.setattr(
        studio_store,
        "record_turn_outcome",
        lambda trace_id, **kwargs: written.append({"trace_id": trace_id, **kwargs}),
    )
    return written


@pytest.mark.parametrize("recorded", [None, _VISITOR], ids=["no-one", "visitor"])
def test_a_turn_that_recorded_no_one_takes_the_resuming_users_identity(recorded) -> None:
    """`take_identity` fills in a turn that recorded no one, which its later
    segments and its outcome then name as their user. A turn that recorded a
    visitor keeps it, and takes the caller and surface it recorded none of."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.begin_turn(TurnUsage(visitor=recorded))

    state.take_identity(
        account=_ACCOUNT, visitor=None, caller="conversation:" + "c" * 32, surface="studio"
    )

    assert state.turn_usage.user_id == (recorded or _ACCOUNT)
    assert state.turn_usage.caller == "conversation:" + "c" * 32
    assert state.turn_usage.surface == "studio"


# --- land_component: the one place a card lands on the canvas -------------


def _component(component_id: str, *, revision: int = 1, kind: str = "line_chart") -> dict[str, Any]:
    """A minimal `write_component` result: just what `land_component` and
    `_digest_entry` read (`component_id`, `type`, `revision`) -- the rest
    of a real platform payload is exercised in `test_studio_translate.py`'s
    `_written_component`."""
    return {"component_id": component_id, "revision": revision, "type": kind}


def test_land_component_appends_a_block() -> None:
    state = ConversationState(conversation_id=CONVERSATION)

    action = studio_state.land_component(state, _component("cmp_two"))

    assert action == "created"
    assert [b.component_id for b in state.composition.blocks] == ["cmp_two"]


def test_land_component_held_when_auto_add_is_off() -> None:
    """`auto_add=False` means there is no block to append: the digest entry still gets written (the card is still
    findable by `recall`), but the composition itself is untouched and the
    return is `"held"`."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.auto_add = False

    action = studio_state.land_component(state, _component("cmp_two"))

    assert action == "held"
    assert state.composition is None
    assert "cmp_two" in state.components


def _titled(component_id: str, title: str | None) -> dict[str, Any]:
    """`_component` titled `title`, where given, as a chart's payload titles it."""
    payload = {"payload": {"component": {"title": {"text": title}}}} if title else {}
    return {**_component(component_id), **payload}


@pytest.mark.parametrize(
    ("titles", "removed", "auto_add", "shown", "named", "unnamed"),
    [
        pytest.param(
            ["Aid to Kenya", "Aid to Ghana"],
            ["cmp_0"],
            True,
            ["cmp_1"],
            ["Aid to Ghana"],
            ["Aid to Kenya", "cmp_"],
            id="a-card-taken-off",
        ),
        pytest.param(
            ["Aid to Kenya", None],
            [],
            True,
            ["cmp_0", "cmp_1"],
            ["line_chart"],
            ["cmp_"],
            id="an-untitled-card",
        ),
        pytest.param(
            ["Aid to Kenya"],
            [],
            False,
            ["cmp_0"],
            ["Aid to Kenya"],
            ["cmp_"],
            id="held-off-the-canvas",
        ),
        pytest.param(
            [f"Aid to country {n}" for n in range(10)],
            [],
            True,
            [f"cmp_{n}" for n in range(10)],
            ["Aid to country 7", "2 more"],
            ["Aid to country 8", "cmp_"],
            id="past-the-card-cap",
        ),
        pytest.param(
            ["Aid " + "x" * 200],
            [],
            True,
            ["cmp_0"],
            ["Aid xxx"],
            ["x" * 150],
            id="a-long-title",
        ),
    ],
)
def test_a_cards_answer_names_by_title_the_cards_the_user_still_sees(
    titles: list[str | None],
    removed: list[str],
    auto_add: bool,
    shown: list[str],
    named: list[str],
    unnamed: list[str],
) -> None:
    """The answer kept for a turn its cards answered names, for a later turn's model,
    the cards the turn drew that the user still sees, within the digest's bounds on a
    turn's cards. A card the turn took off again is left out, a card held off the
    canvas stays, and no component id is named."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.auto_add = auto_add
    for n, title in enumerate(titles):
        land_component(state, _titled(f"cmp_{n}", title))
    remove_cards(state, removed, all_cards=False)

    answer = state.cards_answer()

    assert [card["component_id"] for card in state.turn_cards_shown()] == shown
    assert all(name in answer for name in named), answer
    assert not any(name in answer for name in unnamed), answer


# --- the canvas digest entry: what each card shows ---------------------------


def _two_entity_component() -> dict:
    """A platform-shaped two-country card. Its fact summary names both
    entities."""
    series = [
        {
            "id": code.rsplit("/", 1)[-1].lower(),
            "label": label,
            "point_count": 64,
            "coordinates": {"stat_var": "dcid:LifeExpectancy_Person", "observationAbout": code},
        }
        for code, label in (("country/GTM", "Guatemala"), ("country/NGA", "Nigeria"))
    ]
    return {
        "component_id": "cmp_two",
        "revision": 1,
        "type": "line_chart",
        "payload": resolved_revision(
            "line_chart",
            component_id="cmp_two",
            revision=1,
            title="Life expectancy in Guatemala and Nigeria",
        ),
        "fact_summary": {
            "component_id": "cmp_two",
            "revision": 1,
            "series": series,
            "attribution": {
                "label": "WDI · 1 indicator(s)",
                "source_count": 1,
                "indicator_count": 1,
            },
            "window": {"start": "1960", "end": "2023"},
        },
    }


def test_digest_entry_carries_the_entities_and_span_the_card_shows() -> None:
    entry = _digest_entry(_two_entity_component())

    assert entry["type"] == "line_chart"
    assert entry["title"] == "Life expectancy in Guatemala and Nigeria"
    assert entry["entities"] == ["Guatemala", "Nigeria"]
    assert entry["time"] == "1960-2023"


def test_digest_entry_takes_nothing_but_the_title_from_the_resolved_revision() -> None:
    """The digest's names come from the wire fact summary alone. A summary
    without labels leaves `entities` and `time` out, although
    the resolved revision in `payload` names its own series and window."""
    component = _two_entity_component()
    component["fact_summary"] = {
        "component_id": "cmp_two",
        "revision": 1,
        "series": [],
        "attribution": {"label": "WDI", "source_count": 1, "indicator_count": 1},
    }
    assert component["payload"]["component"]["series"][0]["label"]

    entry = _digest_entry(component)

    assert entry["title"] == "Life expectancy in Guatemala and Nigeria"
    assert "entities" not in entry
    assert "time" not in entry


def test_digest_entry_stays_bounded_however_many_series_the_frame_holds() -> None:
    """The digest is model context: an entry's size stays capped no matter
    how many series the frame holds."""
    component = _two_entity_component()
    many = [
        {
            "id": f"s{index}",
            "label": "Entity " + "long" * 60 + f" {index}",
            "point_count": 1,
            "coordinates": {},
        }
        for index in range(200)
    ]
    component["fact_summary"]["series"] = many

    entry = _digest_entry(component)

    assert len(entry["entities"]) == studio_state._DIGEST_SHOW_CAP
    state = _canvas({"cmp_two": entry})
    digest = state.canvas_listing_state()
    assert digest is not None
    (block,) = digest["blocks"]
    assert len(block["entities"]) == studio_state._DIGEST_SHOW_CAP
    assert all(
        _serialised(name) <= studio_state._DIGEST_NAME_BYTE_CAP for name in block["entities"]
    )
    assert block["entities_truncated"] == {
        "showing": studio_state._DIGEST_SHOW_CAP,
        "total": 200,
        "rule": "display order",
    }


def test_digest_entry_omits_what_a_component_does_not_show() -> None:
    """A note card names no entity and no span: it keeps neither key
    rather than carrying empty lists. It carries no title
    either, so its type stands in."""
    entry = _digest_entry(
        {
            "type": "note_narration",
            "revision": 1,
            "payload": resolved_revision("note_narration"),
            "fact_summary": None,
        }
    )

    assert entry["title"] == "note_narration"
    assert "entities" not in entry
    assert "time" not in entry


BIG = 100_000
# A code point outside the basic plane: `json.dumps` escapes it as a surrogate
# pair, twelve bytes per character.
ASTRAL = "\U0001f600"


def _fact_summary(series: int = 1) -> dict[str, Any]:
    """A fact summary shaped like `mcp_client._wire_fact_summary`'s output."""
    return {
        "component_id": "cmp_01j9zc3aqw",
        "revision": 1,
        "window": {"start": "1960", "end": "2023"},
        "series": [
            {
                "id": f"series_{index}",
                "label": "Democratic Republic of the Congo",
                "point_count": 64,
                "coordinates": {"unit": "Year", "observationAbout": "country/COD"},
                "first": 48.512,
                "last": 63.612,
            }
            for index in range(series)
        ],
        "attribution": {"label": "World Bank IDS", "source_count": 1, "indicator_count": series},
    }


def _entry(
    title: str,
    fact_summary: Any,
    *,
    revision: int = 1,
    shows: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "line_chart",
        "title": title,
        "revision": revision,
        "fact_summary": fact_summary,
        **(shows or {}),
    }


def test_a_cards_series_ids_are_stored_whole_and_stated_to_no_reader() -> None:
    """A stored entry holds the id of every series its card draws, past the
    twelve the platform's fact summary lists. No canvas view, recall or
    manifest states them, and a revised card's ids follow its new payload."""
    codes = [f"country/P{number:02d}" for number in range(24)]
    series = {code: {"2024": 1.0} for code in codes}
    card = drawn_cards.write_reply(
        drawn_cards.chart("bar_chart", series), drawn_cards.stored(codes)
    )
    state = ConversationState(conversation_id=CONVERSATION)
    intent = {"type": "bar_chart", "title": "t", "select": {"observationAbout": codes}}
    land_component(state, card, intent=intent)
    card_id = card["component_id"]

    ids = [drawn_cards.series_id(code) for code in codes]
    assert state.canvas_series_ids() == ids
    assert state.canvas_cards() == [
        {"component_id": card_id, "intent": intent, "revision": card["revision"]}
    ]
    assert len(state.components[card_id]["fact_summary"]["series"]) == 12
    model_facing = json.dumps(
        [
            state.canvas_summary_state(focus=card_id),
            state.canvas_listing_state(),
            state.recall_part(part="canvas"),
            state.recall_part(part="canvas", component_id=card_id),
            state.composition.model_dump(mode="json"),
        ]
    )
    assert "series_ids" not in model_facing
    assert "drawn_card" not in model_facing
    assert ids[11] in model_facing
    assert ids[12] not in model_facing

    assert studio_state.drawn_series_ids({"component_id": card_id, "revision": 2}) == {}
    fewer = drawn_cards.write_reply(
        drawn_cards.chart("bar_chart", {codes[0]: {"2024": 1.0}}), drawn_cards.stored(codes[:1])
    )
    state.components[card_id].update(studio_state.drawn_series_ids(fewer))
    assert state.canvas_series_ids() == ids[:1]


def test_a_drawn_card_is_listed_while_the_canvas_holds_the_revision_it_was_drawn_as() -> None:
    """A card taken off the canvas is not listed, and listed again once the
    user adds it back unchanged; one the canvas holds at a later revision is
    not listed."""
    state = ConversationState(conversation_id=CONVERSATION)
    for card_id in ("cmp_one", "cmp_two"):
        card = {"component_id": card_id, "type": "line_chart", "revision": 1}
        land_component(state, card, intent={"title": card_id})

    def listed() -> list[str]:
        return [card["component_id"] for card in state.canvas_cards()]

    assert studio_state.remove_cards(state, ["cmp_one"], all_cards=False)[0] == ["cmp_one"]
    assert listed() == ["cmp_two"]

    applied = studio_state._apply_operations(
        state.components, state.composition, [AddOp(op=MutationOp.ADD, component_id="cmp_one")]
    )
    assert isinstance(applied, tuple)
    state.composition = _next_manifest(state.composition, applied[0], actor=CompositionActor.USER)
    assert listed() == ["cmp_two", "cmp_one"]

    revised = [
        block.model_copy(update={"revision": 2}) if block.component_id == "cmp_two" else block
        for block in state.composition.blocks
    ]
    state.composition = _next_manifest(state.composition, revised, actor=CompositionActor.USER)
    assert listed() == ["cmp_one"]


def test_an_entry_stored_without_series_ids_gives_those_its_fact_summary_lists() -> None:
    """An entry a conversation stored before entries held series ids loads
    as it is, and the canvas's series are then the ones its summary lists."""
    entry = _digest_entry(_two_entity_component())
    stored = {key: value for key, value in entry.items() if key != "series_ids"}
    assert stored != entry

    assert _canvas({"cmp_two": stored}).canvas_series_ids() == ["gtm", "nga"]
    assert _canvas({"cmp_two": {"type": "line_chart", "revision": 1}}).canvas_series_ids() == []


def _shows(name: str, time: str, *, count: int = studio_state._DIGEST_SHOW_CAP) -> dict[str, Any]:
    """The `entities` / `time` keys `_digest_shows` adds to an entry."""
    return {"entities": [name] * count, "time": time}


def _canvas(entries: dict[str, dict[str, Any]], *, turn: int | None = None) -> ConversationState:
    """A conversation whose composition holds every entry, in order, each drawn
    by the open turn, or with `turn` by that closed turn (1-based) after
    `turn - 1` turns that drew nothing."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.components.update(entries)
    cards = [{"component_id": component_id, "action": "created"} for component_id in entries]
    if turn is None:
        state.turn_components = cards
    else:
        state.turns = [{"user": "u", "assistant": "a"} for _ in range(turn - 1)]
        state.turns.append({"user": "u", "assistant": "a", "cards": cards})
    blocks = [
        CompositionBlock(
            component_id=component_id,
            revision=entry["revision"],
            type=ComponentType(entry["type"]),
            title="block",
        )
        for component_id, entry in entries.items()
    ]
    state.composition = _next_manifest(None, blocks, actor=CompositionActor.MODEL)
    return state


def _serialised(value: Any) -> int:
    return len(json.dumps(value))


def test_a_realistic_canvas_block_states_its_shared_coordinates_once():
    """Every series of the four-series block holds the same coordinates, so
    the block states them once and no series repeats them. The fact
    summary's component id and revision, which the block carries, go."""
    entries = {
        "cmp_01j9zc3aqw": _entry("Life expectancy in Kenya (1960-2023)", _fact_summary(1)),
        "cmp_01j9zc3aqx": _entry("Aid to five East African countries, 2000-2023", _fact_summary(4)),
        "cmp_01j9zc3aqy": _entry("Cote d'Ivoire \u2014 cocoa exports", None),
    }
    state = _canvas(entries)

    digest = state.canvas_listing_state()

    assert digest is not None

    def factored(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "window": summary["window"],
            "attribution": summary["attribution"],
            "shared_coordinates": {"unit": "Year", "observationAbout": "country/COD"},
            "series": [
                {key: value for key, value in series.items() if key != "coordinates"}
                for series in summary["series"]
            ],
        }

    assert digest["blocks"] == [
        {
            "component_id": component_id,
            **entry,
            "fact_summary": entry["fact_summary"] and factored(entry["fact_summary"]),
            "turn": 1,
        }
        for component_id, entry in entries.items()
    ]


def test_a_coordinate_one_series_differs_in_stays_on_every_series():
    """Only a coordinate every series holds at the same value is factored.
    `provenance` differs on one series and `financingProvider` on all, so
    each series keeps its own value of both; a coordinate one series lacks
    stays with the series that hold it."""
    summary = _fact_summary(3)
    providers = ["country/USA", "country/ESP", "country/FRA"]
    for series, provider in zip(summary["series"], providers, strict=True):
        series["coordinates"] = {
            "unit": "CurrencyUSD",
            "stat_var": "dcid:spec_f8bf",
            "provenance": "provenance/CRS",
            "financingProvider": provider,
        }
    summary["series"][1]["coordinates"]["provenance"] = "provenance/Imputed"
    summary["series"][2]["coordinates"]["measurementMethod"] = "Modelled"
    state = _canvas({"cmp_01": _entry("Aid to Guatemala", summary)})

    digest = state.canvas_listing_state()

    assert digest is not None
    shown = digest["blocks"][0]["fact_summary"]
    assert shown["shared_coordinates"] == {"unit": "CurrencyUSD", "stat_var": "dcid:spec_f8bf"}
    assert [series["coordinates"] for series in shown["series"]] == [
        {"provenance": "provenance/CRS", "financingProvider": "country/USA"},
        {"provenance": "provenance/Imputed", "financingProvider": "country/ESP"},
        {
            "provenance": "provenance/CRS",
            "financingProvider": "country/FRA",
            "measurementMethod": "Modelled",
        },
    ]


def test_an_oversized_title_is_cut_and_says_so():
    state = _canvas({"cmp_01": _entry("t" * BIG, _fact_summary())})

    digest = state.canvas_listing_state()

    assert digest is not None
    title = digest["blocks"][0]["title"]
    assert title.endswith(studio_state._CUT_MARKER)
    assert title.startswith("ttt")
    assert _serialised(title) <= studio_state._TITLE_BYTE_CAP


def test_a_title_of_escaped_characters_is_cut_by_its_serialised_size():
    state = _canvas({"cmp_01": _entry(ASTRAL * 1_000, _fact_summary())})

    digest = state.canvas_listing_state()

    assert digest is not None
    title = digest["blocks"][0]["title"]
    assert title.endswith(studio_state._CUT_MARKER)
    assert _serialised(title) <= studio_state._TITLE_BYTE_CAP


def test_an_oversized_fact_summary_keeps_its_leading_series_and_labels_the_subset():
    """A summary over its cap keeps the series that fit from the front, in
    display order, and states how many of the card's series it shows and by
    which rule."""
    series = _fact_summary(BIG // 500)["series"]
    for index, item in enumerate(series):
        item["coordinates"] = {"unit": "Year", "observationAbout": f"country/C{index:03d}"}
    oversized = {**_fact_summary(), "series": series}
    state = _canvas({"cmp_01": _entry("Aid", oversized)})

    digest = state.canvas_listing_state()

    assert digest is not None
    summary = digest["blocks"][0]["fact_summary"]
    showing = summary["series_truncated"]["showing"]
    assert 0 < showing < len(series)
    assert summary["series_truncated"] == {
        "showing": showing,
        "total": len(series),
        "rule": "display order",
    }
    assert [item["id"] for item in summary["series"]] == [item["id"] for item in series[:showing]]
    assert summary["shared_coordinates"] == {"unit": "Year"}
    assert _serialised(summary) <= studio_state._FACT_SUMMARY_BYTE_CAP
    assert (
        _serialised({**summary, "series": series[: showing + 1]})
        > studio_state._FACT_SUMMARY_BYTE_CAP
    )


@pytest.mark.parametrize(
    ("recipe_bytes", "series", "kept", "cut"),
    [
        (studio_state._FACT_SUMMARY_BYTE_CAP, 3, False, False),
        (studio_state._UNIT_OBJECT_BYTE_CAP + 100, 3, True, False),
        (0, BIG // 500, True, True),
    ],
    ids=["crowds-the-series-out", "over-its-cap-but-the-summary-fits", "small-beside-cut-series"],
)
def test_a_unit_object_is_shed_only_when_it_is_over_its_cap_and_the_summary_is_over_the_cap(
    recipe_bytes: int, series: int, kept: bool, cut: bool
):
    """A unit object the wire forwarded whole becomes `_CUT_MARKER` only when it
    is over its cap and the summary is over the cap with it, and then before any
    series is cut."""
    unit = {"currency": "USD", "recipe": "r" * recipe_bytes}
    summary = {**_fact_summary(series), "unit": unit}
    state = _canvas({"cmp_01": _entry("Aid", summary)})

    digest = state.canvas_listing_state()

    assert digest is not None
    shown = digest["blocks"][0]["fact_summary"]
    assert shown["unit"] == (unit if kept else studio_state._CUT_MARKER)
    assert ("series_truncated" in shown) is cut
    assert (len(shown["series"]) < series) is cut


@pytest.mark.parametrize("oversized", [False, True], ids=["fits", "oversized"])
def test_a_summary_the_platform_cut_states_the_platforms_total(oversized: bool):
    """The platform's `series_truncated` (its series cap) reaches the digest
    with the platform's total, whether or not the harness cuts further."""
    count = BIG // 500 if oversized else 2
    summary = {
        **_fact_summary(count),
        "series_truncated": {"showing": count, "total": count + 40},
    }
    state = _canvas({"cmp_01": _entry("Aid", summary)})

    digest = state.canvas_listing_state()

    assert digest is not None
    block = digest["blocks"][0]["fact_summary"]
    assert block["series_truncated"]["total"] == count + 40
    assert block["series_truncated"]["showing"] == len(block["series"])
    assert block["series_truncated"]["rule"] == "display order"
    assert (len(block["series"]) < count) is oversized


def test_an_oversized_fact_summary_becomes_a_stub_that_says_it_was_cut():
    """A summary that does not fit even with no series, or that is not a
    summary at all, becomes a stub."""
    no_series_fit = {
        **_fact_summary(),
        "attribution": {**_fact_summary()["attribution"], "label": "s" * BIG},
    }
    for oversized in (no_series_fit, "s" * BIG):
        state = _canvas({"cmp_01": _entry("Aid", oversized)})

        digest = state.canvas_listing_state()

        assert digest is not None
        summary = digest["blocks"][0]["fact_summary"]
        assert set(summary) == {"cut"}
        assert str(_serialised(oversized)) in summary["cut"]
        assert _serialised(summary) <= studio_state._FACT_SUMMARY_BYTE_CAP


def test_a_fact_summary_at_its_cap_is_kept_and_one_byte_over_is_cut():
    cap = studio_state._FACT_SUMMARY_BYTE_CAP
    at_cap = "s" * (cap - 2)  # the two quotes complete the cap
    state = _canvas(
        {"cmp_01": _entry("at", at_cap), "cmp_02": _entry("over", at_cap + "s")},
    )

    digest = state.canvas_listing_state()

    assert digest is not None
    kept, cut = digest["blocks"]
    assert kept["fact_summary"] == at_cap
    assert set(cut["fact_summary"]) == {"cut"}


def _largest_canvas() -> ConversationState:
    """A conversation with the canvas listing at its caps: every field of
    every card just inside its cap, and past their caps this turn's cards, the
    pinned cards of an earlier turn, the cards still on the canvas of another,
    and turns of cards off the canvas."""
    title = "t" * (studio_state._TITLE_BYTE_CAP - 2)
    summary = "s" * (studio_state._FACT_SUMMARY_BYTE_CAP - 2)
    shows = _shows(
        "n" * (studio_state._DIGEST_NAME_BYTE_CAP - 2),
        "y" * (studio_state._DIGEST_TIME_BYTE_CAP - 2),
    )
    group = studio_state._DIGEST_BLOCK_CAP + studio_state._LISTING_COMPACT_CAP + 2
    # Oldest first: the pinned cards' turn, turns of one card off the canvas,
    # a turn whose cards fill the off-canvas places, a turn whose cards stay
    # on the canvas, and the open turn.
    sizes = [
        group,
        *[1] * (studio_state._EARLIER_TURN_CAP + 2),
        studio_state._LISTING_COMPACT_CAP,
        group,
        group,
    ]
    off = range(1, len(sizes) - 2)
    state = ConversationState(conversation_id=CONVERSATION)
    blocks: list[CompositionBlock] = []
    for turn, size in enumerate(sizes):
        cards = []
        for index in range(size):
            component_id = f"cmp_{turn:04d}_{index:053d}"
            state.components[component_id] = _entry(
                title, summary, revision=9_999_999_999, shows=shows
            )
            cards.append({"component_id": component_id, "action": "created"})
            if turn not in off:
                blocks.append(
                    CompositionBlock(
                        component_id=component_id,
                        revision=1,
                        type=ComponentType.LINE_CHART,
                        title="block",
                        pinned=turn == 0,
                    )
                )
        if turn == len(sizes) - 1:
            state.turn_components = cards
        else:
            state.turns.append({"user": "u", "assistant": "a", "cards": cards})
    state.composition = _next_manifest(None, blocks, actor=CompositionActor.MODEL)
    return state


def test_the_canvas_listing_stays_under_its_maximum_with_every_field_at_its_cap():
    listing = _largest_canvas().canvas_listing_state()

    assert len(listing["blocks"]) == studio_state._DIGEST_BLOCK_CAP
    assert len(listing["titles"]) == 3 * studio_state._LISTING_COMPACT_CAP
    assert len(listing["off_canvas"]) == studio_state._LISTING_COMPACT_CAP
    assert len(listing["counted"]) == studio_state._EARLIER_TURN_CAP
    assert "counted_rest" in listing
    size = _serialised({"part": "canvas", **listing})
    assert size <= CANVAS_LISTING_MAX_BYTES
    assert size > CANVAS_LISTING_MAX_BYTES * 0.9  # the maximum is within reach


def test_the_canvas_summary_stays_under_its_maximum_with_every_field_at_its_cap():
    state = _largest_canvas()

    summary = state.canvas_summary_state(focus=next(iter(state.components)))

    assert summary is not None
    size = _serialised(summary)
    assert size <= CANVAS_SUMMARY_MAX_BYTES
    assert size > CANVAS_SUMMARY_MAX_BYTES * 0.9


def test_one_turns_cards_fit_the_places_a_view_names_them_in():
    """Every card a turn can draw is named, in full or compactly."""
    most = BUDGETS["make_charts"] * MAX_CHARTS_PER_CALL

    assert most <= studio_state._DIGEST_BLOCK_CAP + studio_state._LISTING_COMPACT_CAP


def test_the_canvas_listing_stays_under_its_maximum_however_large_the_inputs():
    entries = {
        f"cmp_{index:016d}": _entry(
            ASTRAL * BIG,
            {"series": ["x" * BIG]},
            shows=_shows(ASTRAL * BIG, ASTRAL * BIG, count=3 * studio_state._DIGEST_SHOW_CAP),
        )
        for index in range(3 * studio_state._DIGEST_BLOCK_CAP)
    }
    state = _canvas(entries)

    listing = state.canvas_listing_state()
    summary = state.canvas_summary_state(focus=next(iter(entries)))

    assert len(listing["blocks"]) == studio_state._DIGEST_BLOCK_CAP
    assert _serialised(listing) <= CANVAS_LISTING_MAX_BYTES
    assert _serialised(summary) <= CANVAS_SUMMARY_MAX_BYTES


def test_a_card_taken_off_the_canvas_this_turn_is_named_off_it_and_says_so():
    """A card this turn drew and then lost from the canvas is still named,
    after the cards on the canvas and flagged, so the model does not read it
    as on the canvas."""
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary()), "cmp_02": _entry("Debt", None)})
    assert state.composition is not None
    state.composition = _next_manifest(
        state.composition, state.composition.blocks[:1], actor=CompositionActor.USER
    )

    listing = state.canvas_listing_state()

    assert [block["component_id"] for block in listing["blocks"]] == ["cmp_01"]
    assert "on_canvas" not in listing["blocks"][0]
    assert listing["off_canvas"] == [
        {
            "component_id": "cmp_02",
            "type": "line_chart",
            "title": "Debt",
            "on_canvas": False,
            "turn": 1,
        }
    ]


def test_the_listing_marks_a_pinned_block():
    state = _canvas({"cmp_01": _entry("Aid", None), "cmp_02": _entry("Debt", None)})
    assert state.composition is not None
    state.composition.blocks[1].pinned = True

    listing = state.canvas_listing_state()

    assert "pinned" not in listing["blocks"][0]
    assert listing["blocks"][1]["pinned"] is True


def test_this_turns_cards_past_the_full_places_are_named_compactly():
    cap = studio_state._DIGEST_BLOCK_CAP
    state = _canvas({f"cmp_{index:02d}": _entry("Aid", None) for index in range(cap + 3)})

    listing = state.canvas_listing_state()

    assert [block["component_id"] for block in listing["blocks"]] == [
        f"cmp_{index:02d}" for index in range(cap)
    ]
    assert listing["titles"] == [
        {
            "component_id": f"cmp_{index:02d}",
            "type": "line_chart",
            "title": "Aid",
            "turn": 1,
        }
        for index in range(cap, cap + 3)
    ]
    assert "titles" not in _canvas({"cmp_01": _entry("Aid", None)}).canvas_listing_state()


def _turns_of_cards(
    *groups: tuple[int, bool, bool],
    order: Callable[[list[str]], list[str]] = list,
) -> ConversationState:
    """One turn per group, oldest first, the last the open turn: each group is
    `(cards, pinned, on_canvas)`. Card ids are `cmp_<turn>_<n>`, and the
    composition holds the on-canvas cards in `order`."""
    state = ConversationState(conversation_id=CONVERSATION)
    held: list[str] = []
    pinned: set[str] = set()
    for turn, (count, pin, on) in enumerate(groups, start=1):
        ids = [f"cmp_{turn}_{n:02d}" for n in range(count)]
        for cid in ids:
            state.components[cid] = _entry(cid, None)
        if on:
            held += ids
        if pin:
            pinned.update(ids)
        cards = [{"component_id": cid} for cid in ids]
        if turn == len(groups):
            state.turn_components = cards
        else:
            state.turns.append({"user": "u", "assistant": "a", "cards": cards})
    state.composition = _next_manifest(
        None,
        [
            CompositionBlock(
                component_id=cid,
                revision=1,
                type=ComponentType.LINE_CHART,
                title="block",
                pinned=cid in pinned,
            )
            for cid in order(held)
        ],
        actor=CompositionActor.MODEL,
    )
    return state


def _named_ids(view: dict[str, Any]) -> list[str]:
    return [entry["component_id"] for key in ("blocks", "titles") for entry in view.get(key, [])]


def test_every_card_drawn_this_turn_and_every_pinned_card_is_named():
    """With forty cards drawn this turn and twenty pinned ones, the listing
    names each, in full or compactly, then the earlier cards still on the canvas."""
    state = _turns_of_cards((20, True, True), (2, False, True), (40, False, True))
    drawn = [f"cmp_3_{n:02d}" for n in range(40)]
    pinned = [f"cmp_1_{n:02d}" for n in range(20)]

    listing = state.canvas_listing_state()

    assert _named_ids(listing) == [*drawn, *pinned, "cmp_2_00", "cmp_2_01"]
    assert [block["component_id"] for block in listing["blocks"]] == drawn[
        : studio_state._DIGEST_BLOCK_CAP
    ]
    assert all(entry["pinned"] for entry in listing["titles"] if entry["turn"] == 1)
    assert "counted" not in listing


def test_a_follow_up_turn_lists_the_earlier_cards_on_the_canvas_in_full():
    """A new turn has drawn nothing yet, so the cards still on the canvas
    from earlier turns take the full places, newest turn first."""
    state = _turns_of_cards((1, False, True), (3, False, True), (0, False, True))

    listing = state.canvas_listing_state()

    assert [block["component_id"] for block in listing["blocks"]] == [
        "cmp_2_00",
        "cmp_2_01",
        "cmp_2_02",
        "cmp_1_00",
    ]
    assert all("fact_summary" in block for block in listing["blocks"])


def test_a_card_left_on_the_canvas_is_named_ahead_of_removed_cards():
    """Turn 2's card stays on the canvas while turn 3's forty cards were taken
    off. The next turn names the card in full, so `remove_cards` can take it
    by id, and the removed ones after it."""
    state = _turns_of_cards(
        (1, False, True), (1, False, True), (40, False, False), (0, False, True)
    )

    listing = state.canvas_listing_state()

    assert [block["component_id"] for block in listing["blocks"]] == ["cmp_2_00", "cmp_1_00"]
    assert [entry["component_id"] for entry in listing["off_canvas"]] == [
        f"cmp_3_{n:02d}" for n in range(40)
    ]
    assert "counted" not in listing


@pytest.mark.parametrize("order", ["reversed", "pinned-last", "interleaved"])
def test_the_canvas_views_are_the_same_however_the_manifest_orders_the_cards(order: str):
    """Studio places cards on a free board and keeps drag positions and its
    own text cards on the client, so the manifest's block order is not the
    order the user sees. The views rank by the turn that drew each card and
    state no place on the canvas: the same cards in any manifest order give
    the same views."""
    reorder: dict[str, Callable[[list[str]], list[str]]] = {
        "reversed": lambda ids: ids[::-1],
        "pinned-last": lambda ids: sorted(ids, key=lambda cid: cid.startswith("cmp_1_")),
        "interleaved": lambda ids: ids[::2] + ids[1::2],
    }
    groups = ((30, True, True), (20, False, True), (5, False, False), (20, False, True))
    plain = _turns_of_cards(*groups)
    moved = _turns_of_cards(*groups, order=reorder[order])

    assert moved.canvas_listing_state() == plain.canvas_listing_state()
    assert moved.canvas_summary_state() == plain.canvas_summary_state()


def test_pinned_cards_past_the_compact_cap_are_counted():
    full, compact = studio_state._DIGEST_BLOCK_CAP, studio_state._LISTING_COMPACT_CAP
    state = _turns_of_cards((full + compact + 5, True, True), (0, False, True))

    listing = state.canvas_listing_state()

    assert len(listing["blocks"]) == full
    assert all(block["pinned"] for block in listing["blocks"])
    assert len(listing["titles"]) == compact
    assert listing["counted"] == [{"turn": 1, "on_canvas": 5}]


def test_earlier_cards_past_the_caps_are_counted_by_turn():
    """Cards still on the canvas past the compact cap, and the turns of cards
    off it past the one that fills the off-canvas places, are counted by the
    turn that drew them, the newest turns apart and the rest in one count."""
    full, compact = studio_state._DIGEST_BLOCK_CAP, studio_state._LISTING_COMPACT_CAP
    cap = studio_state._EARLIER_TURN_CAP
    state = _turns_of_cards(
        *[(1, False, False)] * (cap + 2),
        (compact, False, False),
        (full + compact + 3, False, True),
        (0, False, True),
    )
    state.components["cmp_unrecorded"] = _entry("Aid", None)

    listing = state.canvas_listing_state()

    assert len(_named_ids(listing)) == full + compact
    assert {entry["turn"] for entry in listing["off_canvas"]} == {cap + 3}
    assert len(listing["off_canvas"]) == compact
    assert listing["counted"][0] == {"turn": cap + 4, "on_canvas": 3}
    assert [count["turn"] for count in listing["counted"]] == [
        cap + 4,
        *range(cap + 2, 3, -1),
    ]
    assert listing["counted_rest"] == {"off_canvas": 4}


def test_cards_this_turn_took_off_rank_behind_the_pinned_ones():
    state = _turns_of_cards((3, True, True), (20, False, False))

    listing = state.canvas_listing_state()

    assert [block["component_id"] for block in listing["blocks"]] == [
        f"cmp_1_{n:02d}" for n in range(3)
    ]
    assert [entry["component_id"] for entry in listing["off_canvas"]] == [
        f"cmp_2_{n:02d}" for n in range(20)
    ]


def test_turns_of_cards_off_the_canvas_are_named_whole_or_counted():
    """The off-canvas cards are named a whole turn at a time, newest first,
    until a turn does not fit. That turn and the older turns are counted."""
    compact = studio_state._LISTING_COMPACT_CAP
    state = _turns_of_cards(
        (5, False, False), (compact - 3, False, False), (2, False, False), (0, False, True)
    )

    listing = state.canvas_listing_state()

    assert [entry["turn"] for entry in listing["off_canvas"]] == [3, 3, *[2] * (compact - 3)]
    assert listing["counted"] == [{"turn": 1, "off_canvas": 5}]


def test_recall_without_a_component_lists_the_canvas():
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary()), "cmp_02": _entry("Debt", None)})
    assert state.composition is not None
    state.components["cmp_old"] = _entry("Old", _fact_summary())
    state.turns = [{"user": "u", "assistant": "a", "cards": [{"component_id": "cmp_old"}]}]
    state.composition.blocks[0].pinned = True

    listing = state.recall_part(part="canvas")

    assert listing["part"] == "canvas"
    assert [card["component_id"] for card in listing["blocks"]] == ["cmp_01", "cmp_02"]
    assert listing["blocks"][0]["pinned"] is True
    assert listing["blocks"][0]["turn"] == 2
    assert [card["component_id"] for card in listing["off_canvas"]] == ["cmp_old"]


def test_the_orchestrators_canvas_summary_counts_the_cards_and_names_none():
    """The always-on summary holds counts only. The listing is a recall away."""
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary()), "cmp_02": _entry("Debt", None)})
    state.components["cmp_old"] = _entry("Old", None)

    assert state.canvas_summary_state() == {"cards_on_canvas": 2, "cards_hidden": 1}
    assert ConversationState(conversation_id=CONVERSATION).canvas_summary_state() is None


def test_the_canvas_summary_keeps_a_focus_card_in_full():
    """#348: a card's "Ask follow up" names the card and carries it whole."""
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary()), "cmp_02": _entry("Debt", None)})

    summary = state.canvas_summary_state(focus="cmp_01")

    assert summary is not None
    assert summary["focus_component_id"] == "cmp_01"
    assert [block["component_id"] for block in summary["blocks"]] == ["cmp_01"]
    assert summary["blocks"][0]["fact_summary"] is not None


def test_recall_says_whether_a_component_is_still_on_the_canvas():
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.components["cmp_off"] = _entry("Removed", None)

    assert state.recall_part(part="canvas", component_id="cmp_01")["on_canvas"] is True
    assert state.recall_part(part="canvas", component_id="cmp_off")["on_canvas"] is False


def test_removing_cards_takes_the_named_blocks_off_and_reports_the_rest():
    state = _canvas({f"cmp_0{n}": _entry("Aid", None) for n in (1, 2, 3)})
    before = state.composition

    removal = remove_cards(state, ["cmp_02", "cmp_99"], all_cards=False)

    assert removal[:3] == (["cmp_02"], ["cmp_99"], [])
    assert state.composition is not None and before is not None
    assert [b.component_id for b in state.composition.blocks] == ["cmp_01", "cmp_03"]
    assert state.composition.revision == before.revision + 1
    assert state.composition.actor == CompositionActor.MODEL
    # The stream owes one update for it, once.
    assert state.take_unreported_removal() is True
    assert state.take_unreported_removal() is False


@pytest.mark.parametrize(
    ("named", "all_cards"),
    [(["cmp_01", "cmp_02"], False), ([], True)],
    ids=["named", "all-cards"],
)
def test_removing_cards_leaves_pinned_ones_and_names_them(named: list[str], all_cards: bool):
    """Only the user unpins a card, so a pinned block stays whether the call
    names it or takes every card, and the call says which stayed."""
    state = _canvas({f"cmp_0{n}": _entry("Aid", None) for n in (1, 2, 3)})
    assert state.composition is not None
    state.composition.blocks[0].pinned = True

    removal = remove_cards(state, named, all_cards=all_cards)

    assert removal.kept_pinned == ["cmp_01"]
    assert "cmp_01" not in removal.removed
    assert state.composition.blocks[0].component_id == "cmp_01"
    assert state.composition.blocks[0].pinned is True


def test_a_call_naming_only_pinned_cards_changes_nothing():
    state = _canvas({"cmp_01": _entry("Aid", None)})
    assert state.composition is not None
    state.composition.blocks[0].pinned = True
    before = state.composition.revision

    assert remove_cards(state, ["cmp_01"], all_cards=False)[:3] == ([], [], ["cmp_01"])
    assert state.composition.revision == before
    assert state.take_unreported_removal() is False


def test_a_card_already_off_the_canvas_is_not_reported_missing():
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.components["cmp_hidden"] = _entry("Hidden", None)

    assert remove_cards(state, ["cmp_hidden"], all_cards=False)[:3] == ([], [], [])
    assert state.take_unreported_removal() is False


def test_removing_every_card_leaves_an_empty_canvas_the_digest_states():
    state = _canvas({f"cmp_0{n}": _entry("Aid", None) for n in (1, 2)})

    removal = remove_cards(state, [], all_cards=True)

    assert removal[:3] == (["cmp_01", "cmp_02"], [], [])
    listing = state.canvas_listing_state()
    assert all(block["on_canvas"] is False for block in listing["blocks"])
    assert remove_cards(state, [], all_cards=True) == ([], [], [], {})


def test_a_focus_that_names_no_stored_component_stays_out_of_the_summary():
    """`focus` is the client's `component_ref.component_id`, an unbounded string."""
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary())})

    summary = state.canvas_summary_state(focus="F" * 1_000_000)

    assert summary == {"cards_on_canvas": 1, "cards_hidden": 0}


def test_a_focus_card_off_the_canvas_comes_whole_and_says_it_is_hidden():
    """#348 keeps the focus card whatever its state. A hidden one stays
    hidden, so the model does not read it as on the user's canvas."""
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.components["cmp_off"] = _entry("Off the canvas", None)

    summary = state.canvas_summary_state(focus="cmp_off")

    assert summary is not None
    assert summary["focus_component_id"] == "cmp_off"
    assert [block["component_id"] for block in summary["blocks"]] == ["cmp_off"]
    assert summary["blocks"][0]["on_canvas"] is False


def test_a_component_id_a_type_and_a_title_of_any_size_are_cut_to_their_caps():
    """The digest's maximum holds for every field of a block.

    The id and the type come from the platform and the title from an edit the
    model wrote (`studio_edit._record_revision` stores `patch["title"]` as given),
    so none of the three can be assumed short.
    """
    huge_id = "cmp_" + "i" * BIG
    state = _canvas({"cmp_01": _entry("Aid", None), "cmp_02": _entry("Debt", None)})
    state.components[huge_id] = state.components.pop("cmp_01")
    state.components[huge_id]["type"] = "t" * BIG
    state.components["cmp_02"]["title"] = {"text": "t" * BIG}
    state.composition = _next_manifest(
        None,
        [
            CompositionBlock(
                component_id=component_id,
                revision=1,
                type=ComponentType.LINE_CHART,
                title="block",
            )
            for component_id in (huge_id, "cmp_02")
        ],
        actor=CompositionActor.MODEL,
    )
    state.turn_components = [{"component_id": huge_id}, {"component_id": "cmp_02"}]

    digest = state.canvas_listing_state()

    assert digest is not None
    assert _serialised(digest) <= CANVAS_LISTING_MAX_BYTES
    first, second = digest["blocks"]
    assert _serialised(first["component_id"]) <= studio_state._BLOCK_ID_BYTES
    assert first["component_id"].endswith(studio_state._CUT_MARKER)
    assert _serialised(first["type"]) <= studio_state._BLOCK_TYPE_BYTES
    assert _serialised(second["title"]) <= studio_state._TITLE_BYTE_CAP
    assert state.components[huge_id]["type"] == "t" * BIG  # the digest cuts a copy


def test_every_component_type_fits_the_type_cap_so_a_real_type_is_never_cut():
    assert all(_serialised(kind.value) <= studio_state._BLOCK_TYPE_BYTES for kind in ComponentType)


def test_recall_serves_the_uncut_canvas_entries():
    title, summary = "t" * BIG, {"series": ["x" * BIG]}
    state = _canvas({"cmp_01": _entry(title, summary)})

    served = state.recall_part(part="canvas", component_id="cmp_01")

    assert served == {
        "part": "canvas",
        "component_id": "cmp_01",
        "on_canvas": True,
        **_entry(title, summary),
    }
    assert state.components["cmp_01"] == _entry(title, summary)  # the digest cuts a copy


_PLAN = _top_donors()[1]
_PLAN_JSON = _PLAN.model_dump(mode="json")
_READING_JSON = SearchReading.from_part(
    {"variable": "aid to Guatemala", "status": "definite", "spec_id": "spec_" + "a" * 32}
).model_dump(mode="json")


def test_a_closed_turn_keeps_its_plans_and_readings_and_reads_each_unsearched_source():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_user_message = "top donors to guatemala since 2020"
    state.turn_plans = [_PLAN_JSON]
    state.turn_readings = [_READING_JSON]
    used = _PLAN.sources[0].spec_id
    state.turn_specs = {used: {"spec_id": used}, _READING_JSON["spec_id"]: {"spec_id": "a"}}

    state.close_turn(final_text="The top donor was the United States.")

    (record,) = state.turns
    assert record["plans"] == [_PLAN_JSON]
    # Only the block of a source the turn's plan used is kept.
    assert record["specs"] == {used: {"spec_id": used}}
    searched, translated = (SearchReading.model_validate(r) for r in record["readings"])
    assert searched == SearchReading.model_validate(_READING_JSON)
    assert (translated.spec_id, translated.origin) == (_PLAN.sources[0].spec_id, "translator")
    assert translated.window is not None and (translated.window.start, translated.window.end) == (
        2020,
        2023,
    )


def test_a_closed_turn_keeps_no_block_for_a_spec_id_only_offered_as_a_candidate():
    candidate = "spec_" + "d" * 32
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_readings = [
        SearchReading.model_validate(
            {
                "text": "aid",
                "status": "candidates",
                "candidates": [{"spec_id": candidate, "label": "Aid"}],
            }
        ).model_dump(mode="json")
    ]
    state.turn_specs = {candidate: {"spec_id": candidate}}

    state.close_turn(final_text="Which aid measure?")

    (record,) = state.turns
    assert "specs" not in record
    assert state.prior_ledger_state()["spec_ids"] == []
    assert state.earlier_block(candidate) is None


# The definitions apis served for Guatemala's nominal GDP over every year
# (turn 0's orchestrator search) and over 1960 to 2024 (turn 2's translator
# discovery) in a recorded Studio session. They differ only in
# `requested_window`.
_GDP_ALL_YEARS, _GDP_1960_2024 = json.loads(
    (Path(__file__).parent / "fixtures" / "gdp_guatemala_definitions.json").read_text()
)
_GDP_STAT_VAR = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
# Recorded `search` spec blocks: Kenya's population serves one StatVar,
# German ODA to Kenya is a family.
_SPEC_BLOCKS = json.loads(
    (Path(__file__).parent / "fixtures" / "family_spec_blocks.json").read_text()
)
_ONE_STAT_VAR_BLOCK = _SPEC_BLOCKS["population_kenya"]["parts"][0]["spec"]
_FAMILY_BLOCK = _SPEC_BLOCKS["germany_oda_to_kenya"]["parts"][0]["spec"]


@pytest.mark.parametrize("block", [_ONE_STAT_VAR_BLOCK, _FAMILY_BLOCK], ids=["one", "family"])
def test_a_source_a_turn_only_fetched_keeps_its_identity_and_nothing_else_of_its_block(block):
    spec_id = block["spec_id"]
    identity = block["definition"]["identity"]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_readings = [
        SearchReading.from_part(
            {"variable": "x", "status": "definite", "spec_id": spec_id}
        ).model_dump(mode="json")
    ]
    state.turn_specs = {spec_id: block}

    assert state.recorded_identity(spec_id) == RecordedIdentity(identity=identity)
    state.close_turn(final_text="Done.")
    state.begin_turn()

    (record,) = state.turns
    assert "specs" not in record
    # The bound: the identity, without the block around it.
    assert record["identities"] == {spec_id: {"identity": identity}}
    assert state.recorded_identity(spec_id) == RecordedIdentity(identity=identity)
    assert state.earlier_block(spec_id) is None


def _plan_json_retrieving(source: SelectedSource) -> dict[str, Any]:
    """The record of `_PLAN`'s plan retrieving `source` in place of its own."""
    plan = _top_donors()[0]
    assert plan.workflow is not None
    (binding,) = plan.workflow.evidence_bindings
    workflow = plan.workflow.model_copy(
        update={
            "sources": (source,),
            "evidence_bindings": (binding.model_copy(update={"spec_id": source.spec_id}),),
        }
    )
    record = PlanRecord.from_plan(plan.model_copy(update={"workflow": workflow}))
    return record.model_dump(mode="json")


def _gdp_plan_json() -> dict[str, Any]:
    """`_PLAN`'s plan retrieving instead the 1960 to 2024 GDP definition its
    translator found, which serves one StatVar, the family's root."""
    (source,) = _top_donors()[0].workflow.sources
    return _plan_json_retrieving(
        source.model_copy(
            update={
                "spec_id": _GDP_1960_2024["spec_id"],
                "identity": _GDP_1960_2024["identity"],
                "served_count": 1,
                "stat_var_reading": None,
            }
        )
    )


@pytest.mark.parametrize("block", [_ONE_STAT_VAR_BLOCK, _FAMILY_BLOCK], ids=["one", "family"])
def test_a_search_block_and_a_plan_source_record_the_same_identity(block):
    spec_id = block["spec_id"]
    searched = ConversationState(conversation_id=CONVERSATION)
    searched.turn_specs = {spec_id: block}
    planned = ConversationState(conversation_id=CONVERSATION)
    planned.turn_plans = [_plan_json_retrieving(source_from_spec_block(block))]

    assert searched.recorded_identity(spec_id) == planned.recorded_identity(spec_id) is not None


@pytest.mark.parametrize("plan_json", [_PLAN_JSON, _gdp_plan_json()], ids=["family", "gdp"])
def test_a_source_only_the_translator_found_keeps_its_identity_into_the_next_turn(plan_json):
    (source,) = plan_json["sources"]
    assert source["origin"] == "translator" and source["identity"] is not None
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_plans = [plan_json]

    state.close_turn(final_text="Done.")
    state.begin_turn()

    assert state.recorded_identity(source["spec_id"]) == RecordedIdentity(
        identity=source["identity"]
    )
    assert state.earlier_block(source["spec_id"]) is None


def test_a_closed_turn_keeps_the_block_of_the_source_a_windowed_copy_copies():
    """A plan retrieving a copy of a searched source over another window keeps
    the searched source's block, so the next turn can pin or fetch it."""
    origin, copy_spec = _GDP_ALL_YEARS["spec_id"], _GDP_1960_2024["spec_id"]
    origin_block = {"spec_id": origin, "definition": {"identity": _GDP_ALL_YEARS["identity"]}}
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_plans = [_gdp_plan_json()]
    state.turn_readings = [
        SearchReading.from_part(
            {"variable": "Guatemala's GDP", "status": "definite", "spec_id": origin}
        ).model_dump(mode="json")
    ]
    state.turn_specs = {
        origin: origin_block,
        copy_spec: {"spec_id": copy_spec, "definition": {"identity": _GDP_1960_2024["identity"]}},
    }

    state.close_turn(final_text="Done.")
    state.begin_turn()

    (record,) = state.turns
    assert set(record["specs"]) == {origin, copy_spec}
    assert origin in state.prior_ledger_state()["spec_ids"]
    assert state.earlier_block(origin) == origin_block


@pytest.mark.parametrize(
    "identity",
    [
        _GDP_1960_2024["identity"],  # hashes to another spec_id
        {**_ONE_STAT_VAR_BLOCK["definition"]["identity"], "padding": "x" * 70_000},  # past 64 KiB
    ],
    ids=["another_spec_id", "past_the_cap"],
)
def test_an_identity_resolve_definition_would_refuse_is_not_recorded(identity):
    block = _ONE_STAT_VAR_BLOCK
    spec_id = block["spec_id"]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_specs = {
        spec_id: {**block, "definition": {**block["definition"], "identity": identity}}
    }

    state.close_turn(final_text="Done.")

    assert "identities" not in state.turns[0]
    assert state.recorded_identity(spec_id) is None


def test_begin_turn_clears_the_previous_turns_plans_and_readings():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_plans = [_PLAN_JSON]
    state.turn_readings = [_READING_JSON]
    state.turn_specs = {"spec_a": {}}

    state.begin_turn()

    assert (state.turn_plans, state.turn_readings, state.turn_specs) == ([], [], {})


def test_begin_turn_and_a_new_plan_lineage_reset_the_lineage_fields_together():
    state = ConversationState(conversation_id=CONVERSATION)

    for reset in (state.begin_turn, state.start_plan_lineage):
        state.plan_version_ids = {1: "plan_a", 2: "plan_b"}
        state.latest_plan_version = 2
        state.turn_choice_baseline = {"aid:sector": "health"}

        reset()

        assert (state.plan_version_ids, state.latest_plan_version, state.turn_choice_baseline) == (
            {},
            None,
            {},
        )


_LONG_ANSWER = (
    "The United States gave the most, followed by the European Union institutions. "
    "Germany came third and Japan fourth. "
    + "Every other donor, "
    + "small and large, " * 12
    + "gave less than a tenth of that. "
    + "Would you like the same ranking for Honduras?"
)


def test_the_digest_keeps_asks_and_the_previous_answer_whole_and_states_each_plan():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {
            "user": "top donors to guatemala since 2020",
            "assistant": _LONG_ANSWER,
            "plans": [_PLAN_JSON],
            "readings": [_READING_JSON],
        },
        {
            "user": "why is the data through 2023 when 2024 is the latest year?",
            "assistant": _LONG_ANSWER,
        },
    ]

    digest = state.conversation_digest_state()

    assert digest is not None
    older, previous = digest["turns"]
    assert older["user"] == "top donors to guatemala since 2020"
    assert older["plans"] == [_PLAN.digest_line()]
    assert "readings" not in older
    # The older answer keeps its leading sentences, cut where one ends.
    assert older["assistant"] == (
        "The United States gave the most, followed by the European Union institutions. "
        "Germany came third and Japan fourth." + studio_state._CUT_MARKER
    )
    assert previous["assistant"] == _LONG_ANSWER
    assert len(json.dumps(digest)) <= CONVERSATION_DIGEST_MAX_BYTES


# A plan record loads past keys it does not declare, so its malformed one has a
# status this release does not know.
_MALFORMED_PLAN = {"status": "superseded"}


def test_the_digest_skips_a_stored_plan_record_that_no_longer_validates(
    caplog: pytest.LogCaptureFixture,
):
    """A record whose status this release does not know is skipped; one
    carrying a key a later release added loads."""
    state = ConversationState(conversation_id=CONVERSATION)
    malformed = {**_PLAN_JSON, **_MALFORMED_PLAN}
    later = {**_PLAN_JSON, "from_a_later_release": 1}
    state.turns = [
        {"user": "q1", "assistant": "a1", "plans": [malformed, later, _PLAN_JSON]},
        {"user": "q2", "assistant": "a2"},
    ]

    with caplog.at_level(logging.WARNING, logger="harness.studio_state"):
        digest = state.conversation_digest_state()

    assert digest is not None
    assert digest["turns"][0]["plans"] == [_PLAN.digest_line()] * 2
    assert "skipping a stored plan record" in caplog.text


_MALFORMED = {"unknown_field": 1}


@pytest.mark.parametrize(
    "read",
    [
        lambda state: state.recall_part(part="plan", turn=1)["text"],
        lambda state: state.recall_part(part="readings", turn=1)["text"],
        lambda state: state.prior_ledger_state()["readings"],
        lambda state: state.close_turn(final_text="a2") or state.turns[-1]["readings"],
    ],
    ids=["recall_plan", "recall_readings", "prior_ledger", "close_turn"],
)
def test_every_reader_of_stored_turn_records_skips_one_that_no_longer_validates(
    read, caplog: pytest.LogCaptureFixture
):
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {
            "user": "q1",
            "assistant": "a1",
            "plans": [{**_PLAN_JSON, **_MALFORMED_PLAN}, _PLAN_JSON],
            "readings": [{**_READING_JSON, **_MALFORMED}, _READING_JSON],
        }
    ]
    state.turn_plans = [{**_PLAN_JSON, **_MALFORMED_PLAN}, _PLAN_JSON]
    state.turn_readings = [_READING_JSON]

    with caplog.at_level(logging.WARNING, logger="harness.studio_state"):
        served = read(state)

    assert served
    assert "skipping a stored" in caplog.text


def test_the_prior_ledger_keeps_each_spec_ids_newest_reading_and_its_turns_block():
    newer = {**_READING_JSON, "sentence": "newer"}
    latest = SearchReading(
        text="latest aid",
        status="definite",
        spec_id="spec_" + "c" * 32,
        requested_window_kind="latest",
    ).model_dump(mode="json")
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {
            "user": "q1",
            "assistant": "a1",
            "readings": [_READING_JSON, latest],
            "specs": {_READING_JSON["spec_id"]: {"turn": 1}, latest["spec_id"]: {"turn": 1}},
        },
        {"user": "q2", "assistant": "a2"},
        {
            "user": "q3",
            "assistant": "a3",
            "readings": [newer],
            "specs": {_READING_JSON["spec_id"]: {"turn": 3}},
        },
    ]

    ledger = state.prior_ledger_state()

    assert ledger is not None
    assert ledger["readings"] == [latest, newer]
    assert ledger["spec_ids"] == [_READING_JSON["spec_id"]]
    assert set(ledger["warnings"]) == {_READING_JSON["spec_id"], latest["spec_id"]}
    # The ledger a turn's session is seeded with carries no block; a pin or
    # a fetch reads the newest one through `earlier_block`.
    assert set(ledger) == {"readings", "warnings", "spec_ids"}
    assert state.earlier_block(_READING_JSON["spec_id"]) == {"turn": 3}
    assert state.earlier_block(latest["spec_id"]) is None
    assert ConversationState(conversation_id=CONVERSATION).prior_ledger_state() is None


def test_a_ledger_turn_holding_keys_this_release_does_not_declare_still_loads():
    """A conversation's ledger holds turns stored when Search listed each source's
    members, where every plan source and recorded identity carries `stat_vars`,
    and turns a later release wrote with fields this one lacks. The turn still
    reads as its plan and its identities, the extra keys left behind."""
    plan = copy.deepcopy(_PLAN_JSON)
    for source in plan["sources"]:
        source["stat_vars"] = [] if source["members"] else [_GDP_STAT_VAR]
    assert plan["decisions"]
    for decision in plan["decisions"]:
        decision["from_a_later_release"] = True
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {
            "user": "q",
            "assistant": "a",
            "plans": [plan],
            "identities": {
                _GDP_ALL_YEARS["spec_id"]: {
                    "identity": _GDP_ALL_YEARS["identity"],
                    "stat_vars": [_GDP_STAT_VAR],
                }
            },
        }
    ]

    assert state.recall_part(part="plan", turn=1)["text"] == [
        _PLAN.render(budget=studio_state._RECALL_TEXT_BUDGET)
    ]
    assert state.recorded_identity(_GDP_ALL_YEARS["spec_id"]) == RecordedIdentity(
        identity=_GDP_ALL_YEARS["identity"]
    )


def _recallable() -> ConversationState:
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {"user": "q1", "assistant": "a1", "plans": [_PLAN_JSON], "readings": [_READING_JSON]},
        {"user": "q2", "assistant": "a2"},
        {"user": "q3", "assistant": "a3", "plans": [_PLAN_JSON]},
    ]
    state.components = {"cmp_01": _entry("Aid", {"series": ["x"]})}
    return state


def test_recall_part_returns_a_turns_plan_record_in_full_and_defaults_to_the_newest_plan():
    state = _recallable()

    first = state.recall_part(part="plan", turn=1)
    newest = state.recall_part(part="plan")

    assert first["turn"] == 1 and first["user"] == "q1"
    assert first["text"] == [_PLAN.render(budget=studio_state._RECALL_TEXT_BUDGET)]
    assert newest["turn"] == 3


def test_recall_part_returns_a_turns_readings_and_a_canvas_entry():
    state = _recallable()

    readings = state.recall_part(part="readings")
    canvas = state.recall_part(part="canvas", component_id="cmp_01")

    assert readings["turn"] == 1 and "aid to Guatemala" in readings["text"][0]
    assert canvas == {
        "part": "canvas",
        "component_id": "cmp_01",
        "on_canvas": False,
        **_entry("Aid", {"series": ["x"]}),
    }


@pytest.mark.parametrize(
    ("target", "legal"),
    [
        ({"part": "plan", "turn": 2}, {"turns": [1, 3]}),
        ({"part": "plan", "turn": 9}, {"turns": [1, 3]}),
        ({"part": "readings", "turn": 3}, {"turns": [1]}),
        ({"part": "canvas", "component_id": "cmp_99"}, {"component_ids": ["cmp_01"]}),
    ],
)
def test_recall_part_refuses_a_missing_target_naming_the_legal_values(target, legal):
    with pytest.raises(studio_state.RecallRefusedError) as refused:
        _recallable().recall_part(**target)

    assert refused.value.legal == legal


def test_recall_part_bounds_what_it_renders_of_a_turn_with_several_plans():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [{"user": "q", "assistant": "a", "plans": [_PLAN_JSON] * 4}]

    text = state.recall_part(part="plan")["text"]

    assert len(text) == 4
    assert all(len(plan) <= studio_state._RECALL_TEXT_BUDGET // 4 for plan in text)


async def test_recall_reads_an_executing_plans_status_from_its_run(
    new_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan recorded executing states its run's status in the run ledger
    once the run has settled; one recorded with no run, or whose run is still
    unfinished, keeps its recorded status. The ledger is read only while a
    record awaits a run."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    async with store.fenced("principal", CONVERSATION, segment_id) as (conn, pk):
        await conn.execute(
            text(
                "INSERT INTO runs (conversation, run_id, record) "
                "VALUES (:pk, :run_id, CAST(:record AS jsonb))"
            ),
            [
                {"pk": pk, "run_id": "run_done", "record": '{"status": "completed"}'},
                {"pk": pk, "run_id": "run_held", "record": '{"status": "pending"}'},
            ],
        )
    executing = {**_PLAN_JSON, "status": PlanStatus.EXECUTING.value}
    plans = [
        {**executing, "run_id": "run_done"},
        {**executing, "run_id": "run_held"},
        executing,
        {**_PLAN_JSON, "status": PlanStatus.FAILED.value, "run_id": "run_done"},
    ]
    state.turns = [{"user": "q", "assistant": "a", "plans": plans}]
    token = current_conversation_key.set(("principal", CONVERSATION))
    try:
        recalled = await make_recall_part(store)(part="plan", turn=1, component_id=None)
    finally:
        current_conversation_key.reset(token)

    stated = [
        PlanStatus.COMPLETED,
        PlanStatus.EXECUTING,
        PlanStatus.EXECUTING,
        PlanStatus.FAILED,
    ]
    expected = [
        PlanRecord.model_validate(plan).model_copy(update={"status": status})
        for plan, status in zip(plans, stated, strict=True)
    ]
    assert recalled["text"] == [
        record.render(budget=studio_state._RECALL_TEXT_BUDGET // len(plans)) for record in expected
    ]

    async def unread(*args: Any) -> dict[str, Any]:
        raise AssertionError("the ledger is read with no record awaiting a run")

    monkeypatch.setattr(store, "_stored_statuses", unread)
    state.turns = [{"user": "q", "assistant": "a", "plans": [_PLAN_JSON, executing]}]
    assert await store.run_statuses("principal", CONVERSATION, conversation=state) == {}


def _turn(index: int, **fields: Any) -> dict[str, Any]:
    return {"user": f"question {index}", "assistant": f"answer {index}", **fields}


def test_realistic_conversation_turns_pass_through_byte_for_byte():
    cards = [
        {"component_id": "cmp_01j9zc3aqw", "type": "line_chart", "title": "Aid to Kenya"},
        {
            "component_id": "cmp_01j9zc3aqx",
            "type": "stat_card",
            "title": "Aid to Uganda",
            "action": "revised",
        },
    ]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [_turn(0, cards=cards), _turn(1), _turn(2, error="cancelled")]

    digest = state.conversation_digest_state()

    assert digest == {"turns": state.turns, "total_turns": 3}


def test_a_turn_with_a_huge_cards_list_stays_under_the_byte_cap():
    cards = [
        {"component_id": f"cmp_{index:016d}", "type": "line_chart", "title": "c" * 500}
        for index in range(BIG // 100)
    ]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [_turn(0, cards=cards)]

    digest = state.conversation_digest_state()

    assert digest is not None
    assert _serialised(digest) <= CONVERSATION_DIGEST_MAX_BYTES
    (turn,) = digest["turns"]
    assert len(turn["cards"]) == studio_state._HISTORY_CARD_CAP
    assert turn["cards_omitted"] == len(cards) - studio_state._HISTORY_CARD_CAP
    assert turn["cards"][0]["title"].endswith(studio_state._CUT_MARKER)
    assert turn["cards"][0]["component_id"] == "cmp_0000000000000000"


def test_an_oversized_turn_error_is_cut_and_says_so():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [_turn(0, error="e" * BIG)]

    digest = state.conversation_digest_state()

    assert digest is not None
    error = digest["turns"][0]["error"]
    assert error.endswith(studio_state._CUT_MARKER)
    assert _serialised(error) <= studio_state._HISTORY_ERROR_BYTE_CAP
    assert _serialised(digest) <= CONVERSATION_DIGEST_MAX_BYTES


def test_the_newest_turn_is_shrunk_to_fit_rather_than_dropped_or_left_over_the_cap():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {"user": "newest " + ASTRAL * BIG, "assistant": ASTRAL * BIG, "error": ASTRAL * BIG}
    ]

    digest = state.conversation_digest_state()

    assert digest is not None
    assert _serialised(digest) <= CONVERSATION_DIGEST_MAX_BYTES
    (turn,) = digest["turns"]
    assert turn["user"].startswith("newest ")
    assert 0 < len(turn["assistant"]) < studio_state._HISTORY_TEXT_CAP


def test_ten_maximal_turns_stay_under_the_byte_cap_and_keep_the_newest():
    cards = [
        {"component_id": f"cmp_{index:016d}", "type": "line_chart", "title": ASTRAL * BIG}
        for index in range(BIG // 100)
    ]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {
            "user": f"turn {index} " + ASTRAL * BIG,
            "assistant": ASTRAL * BIG,
            "cards": cards,
            "error": ASTRAL * BIG,
        }
        for index in range(studio_state._HISTORY_TURN_CAP)
    ]

    digest = state.conversation_digest_state()

    assert digest is not None
    assert digest["total_turns"] == studio_state._HISTORY_TURN_CAP
    assert _serialised(digest) <= CONVERSATION_DIGEST_MAX_BYTES
    assert digest["turns"][-1]["user"].startswith(f"turn {studio_state._HISTORY_TURN_CAP - 1} ")


def test_the_conversation_digest_holds_its_cap_for_any_size_of_a_single_id():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [_turn(0, cards=[{"component_id": "c" * BIG, "type": "line_chart"}])]

    digest = state.conversation_digest_state()

    assert digest is not None
    assert _serialised(digest) <= CONVERSATION_DIGEST_MAX_BYTES


# --- awaiting_answer / drop_resuming: a pending interrupt marked resuming ---
# --- reads as no longer awaiting a client answer -----------------------------


def _pending(*, resuming_segment: str | None = None) -> _PendingInterrupt:
    return _PendingInterrupt(adk_id="adk_1", kind="plan", resuming_segment=resuming_segment)


@pytest.mark.parametrize("new_plan", [True, False])
def test_a_stored_pending_card_keeps_new_plan_beside_the_card(new_plan: bool) -> None:
    """The member marks the emitted frame. An instance rolled back to a model
    without it cannot load a stored card that holds it, so it is stored beside the
    card and restored on load, and the card's own entry leaves it out."""
    raw = json.loads(
        (
            Path(__file__).parent / "wire_fixtures" / "studio" / "event-03-plan-proposed.json"
        ).read_text()
    )
    card = PlanProposed.model_validate({**raw, "new_plan": new_plan})
    pending = _PendingInterrupt(adk_id="adk_1", kind="plan", card=card)

    entry = studio_store._serialize_pending({"int_1": pending})["int_1"]
    restored = studio_store._deserialize_entry(json.loads(json.dumps(entry)))

    assert "new_plan" not in entry["card"]
    assert restored.card is not None
    assert bool(restored.card.new_plan) is new_plan


def test_a_stored_pending_card_keeps_each_decision_s_unit_beside_the_card_and_frames() -> None:
    """A unit row's offer marks the emitted frame. An instance rolled back to a
    model without it cannot load a stored card or frame that holds it, so the
    card and every frame are stored without it, the offers beside them, and
    restored on load."""
    raw = json.loads(
        (
            Path(__file__).parent / "wire_fixtures" / "studio" / "event-22-plan-proposed-unit.json"
        ).read_text()
    )
    card = PlanProposed.model_validate(raw)
    note = MessageComplete(event="message.complete", seq=8, text="Version 1 is still there.")
    pending = _PendingInterrupt(adk_id="adk_1", kind="plan", card=card, frames=(note, card))

    entry = studio_store._serialize_pending({"int_1": pending})["int_1"]
    restored = studio_store._deserialize_entry(json.loads(json.dumps(entry)))

    stored = [entry["card"], *entry["frames"]]
    assert all("unit" not in d for frame in stored for d in frame.get("decisions") or ())
    assert (restored.card, restored.frames) == (card, (note, card))


def test_a_stored_unit_question_is_kept_in_its_whole_units_form() -> None:
    """A unit question marks the emitted frame with the `unit` input and its offer. An
    instance rolled back to a model without them, or to one that adds `decisions` to
    each frame holding units, must still load the stored frame and answer the question.
    So the frame and the pending question are kept in the form a client that renders
    no unit question is asked it, a `single_select` over the offer's whole units, the
    offer under a key of its own, and both are restored on load."""
    raw = json.loads(
        (
            Path(__file__).parent / "wire_fixtures" / "studio" / "event-18-elicitation-unit.json"
        ).read_text()
    )
    asked = ElicitationRequest.model_validate(raw)
    note = MessageComplete(event="message.complete", seq=4, text="One question first.")
    offer = asked.questions[0].unit
    assert offer is not None
    question = dataclasses.replace(
        _UNIT_QUESTION, id="q1", unit=offer, options=dict(whole_unit_options(offer))
    )
    pending = _PendingInterrupt(
        adk_id="adk_1",
        kind="elicitation",
        frames=(note, asked),
        questions=(question,),
        skippable=True,
    )

    entry = studio_store._serialize_pending({"int_1": pending})["int_1"]
    restored = studio_store._deserialize_entry(json.loads(json.dumps(entry)))

    whole = list(whole_units(asked.questions[0].unit))
    (framed,) = [q for frame in entry["frames"] for q in frame.get("questions") or ()]
    assert "unit" not in framed
    assert framed["input"] == ElicitationInput.SINGLE_SELECT.value
    assert [option["id"] for option in framed["options"]] == whole
    assert framed["allow_free_text"] is False
    (kept,) = entry["questions"]
    assert (kept["input"], kept["native_input"]) == ("single_select", "unit")
    assert list(kept["options"]) == whole
    assert "frames_units" not in entry
    assert restored.frames == (note, asked)
    assert restored.questions == (question,)
    # A page sent the stored form, as a release with no unit question serves it,
    # answers with a pick, which reads as that whole unit.
    pick = OptionAnswer(kind="option", option_id=whole[-1])
    assert _answers_refusal(restored.questions, {"q1": pick}) is None
    unit = whole_units(asked.questions[0].unit)[whole[-1]]
    assert _resume_answer(restored.questions[0], pick) == {
        "kind": "unit",
        **unit.model_dump(mode="json", exclude_none=True),
    }


def test_awaiting_answer_excludes_an_entry_marked_resuming() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    state.pending["int_1"] = _pending(resuming_segment="seg_1")

    assert state.awaiting_answer is False


def test_awaiting_answer_is_true_for_an_entry_not_yet_resuming() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    state.pending["int_1"] = _pending()
    state.pending["int_2"] = _pending(resuming_segment="seg_1")  # a second, already resuming

    assert state.awaiting_answer is True


def test_drop_resuming_removes_only_entries_under_the_named_segment() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    state.pending["int_1"] = _pending(resuming_segment="seg_1")
    state.pending["int_2"] = _pending(resuming_segment="seg_2")
    state.pending["int_3"] = _pending()

    state.drop_resuming("seg_1")

    assert set(state.pending) == {"int_2", "int_3"}


# --- the ledger-backed store: create / load / save / close_segment ---------
# --- and the segment lease: claim_segment / renew / request_cancel ----------


@pytest.fixture
def new_store(pg_engine: AsyncEngine) -> Callable[..., StudioStore]:
    """A fresh `StudioStore` over the test's Postgres schema: each call
    stands for another instance. With `clock`, the
    store reads its leases' time from that clock instead of the database's."""

    def build(*, clock: StoreClock | None = None) -> StudioStore:
        return StudioStore(pg_engine, **(clock.store_kwargs if clock is not None else {}))

    return build


@pytest.fixture
async def clock(pg_engine: AsyncEngine) -> StoreClock:
    """A clock both a store's lease time and instance time read, at 1000."""
    return await StoreClock.start(pg_engine)


def _approval(**overrides: Any) -> ApprovalRecord:
    fields: dict[str, Any] = {
        "revision": "rev_1",
        "approver": "u_test",
        "decision": ApprovalDecision.APPROVED,
        "at": datetime(2026, 1, 1, tzinfo=UTC),
        "conversation_id": CONVERSATION,
        "interrupt_id": "int_1",
        "plan_version": 2,
        "auth_enforced": True,
    }
    fields.update(overrides)
    return ApprovalRecord(**fields)


async def test_the_ledger_stores_a_turns_blocks_only_when_the_turn_closes(
    new_store, pg_engine
) -> None:
    """A save after a search writes no spec block to the conversation's rows;
    closing the turn on another instance stores the block of the source its
    plan used, given the blocks as the studio reads them from the turn's
    cursor (`harness.turn.store.TurnStore.specs`)."""
    used = _PLAN.sources[0].spec_id
    candidate = "spec_" + "d" * 32
    blocks = {used: {"spec_id": used, "note": "used block"}, candidate: {"note": "unused block"}}
    store_a = new_store()
    state = await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store_a.save("principal", CONVERSATION)
    state.turn_specs = dict(blocks)
    state.turn_plans = [_PLAN_JSON]

    await store_a.save("principal", CONVERSATION)

    written = await _stored_rows_text(pg_engine)
    assert "used block" not in written and "unused block" not in written

    resumed = await new_store().load("principal", CONVERSATION)
    assert resumed is not None
    resumed.turn_specs = dict(blocks)
    resumed.close_turn(final_text="The top donor was the United States.")

    assert resumed.turns[-1]["specs"] == {used: blocks[used]}


async def _stored_rows_text(engine: Any) -> str:
    """Every row the conversation tables hold, as one JSON text."""
    tables = (
        "conversations",
        "conversation_turns",
        "conversation_components",
        "conversation_write_tokens",
        "conversation_approvals",
    )
    rows: list[Any] = []
    async with engine.connect() as conn:
        for table in tables:
            rows += [list(row) for row in await conn.execute(text(f"SELECT * FROM {table}"))]
    return json.dumps(rows, default=str)


async def test_a_readings_scope_choices_survive_the_turns_close_and_a_reload(
    new_store,
) -> None:
    """A later turn on another instance reads the scope choices a search stored,
    which its searches carry."""
    scoped = SearchReading.from_part(_scoped_part())
    store_a = new_store()
    state = await store_a.create("principal", CONVERSATION)
    state.begin_turn()
    state.turn_user_message = "health aid to Africa, CRS 120 alone"
    state.turn_readings = [scoped.model_dump(mode="json")]
    state.close_turn(final_text="Health aid to Africa was $1.2bn.")
    await store_a.save("principal", CONVERSATION)

    loaded = await new_store().load("principal", CONVERSATION)

    assert loaded is not None
    ledger = loaded.prior_ledger_state()
    assert ledger is not None
    (reading,) = (SearchReading.model_validate(stored) for stored in ledger["readings"])
    assert reading.scope == scoped.scope


async def test_round_trip_of_every_durable_field(new_store) -> None:
    """Every durable field survives a save on one store and a load on a
    SECOND store over the same ledger -- the actual cross-instance case,
    not just a read of the same in-process object."""
    store_a = new_store()
    state = await store_a.create("principal", CONVERSATION)
    state.turn_index = 3
    state.seq = 7
    state.trace_id = "tr_seed"
    state.auto_add = False
    state.client_features = frozenset({ClientFeature.MULTI_SELECT})
    state.turn_user_message = "how much aid did Kenya receive?"
    state.turn_final_text = "Kenya received $1.2B in 2023."
    state.turn_recorded = True
    state.turn_components = [{"component_id": "cmp_1", "type": "line_chart", "action": "created"}]
    state.turn_plans = [_PLAN_JSON]
    state.turn_readings = [_READING_JSON]
    state.timed_out_calls = {"get_data:call_a", "get_data:call_b"}
    state.plan_version_ids = {1: "plan_a", 2: "plan_b"}
    state.latest_plan_version = 2
    state.pending["int_1"] = _PendingInterrupt(
        adk_id="adk_1",
        kind="plan",
        plan_version=2,
        plan_revision="rev_1",
        questions=(
            _PendingQuestion(
                id="q1", question="Which count?", options={"spec_a": "Net"}, skippable=True
            ),
            _UNIT_QUESTION,
            _UNIT_AS_PICK,
        ),
        skippable=True,
        step_order=("step_b", "step_a"),
        step_labels={"step_b": "Fetch aid to Kenya", "step_a": "Total aid by year"},
        resuming_segment="seg_1",
        answer={"decision": "approve", "option_id": "spec_a"},
    )
    state.segment = Segment(id="seg_1", owner="owner_a", expires_at=1_700_000_045.0)
    state.components["cmp_1"] = {"type": "line_chart", "title": "Aid to Kenya", "revision": 1}
    # A record stored before turns held plans and readings, and one that holds them.
    state.turns.append({"user": "q1", "assistant": "a1"})
    state.turns.append({"user": "q2", "assistant": "a2", "plans": [_PLAN_JSON]})
    state.approvals.append(_approval())
    block = CompositionBlock(
        component_id="cmp_1", revision=1, type=ComponentType.LINE_CHART, title="Aid to Kenya"
    )
    await store_a.mutate_composition(
        "principal",
        CONVERSATION,
        lambda manifest, _components: _next_manifest(
            manifest, [block], actor=CompositionActor.MODEL
        ),
    )

    await store_a.save("principal", CONVERSATION)

    store_b = new_store()
    loaded = await store_b.load("principal", CONVERSATION)

    assert loaded is not None
    assert loaded.conversation_id == CONVERSATION
    assert loaded.turn_index == 3
    assert loaded.seq == 7
    assert loaded.trace_id == "tr_seed"
    assert loaded.auto_add is False
    assert loaded.client_features == {ClientFeature.MULTI_SELECT}
    assert loaded.turn_user_message == state.turn_user_message
    assert loaded.turn_final_text == state.turn_final_text
    assert loaded.turn_recorded is True
    assert loaded.turn_components == state.turn_components
    assert loaded.turn_plans == [_PLAN_JSON]
    assert loaded.turn_readings == [_READING_JSON]
    assert loaded.timed_out_calls == state.timed_out_calls
    assert loaded.plan_version_ids == {1: "plan_a", 2: "plan_b"}
    assert loaded.latest_plan_version == 2
    pending = loaded.pending["int_1"]
    assert pending.adk_id == "adk_1"
    assert pending.kind == "plan"
    assert pending.plan_version == 2
    assert pending.plan_revision == "rev_1"
    assert pending.questions == (
        _PendingQuestion(
            id="q1", question="Which count?", options={"spec_a": "Net"}, skippable=True
        ),
        _UNIT_QUESTION,
        _UNIT_AS_PICK,
    )
    assert (pending.skippable, pending.legacy) == (True, False)
    assert pending.step_order == ("step_b", "step_a")
    assert pending.step_labels == {"step_b": "Fetch aid to Kenya", "step_a": "Total aid by year"}
    assert pending.resuming_segment == "seg_1"
    assert pending.answer == {"decision": "approve", "option_id": "spec_a"}
    assert loaded.awaiting_answer is False  # its only entry is resuming
    assert loaded.segment == Segment(id="seg_1", owner="owner_a", expires_at=1_700_000_045.0)
    assert loaded.components == {
        "cmp_1": {"type": "line_chart", "title": "Aid to Kenya", "revision": 1}
    }
    assert loaded.turns == state.turns
    assert loaded.conversation_digest_state() == state.conversation_digest_state()
    assert len(loaded.approvals) == 1
    assert loaded.approvals[0].revision == "rev_1"
    assert loaded.approvals[0].decision == ApprovalDecision.APPROVED
    assert loaded.composition is not None
    assert loaded.composition.model_dump(mode="json") == state.composition.model_dump(mode="json")
    # process-local fields are never touched by the ledger
    assert loaded.lock is not state.lock
    assert loaded.renew_now is None


async def test_load_returns_none_for_a_conversation_the_ledger_does_not_hold(
    new_store,
) -> None:
    store = new_store()

    assert await store.load("principal", "conv_unknown") is None


async def test_save_appends_only_the_keys_that_changed(new_store) -> None:
    calls: list[dict[str, Any]] = []
    store = new_store()
    real_write = store._write

    async def recording_write(conn, pk, delta, **kwargs):
        calls.append(dict(delta))
        return await real_write(conn, pk, delta, **kwargs)

    store._write = recording_write
    state = await store.create("principal", CONVERSATION)
    state.turn_index = 1
    state.turn_user_message = "first save touches many fields"

    await store.save("principal", CONVERSATION)

    assert len(calls) == 1
    first_save_key_count = len(calls[0])
    assert first_save_key_count > 1

    state.turn_index = 2  # only this one field changes

    await store.save("principal", CONVERSATION)

    assert len(calls) == 2
    assert calls[1] == {studio_store._KEY_TURN_INDEX: 2}

    # a save with nothing changed appends no event at all
    await store.save("principal", CONVERSATION)
    assert len(calls) == 2


async def test_a_non_json_value_raises_rather_than_being_silently_coerced(new_store) -> None:
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.turn_components = [{"bad": object()}]

    with pytest.raises(TypeError):
        await store.save("principal", CONVERSATION)


async def test_second_claim_over_one_database_service_raises_stale(new_store) -> None:
    """Two `StudioStore`s over one database. Both load the same brand-new
    conversation. The first to save its claimed `turn_index` wins, and the
    second, which still holds its pre-claim snapshot, raises
    `ConversationStale` instead of overwriting the first claim."""
    store_a = new_store()
    store_b = new_store()
    await store_a.create("principal", CONVERSATION)
    await store_b.load("principal", CONVERSATION)

    store_a.current("principal", CONVERSATION).turn_index = 1
    await store_a.save("principal", CONVERSATION)

    store_b.current("principal", CONVERSATION).turn_index = 1
    with pytest.raises(ConversationStale):
        await store_b.save("principal", CONVERSATION)


async def test_a_lease_less_save_during_another_instances_live_lease_is_refused(
    new_store, clock
) -> None:
    """B saves with no lease of its own while A streams. The save is refused,
    since A's stream alone writes the turn state. Once A's lease lapses, the
    same save lands."""
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    state_b = await store_b.load("principal", CONVERSATION)

    state_b.turn_final_text = "b's write"
    with pytest.raises(ConversationStale):
        await store_b.save("principal", CONVERSATION)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    await store_b.save("principal", CONVERSATION)
    reloaded = await new_store(clock=clock).load("principal", CONVERSATION)
    assert reloaded.turn_final_text == "b's write"


async def test_the_lease_holders_own_save_is_not_refused(new_store) -> None:
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    state.turn_final_text = "the holder's write"
    await store.save("principal", CONVERSATION)

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.turn_final_text == "the holder's write"


async def test_instances_whose_clocks_disagree_agree_on_a_lease_by_the_databases_clock(
    pg_engine,
) -> None:
    """B's own clock runs a day ahead of A's. A's lease is live by the
    database's clock, so B can neither claim it nor miss it when cancelling."""
    store_a = StudioStore(pg_engine)
    store_b = StudioStore(pg_engine, clock=lambda: time.time() + 86_400)
    await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    await store_b.load("principal", CONVERSATION)

    with pytest.raises(SegmentBusy):
        store_b.claim_segment("principal", CONVERSATION)
    assert await store_b.request_cancel("principal", CONVERSATION) is True


async def test_each_claim_numbers_its_frames_past_every_number_reserved_before(
    new_store, clock
) -> None:
    """A turn's first claim numbers from 0. A resume on another instance
    starts at the end of everything reserved before, however few numbers the
    earlier segment used, and so does a takeover of a segment that died. A
    new turn numbers from 0 again."""
    store_a = new_store(clock=clock)
    state_a = await store_a.create("principal", CONVERSATION)
    state_a.begin_turn()
    first = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    assert state_a.seq == 0
    state_a.seq = 37
    await store_a.close_segment("principal", CONVERSATION, first)

    store_b = new_store(clock=clock)
    state_b = await store_b.load("principal", CONVERSATION)
    assert state_b.seq == 37
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)
    assert state_b.seq == SEQ_BLOCK
    state_b.seq = SEQ_BLOCK + 10
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)  # B dies mid-segment

    store_c = new_store(clock=clock)
    state_c = await store_c.load("principal", CONVERSATION)
    store_c.claim_segment("principal", CONVERSATION)
    await store_c.save("principal", CONVERSATION)
    assert state_c.seq == 2 * SEQ_BLOCK

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    state_c.segment = None
    await store_c.save("principal", CONVERSATION)
    state_c.begin_turn()
    store_c.claim_segment("principal", CONVERSATION)
    await store_c.save("principal", CONVERSATION)
    assert state_c.seq == 0


async def test_a_takeover_numbers_past_every_frame_a_segment_sent_without_reserving_more(
    new_store, clock
) -> None:
    """The claim reserves more numbers than a segment sends, so a stream
    that sent ten thousand frames, reserving none, and died leaves a
    takeover starting past all of them."""
    store_a = new_store(clock=clock)
    state_a = await store_a.create("principal", CONVERSATION)
    state_a.begin_turn()
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    state_a.seq = 10_000
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)

    store_b = new_store(clock=clock)
    state_b = await store_b.load("principal", CONVERSATION)
    assert state_b is not None
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    assert state_b.seq > 10_000


# --- the segment lease: claim_segment / renew / request_cancel --------------


async def test_claim_segment_raises_segment_busy_while_a_live_segment_exists(
    new_store,
    clock,
) -> None:
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    await clock.advance(10)  # well inside the 45s TTL
    await store_b.load("principal", CONVERSATION)

    with pytest.raises(SegmentBusy) as excinfo:
        store_b.claim_segment("principal", CONVERSATION)

    assert excinfo.value.retry_after_seconds == studio_state.SEGMENT_TTL_S - 10


@pytest.mark.parametrize("gap", [0, studio_state.SEGMENT_TTL_S + 5], ids=["at-once", "past-ttl"])
async def test_a_claim_save_starts_the_lease_at_the_write(new_store, clock, gap) -> None:
    """A claim save (`/interrupt`'s, Edit Card's) may come long after its
    claim. The lease runs `SEGMENT_TTL_S` from the save however long that
    took."""
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    await store_a.load("principal", CONVERSATION)
    work = store_a.draft("principal", CONVERSATION)
    work.begin_turn()
    segment_id = store_a.claim_segment("principal", CONVERSATION, work)

    await clock.advance(gap)
    await store_a.save("principal", CONVERSATION, work)

    saved_at = clock.at
    stored = (await new_store(clock=clock).load("principal", CONVERSATION)).segment
    assert stored is not None
    assert (stored.id, stored.expires_at) == (segment_id, saved_at + studio_state.SEGMENT_TTL_S)
    live = store_a.current("principal", CONVERSATION)
    assert live is not None and live.segment == stored
    assert store_a.lease_deadline("principal", CONVERSATION) == (
        saved_at + studio_state.SEGMENT_TTL_S - studio_store.LEASE_MARGIN_S
    )
    await store_b.load("principal", CONVERSATION)
    with pytest.raises(SegmentBusy):
        store_b.claim_segment("principal", CONVERSATION)
    assert await store_a.renew("principal", CONVERSATION, segment_id) == Renewal(
        held=True, stop_id=None
    )


async def test_claim_segment_succeeds_once_the_live_segment_has_expired(
    new_store,
    clock,
) -> None:
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    first_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    await store_b.load("principal", CONVERSATION)

    second_id = store_b.claim_segment("principal", CONVERSATION)

    assert second_id != first_id


async def test_renew_reports_lost_after_a_takeover(new_store, clock) -> None:
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)  # A's segment has expired
    await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    assert not (await store_a.renew("principal", CONVERSATION, segment_id)).held


async def test_renew_reports_ok_and_pushes_the_ttl_forward(new_store, clock, pg_engine) -> None:
    store = new_store(clock=clock)
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    await clock.advance(30)
    assert await store.renew("principal", CONVERSATION, segment_id) == Renewal(
        held=True, stop_id=None
    )

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.segment is not None
    assert await _stop_columns(pg_engine) == (False, None)
    assert reloaded.segment.expires_at == clock.at + studio_state.SEGMENT_TTL_S


async def test_renew_reports_cancelled_and_still_pushes_the_ttl_forward(
    new_store, clock, pg_engine
) -> None:
    """`request_cancel`, from another instance, records a Stop request on
    the segment; the owner's renewals report it while no party has acted on
    it, and still renew, so a renewal whose acknowledgement is lost cannot
    drop it. A later request has a new id."""
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    assert await store_b.request_cancel("principal", CONVERSATION) is True

    await clock.advance(30)
    first = await store_a.renew("principal", CONVERSATION, segment_id)
    again = await store_a.renew("principal", CONVERSATION, segment_id)
    assert first.held and first.stop_id is not None
    assert again == first

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.segment is not None
    assert await _stop_columns(pg_engine) == (False, first.stop_id)
    assert await store_b.request_cancel("principal", CONVERSATION) is True
    assert (await store_a.renew("principal", CONVERSATION, segment_id)).stop_id not in (
        None,
        first.stop_id,
    )
    assert reloaded.segment.expires_at == clock.at + studio_state.SEGMENT_TTL_S


async def _stop_columns(engine: AsyncEngine) -> tuple[bool, str | None]:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT lease_cancel, stop_id FROM conversations WHERE conversation_id = :c"),
                {"c": CONVERSATION},
            )
        ).one()
    return row.lease_cancel, row.stop_id


async def test_a_stop_set_only_as_the_flag_reaches_a_renewal(new_store, clock, pg_engine) -> None:
    """A writer that knows only the `lease_cancel` flag requests a Stop by
    setting it. A renewal reports it under an id, clears the flag, and
    reports the same id again, so a lost acknowledgement drops nothing."""
    store = new_store(clock=clock)
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("UPDATE conversations SET lease_cancel = true WHERE conversation_id = :c"),
            {"c": CONVERSATION},
        )

    first = await store.renew("principal", CONVERSATION, segment_id)
    assert first.held and first.stop_id is not None
    assert await store.renew("principal", CONVERSATION, segment_id) == first
    assert await _stop_columns(pg_engine) == (False, first.stop_id)


async def test_a_stop_request_sets_only_the_flag_and_the_renewal_records_its_id(
    new_store, clock, pg_engine
) -> None:
    """`request_cancel` sets `lease_cancel` and leaves the id to the
    renewal, so a reader of the flag alone sees the Stop and the id a
    renewal reported is the one that stays stored."""
    store = new_store(clock=clock)
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    assert await new_store(clock=clock).request_cancel("principal", CONVERSATION) is True
    assert await _stop_columns(pg_engine) == (True, None)
    renewed = await store.renew("principal", CONVERSATION, segment_id)
    assert renewed.stop_id is not None
    assert await _stop_columns(pg_engine) == (False, renewed.stop_id)


async def test_a_stop_naming_its_turn_is_live_for_that_turn_alone(
    new_store, clock, pg_engine
) -> None:
    """`request_cancel` given the turn a Stop is for records the Stop naming
    that turn at once, with no flag, while a live lease holds it, and the
    holder's renewal reports it. For another turn it raises `TurnMoved`
    naming the row's turn and records nothing. A turn the holder begins
    after the Stop, under the same lease, runs on: a flag stamped by a
    later lease write would name the new turn."""
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    stopper = new_store(clock=clock)

    with pytest.raises(TurnMoved) as moved:
        await stopper.request_cancel("principal", CONVERSATION, turn_index=2)
    assert moved.value.latest == 1
    assert await _stop_row(pg_engine) == (False, None, None)

    assert await stopper.request_cancel("principal", CONVERSATION, turn_index=1) is True
    flagged, stop_id, stop_turn = await _stop_row(pg_engine)
    assert (flagged, stop_id is not None, stop_turn) == (False, True, 1)
    assert (await store.renew("principal", CONVERSATION, segment_id)).stop_id == stop_id

    state.begin_turn()
    await store.save("principal", CONVERSATION)
    assert (await store.renew("principal", CONVERSATION, segment_id)).stop_id is None
    assert not await stopper.stop_live("principal", CONVERSATION)


class _ClaimAfterFirstStatement:
    """A connection that runs `claim` once its first statement returns, so a
    claim from another instance commits between that statement and the
    next."""

    def __init__(self, conn: Any, claim: Callable[[], Awaitable[None]]) -> None:
        self._conn = conn
        self._claim: Callable[[], Awaitable[None]] | None = claim

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._conn.execute(*args, **kwargs)
        if self._claim is not None:
            claim, self._claim = self._claim, None
            await claim()
        return result


async def test_a_stop_racing_a_claim_of_its_own_turn_is_recorded_for_it(
    new_store, clock, pg_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 1 is open with no lease, as a turn whose client left is, when a
    Stop names it. Another instance's resume claims turn 1 after the Stop's
    guarded update found no live lease and before it reads the row again.
    The Stop finds turn 1 held, so it records itself naming turn 1, and the
    holder's renewal reports it, with no `TurnMoved` naming the turn the
    Stop named."""
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    resumer = new_store(clock=clock)
    held: list[str] = []

    async def claim() -> None:
        await resumer.load("principal", CONVERSATION)
        work = resumer.draft("principal", CONVERSATION)
        held.append(resumer.claim_segment("principal", CONVERSATION, work))
        await resumer.save("principal", CONVERSATION, work)

    stopper = new_store(clock=clock)
    transaction = stopper._transaction

    @asynccontextmanager
    async def claimed_between() -> AsyncIterator[Any]:
        async with transaction() as conn:
            yield _ClaimAfterFirstStatement(conn, claim)

    monkeypatch.setattr(stopper, "_transaction", claimed_between)

    assert await stopper.request_cancel("principal", CONVERSATION, turn_index=1) is True
    (segment_id,) = held
    flagged, stop_id, stop_turn = await _stop_row(pg_engine)
    assert (flagged, stop_id is not None, stop_turn) == (False, True, 1)
    assert (await resumer.renew("principal", CONVERSATION, segment_id)).stop_id == stop_id


async def _stop_row(engine: AsyncEngine) -> tuple[bool, str | None, int | None]:
    """The conversation's `lease_cancel`, `stop_id` and `stop_turn`."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(text("SELECT lease_cancel, stop_id, stop_turn FROM conversations"))
        ).one()
    return row.lease_cancel, row.stop_id, row.stop_turn


async def _streaming_turn(
    new_store: Callable[..., StudioStore], clock: StoreClock, pg_engine: AsyncEngine, cursor: str
) -> tuple[StudioStore, str]:
    """A store streaming turn 1 under a claimed lease, with a Stop requested
    and not yet stamped. `cursor` is the turn's cursor: `none` before it
    opens, else `running`, `paused`, `taken` (paused by a pause write that
    answered a Stop) or `done`."""
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    turns = TurnStore(pg_engine)
    if cursor != "none":
        opened = await turns.open_turn(
            "principal",
            CONVERSATION,
            segment_id=segment_id,
            turn_index=1,
            node="seed",
            inputs={"message": "q"},
            state={},
        )
    assert await new_store(clock=clock).request_cancel("principal", CONVERSATION) is True
    if cursor in ("paused", "taken"):
        taken = None
        if cursor == "taken":
            taken = (await store.renew("principal", CONVERSATION, segment_id)).stop_id
            assert taken is not None
        await turns.pause(
            opened,
            segment_id=segment_id,
            node="ask",
            node_input=None,
            kind="ask",
            interrupt_id="int_1",
            payload={},
            state={},
            stop_taken_id=taken,
            lineage=None,
        )
    elif cursor == "done":
        await turns.finish(opened, segment_id=segment_id, status=TurnStatus.DONE)
    return store, segment_id


@pytest.mark.parametrize(
    "write", ["renew", "close-segment", "drop-lease", "claim-resume", "claim-next-turn"]
)
async def test_the_next_lease_write_stamps_a_stop_with_the_turn_index_it_found(
    new_store, clock, pg_engine, write: str
) -> None:
    """A Stop is a flag until the next write of the lease columns, which
    records it as a fresh id naming the row's turn index as stored before
    that write, and clears the flag. A claim that begins the next turn
    stamps the turn it moves past."""
    store, segment_id = await _streaming_turn(new_store, clock, pg_engine, "none")
    assert await _stop_row(pg_engine) == (True, None, None)

    match write:
        case "renew":
            assert (await store.renew("principal", CONVERSATION, segment_id)).stop_id is not None
        case "close-segment":
            await store.close_segment("principal", CONVERSATION, segment_id)
        case "drop-lease":
            await store.drop_lease("principal", CONVERSATION, segment_id)
        case _:
            await clock.advance(studio_state.SEGMENT_TTL_S + 1)
            other = new_store(clock=clock)
            await other.load("principal", CONVERSATION)
            work = other.draft("principal", CONVERSATION)
            if write == "claim-next-turn":
                work.begin_turn()
            other.claim_segment("principal", CONVERSATION, work)
            await other.save("principal", CONVERSATION, work)

    flagged, stop_id, stop_turn = await _stop_row(pg_engine)
    assert (flagged, stop_id is not None, stop_turn) == (False, True, 1)


async def _admit_run(engine: AsyncEngine) -> None:
    """An admitted run on the conversation, which a close `cancelled` ends."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO runs (conversation, run_id, record) "
                "SELECT id, 'run_1', CAST(:record AS jsonb) FROM conversations"
            ),
            {"record": json.dumps(ADMITTED_RUN)},
        )


async def _computing_runs(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT count(*) FROM runs WHERE record ->> 'status' IN ('admitted', 'running')"
                )
            )
        ).scalar_one()


@pytest.mark.parametrize(
    ("cursor", "closed"),
    [
        pytest.param("none", SegmentClose.STOPPED, id="no-cursor"),
        pytest.param("running", SegmentClose.STOPPED, id="running"),
        pytest.param("paused", SegmentClose.STOPPED, id="paused"),
        pytest.param("taken", SegmentClose.SAVED, id="paused-on-the-stop-its-scope-took"),
        pytest.param("done", SegmentClose.SAVED, id="closed"),
    ],
)
async def test_a_closing_write_saves_the_segment_and_closes_a_turn_a_stop_is_live_for(
    new_store, clock, pg_engine, cursor: str, closed: SegmentClose
) -> None:
    """The closing write stores the live state and clears the lease in one
    transaction. When a Stop is live for the open turn, that transaction
    also closes the cursor and the record `cancelled`, clears the pending
    interrupts and ends the conversation's runs. Either way no Stop is left
    live and the live state equals the row."""
    store, segment_id = await _streaming_turn(new_store, clock, pg_engine, cursor)
    await _admit_run(pg_engine)
    state = store.current("principal", CONVERSATION)
    assert state is not None
    state.pending["int_1"] = _pending()
    state.turn_final_text = "the answer so far"
    state.seq = 41
    if cursor == "done":
        # The stream's last frames record the turn its cursor closed.
        state.close_turn(final_text=state.turn_final_text)

    assert await store.close_segment("principal", CONVERSATION, segment_id) is closed

    stopped = closed is SegmentClose.STOPPED
    stored = await new_store(clock=clock).load("principal", CONVERSATION)
    assert stored is not None
    assert (stored.segment, stored.seq, stored.turn_final_text) == (None, 41, "the answer so far")
    assert set(stored.pending) == (set() if stopped else {"int_1"})
    assert stored.turn_recorded is (stopped or cursor == "done")
    if stopped:
        assert stored.turns[-1]["error"] == "cancelled"
    assert (await TurnStore(pg_engine).open_cursor("principal", CONVERSATION) is None) is (
        stopped or cursor in ("none", "done")
    )
    assert await _computing_runs(pg_engine) == (0 if stopped else 1)
    assert not await new_store(clock=clock).stop_live("principal", CONVERSATION)
    assert not (await store.renew("principal", CONVERSATION, segment_id)).held
    assert studio_store._serialize_durable(state) == studio_store._serialize_durable(stored)


@pytest.mark.parametrize(
    ("fault", "closed"),
    [
        pytest.param("takeover", SegmentClose.LOST, id="takeover"),
        pytest.param("database-error", SegmentClose.FAILED, id="database-error"),
        pytest.param("cursor-close-fails", SegmentClose.FAILED, id="cursor-close-fails"),
        pytest.param("timeout", SegmentClose.FAILED, id="timeout"),
    ],
)
async def test_a_closing_write_that_does_not_commit_stores_nothing_and_holds_the_lease_no_longer(
    new_store, clock, pg_engine, monkeypatch, fault: str, closed: SegmentClose
) -> None:
    """A closing write fenced out by a takeover, refused by the database,
    failing to close the turn on a live Stop, or past its timeout raises
    nothing and stores nothing: the cursor stays open, the record and the
    live fields unsaved. The live state keeps what was not stored, and the
    instance gives the lease up, so a later save that claims nothing is
    fenced out."""
    store, segment_id = await _streaming_turn(new_store, clock, pg_engine, "running")
    state = store.current("principal", CONVERSATION)
    assert state is not None
    state.turn_final_text = "unsaved"
    timeout = 5.0
    match fault:
        case "takeover":
            await clock.advance(studio_state.SEGMENT_TTL_S + 1)
            other = new_store(clock=clock)
            await other.load("principal", CONVERSATION)
            other.claim_segment("principal", CONVERSATION)
            await other.save("principal", CONVERSATION, stopping=True)
        case "database-error":

            async def refused(*args: Any, **kwargs: Any) -> int:
                raise RuntimeError("the database refused the write")

            monkeypatch.setattr(store, "_write", refused)
        case "cursor-close-fails":

            async def unended(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("the runs could not be ended")

            monkeypatch.setattr("harness.turn.store.end_unfinished", unended)
        case "timeout":
            real_transaction = store._transaction

            @asynccontextmanager
            async def stuck() -> Any:
                async with real_transaction() as conn:
                    await asyncio.Event().wait()  # a row lock never granted
                    yield conn

            monkeypatch.setattr(store, "_transaction", stuck)
            timeout = 0.05

    assert (
        await store.close_segment("principal", CONVERSATION, segment_id, timeout=timeout) is closed
    )

    monkeypatch.undo()
    stored = await new_store(clock=clock).load("principal", CONVERSATION)
    assert stored is not None
    assert (stored.turn_recorded, stored.turn_final_text) == (False, "")
    assert stored.segment is not None
    open_turn = await TurnStore(pg_engine).open_cursor("principal", CONVERSATION)
    assert open_turn is not None and open_turn.status is TurnStatus.RUNNING
    assert (state.turn_final_text, state.segment is not None) == ("unsaved", True)
    reloaded = await store.load("principal", CONVERSATION)
    assert reloaded is not None
    reloaded.turn_final_text = "written without the lease"
    with pytest.raises(ConversationStale):
        await store.save("principal", CONVERSATION)


async def test_an_evicted_conversation_is_read_afresh_by_the_next_load(new_store) -> None:
    """An instance whose live state holds what its closing write did not
    store forgets it. The next load reads the row into a fresh state that
    keeps the conversation's lock."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.turn_final_text = "never stored"

    store.evict("principal", CONVERSATION)

    assert store.current("principal", CONVERSATION) is None
    reloaded = await store.load("principal", CONVERSATION)
    assert reloaded is not None and reloaded is not state
    assert reloaded.turn_final_text == ""
    assert reloaded.lock is store.lock_for("principal", CONVERSATION)


_FAILED_AT_ORCHESTRATE = {
    "node": "orchestrate",
    "type": "ToolRefused",
    "message": "the source is gone",
    "transient": False,
    "refusal": {"code": "capability_unavailable", "retry_mode": "do_not_retry"},
}


@pytest.mark.parametrize(
    ("status", "error", "recorded"),
    [
        pytest.param(TurnStatus.DONE, None, "state_not_saved", id="done"),
        pytest.param(
            TurnStatus.FAILED, _FAILED_AT_ORCHESTRATE, "capability_unavailable", id="failed"
        ),
        pytest.param(TurnStatus.CANCELLED, None, "cancelled", id="cancelled"),
    ],
)
@pytest.mark.parametrize("lease", ["lapsed", "live"])
async def test_a_turn_closed_under_a_failed_closing_write_is_recorded_as_its_stream_ended(
    new_store,
    clock,
    pg_engine,
    monkeypatch,
    caplog,
    status: TurnStatus,
    error: dict[str, Any] | None,
    recorded: str,
    lease: str,
) -> None:
    """The turn's cursor closes, then the stream's closing write fails, and
    the record stays open with no line. Once the lease has lapsed, the next
    load records the turn with the error the stream told the client, with no
    reply: a Stop's `cancelled`, a failed turn's own code, and otherwise
    `state_not_saved`, and writes its one `studio.turn` line. It keeps the
    plans and readings the cursor's state holds, and the next turn claims
    and opens with no second line. While the lease is live its holder
    records the turn, so a load leaves the record open and writes no
    line."""
    caplog.set_level(logging.INFO, logger="harness.studio_store")
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    opened_trace_id = state.trace_id
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    turns = TurnStore(pg_engine)
    opened = await turns.open_turn(
        "principal",
        CONVERSATION,
        segment_id=segment_id,
        turn_index=1,
        node="seed",
        inputs={},
        state={},
    )
    await turns.finish(
        opened,
        segment_id=segment_id,
        status=status,
        error=error,
        state={
            ORCHESTRATOR_PLAN_RECORDS_KEY: [_PLAN_JSON],
            ORCHESTRATOR_SEARCHES_KEY: [{"part": "kenya gdp"}],
        },
    )
    state.close_turn(final_text="the answer the client never got")

    async def refused(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("the database refused the write")

    monkeypatch.setattr(store, "_write", refused)
    assert await store.close_segment("principal", CONVERSATION, segment_id) is SegmentClose.FAILED
    monkeypatch.undo()
    assert _turn_log_lines(caplog, status=None) == []
    store.evict("principal", CONVERSATION)
    if lease == "lapsed":
        await clock.advance(studio_state.SEGMENT_TTL_S + 1)

    other = new_store(clock=clock)
    loaded = await other.load("principal", CONVERSATION)

    assert loaded is not None
    if lease == "live":
        assert not loaded.turn_recorded
        assert _turn_log_lines(caplog, status=None) == []
        return
    assert [(line["trace_id"], line["error"]) for line in _turn_log_lines(caplog, status=None)] == [
        (opened_trace_id, recorded)
    ]
    assert loaded.turn_recorded
    record = loaded.turns[-1]
    assert (record["error"], record["assistant"]) == (recorded, "")
    assert record["plans"] == [_PLAN_JSON]
    assert record["readings"][0] == {"part": "kenya gdp"}
    stored = await new_store(clock=clock).load("principal", CONVERSATION)
    assert stored is not None and stored.turns == loaded.turns
    work = other.draft("principal", CONVERSATION)
    assert work.begin_turn() is None
    next_segment = other.claim_segment("principal", CONVERSATION, work)
    await other.save("principal", CONVERSATION, work)
    await turns.open_turn(
        "principal",
        CONVERSATION,
        segment_id=next_segment,
        turn_index=2,
        node="seed",
        inputs={},
        state={},
    )
    assert len(_turn_log_lines(caplog, status=None)) == 1


@pytest.mark.parametrize("claim", ["next-turn", "resume"])
async def test_a_claim_that_starts_work_refuses_a_live_stop_and_one_that_acts_on_it_lands(
    new_store, clock, pg_engine, claim: str
) -> None:
    """With a Stop live for the open turn whose lease lapsed, a claim that
    begins the next turn or resumes this one refuses, and leaves the row as
    it was. A claim that acts on the Stop lands and names the open turn."""
    await _streaming_turn(new_store, clock, pg_engine, "running")
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    other = new_store(clock=clock)
    await other.load("principal", CONVERSATION)

    work = other.draft("principal", CONVERSATION)
    if claim == "next-turn":
        work.begin_turn()
    other.claim_segment("principal", CONVERSATION, work)
    with pytest.raises(StopPending):
        await other.save("principal", CONVERSATION, work)
    other.drop_draft("principal", CONVERSATION)
    assert await _stop_row(pg_engine) == (True, None, None)

    other.claim_segment("principal", CONVERSATION)
    await other.save("principal", CONVERSATION, stopping=True)
    flagged, stop_id, stop_turn = await _stop_row(pg_engine)
    assert (flagged, stop_id is not None, stop_turn) == (False, True, 1)
    assert await other.stop_live("principal", CONVERSATION)


async def test_request_cancel_returns_false_with_no_live_segment(new_store) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)

    assert await store.request_cancel("principal", CONVERSATION) is False


async def test_request_cancel_returns_false_once_the_segment_has_expired(new_store, clock) -> None:
    store = new_store(clock=clock)
    await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)

    assert await store.request_cancel("principal", CONVERSATION) is False


# --- closing write: fencing on the segment ---------------------------------


async def test_the_closing_write_is_lost_after_a_takeover_and_writes_nothing(
    new_store,
    clock,
) -> None:
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)  # A's segment has expired
    await store_b.load("principal", CONVERSATION)
    new_segment_id = store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)
    assert new_segment_id != segment_id

    store_a.current("principal", CONVERSATION).turn_final_text = "a's answer, too late"
    assert await store_a.close_segment("principal", CONVERSATION, segment_id) is SegmentClose.LOST

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.turn_final_text == ""
    assert reloaded.segment is not None and reloaded.segment.id == new_segment_id


async def test_an_evicted_instance_holds_no_lease_it_did_not_close(new_store) -> None:
    """A stream ended with no closing write (a lost lease) evicts its state.
    The lease it claimed is still stored and live, and a later load on the
    same instance does not count it as held, so a save without a claim of
    its own is refused as it would be on any other instance."""
    store = new_store()
    await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    store.evict("principal", CONVERSATION)
    state = await store.load("principal", CONVERSATION)
    state.turn_final_text = "written under a lease this stream gave up"

    with pytest.raises(ConversationStale):
        await store.save("principal", CONVERSATION)


async def test_the_closing_write_is_lost_when_the_ledger_is_gone(new_store, pg_engine) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    async with pg_engine.begin() as conn:
        await conn.execute(text("DELETE FROM conversations"))

    store.current("principal", CONVERSATION).turn_final_text = "an answer with no ledger"
    assert await store.close_segment("principal", CONVERSATION, segment_id) is SegmentClose.LOST


# --- canvas writes: change_canvas and mutate_composition ------------------


def _one_block_manifest(
    manifest: CompositionManifest | None, _components: object, *, component_id: str = "cmp_one"
) -> CompositionManifest:
    block = CompositionBlock(
        component_id=component_id, revision=1, type=ComponentType.LINE_CHART, title="One"
    )
    return _next_manifest(manifest, [block], actor=CompositionActor.USER)


async def test_mutate_composition_writes_the_returned_manifest(new_store) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)
    state = store.current("principal", CONVERSATION)
    await store.mutate_composition("principal", CONVERSATION, _one_block_manifest)

    def mutate(manifest: CompositionManifest | None, _components: object) -> CompositionManifest:
        assert manifest is not None
        pinned = manifest.blocks[0].model_copy(update={"pinned": True})
        return _next_manifest(manifest, [pinned], actor=CompositionActor.USER)

    result = await store.mutate_composition("principal", CONVERSATION, mutate)

    assert result is not None and result.blocks[0].pinned is True
    assert state.composition is result

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.composition.blocks[0].pinned is True


_CARD = {"component_id": "cmp_two", "type": "table", "revision": 1}


async def test_a_card_the_turn_lands_is_written_onto_the_canvas_as_stored(new_store) -> None:
    """A streams a turn holding a canvas it read before B pinned a card on
    another instance. A's card lands on the canvas as stored, so B's pin is
    kept, and A's live canvas is the one written."""
    store_a = new_store()
    store_b = new_store()
    state_a = await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    await store_a.mutate_composition("principal", CONVERSATION, _one_block_manifest)
    await store_b.load("principal", CONVERSATION)

    def pin(manifest: CompositionManifest | None, _components: object) -> CompositionManifest:
        assert manifest is not None
        pinned = manifest.blocks[0].model_copy(update={"pinned": True})
        return _next_manifest(manifest, [pinned], actor=CompositionActor.USER)

    await store_b.mutate_composition("principal", CONVERSATION, pin)

    def land(work: ConversationState) -> str:
        # The create's write token lands with its card (`_Translator._land`).
        work.write_tokens = {**work.write_tokens, "cmp_two": "token-two"}
        return land_component(work, _CARD)

    landed = await store_a.change_canvas("principal", CONVERSATION, land)

    assert landed == "created"
    reloaded = await new_store().load("principal", CONVERSATION)
    assert [b.component_id for b in reloaded.composition.blocks] == ["cmp_one", "cmp_two"]
    assert reloaded.composition.blocks[0].pinned is True
    assert state_a.composition == reloaded.composition
    assert "cmp_two" in state_a.components
    # The card's digest entry, write token and turn line commit with its block.
    assert reloaded.components == state_a.components
    assert reloaded.write_tokens == {"cmp_two": "token-two"}
    assert reloaded.turn_components == state_a.turn_components


def _card_lineage(component_id: str, *, count: int | None) -> CardLineage:
    """A card drawn on frame `ops_f` resting on two retrievals, asked for as `ops_asked`."""
    return CardLineage(
        component_id=component_id,
        frame=("ops_f", "ocv_f"),
        lineage=(
            ProvenanceInput("spec_a", "cv_a", {"citation": {"sources": [{"ref": "src:a"}]}}),
            ProvenanceInput("spec_b", "cv_b", {"trace": {"resolved_stat_vars": [{"ref": "x"}]}}),
        ),
        stat_var_count=count,
        requested=("ops_asked", "ocv_asked"),
    )


async def _claimed(store: StudioStore) -> None:
    await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)


def _landing(component_id: str) -> Callable[[ConversationState], Any]:
    return lambda work: land_component(work, {**_CARD, "component_id": component_id})


async def test_a_cards_lineage_commits_with_it_once_and_any_instance_reads_it(new_store) -> None:
    """The lineage of a card's frame commits in the card's own write. Another
    instance reads it by the card, by the frame it was drawn from and by the
    frame its write named. A later write of the same card, a revise or a
    retried commit, leaves the first lineage stored."""
    store_a = new_store()
    await _claimed(store_a)
    first = _card_lineage("cmp_two", count=1)

    await store_a.change_canvas("principal", CONVERSATION, _landing("cmp_two"), lineage=first)
    await store_a.change_canvas(
        "principal",
        CONVERSATION,
        _landing("cmp_two"),
        lineage=_card_lineage("cmp_two", count=7),
    )

    store_b = new_store()
    await store_b.load("principal", CONVERSATION)
    read = store_b.card_lineage
    assert await read("principal", CONVERSATION, component_id="cmp_two") == first
    assert await read("principal", CONVERSATION, frame=("ops_f", "ocv_f")) == first
    assert await read("principal", CONVERSATION, frame=("ops_asked", "ocv_asked")) == first
    assert await read("principal", CONVERSATION, component_id="cmp_other") is None
    assert await read("principal", CONVERSATION, frame=("ops_g", "ocv_g")) is None
    with pytest.raises(LookupError):
        await read("someone_else", CONVERSATION, component_id="cmp_two")


@pytest.mark.parametrize("wrote_block", [True, False], ids=["block-written", "block-gone"])
async def test_a_card_write_stores_its_lineage_only_when_it_wrote_the_cards_block(
    new_store, wrote_block: bool
) -> None:
    """A revise whose card's block is gone from the canvas writes no block,
    so its lineage is not stored; one that wrote the block stores it."""
    store = new_store()
    await _claimed(store)
    lineage = _card_lineage("cmp_two", count=1)

    returned = await store.change_canvas(
        "principal", CONVERSATION, lambda work: wrote_block, lineage=lineage
    )

    assert returned is wrote_block
    stored = await store.card_lineage("principal", CONVERSATION, component_id="cmp_two")
    assert stored == (lineage if wrote_block else None)


async def test_a_card_lineage_read_prefers_the_cards_own_row_then_its_frame_then_the_latest(
    new_store,
) -> None:
    """One read: the card's own row over a row on the frame, a row drawn from
    the frame over one whose write named it, and the latest of equals."""
    store = new_store()
    await _claimed(store)
    frame, other, named = ("ops_f", "ocv_f"), ("ops_o", "ocv_o"), ("ops_n", "ocv_n")
    rows = [
        dataclasses.replace(_card_lineage("cmp_own", count=1), frame=other, requested=named),
        dataclasses.replace(_card_lineage("cmp_drawn", count=2), requested=None),
        dataclasses.replace(_card_lineage("cmp_later", count=3), requested=None),
        # Written last, and only named the frame.
        dataclasses.replace(_card_lineage("cmp_named", count=4), frame=other, requested=frame),
    ]
    for lineage in rows:
        await store.change_canvas(
            "principal", CONVERSATION, _landing(lineage.component_id), lineage=lineage
        )

    async def found(**by: Any) -> str | None:
        card = await store.card_lineage("principal", CONVERSATION, **by)
        return card.component_id if card is not None else None

    assert await found(component_id="cmp_own", frame=frame) == "cmp_own"
    assert await found(frame=frame) == "cmp_later"
    assert await found(component_id="cmp_none", frame=frame) == "cmp_later"
    assert await found(frame=named) == "cmp_own"


@pytest.mark.parametrize("fault", ["nan-in-a-manifest", "insert-refused"])
async def test_a_card_lands_without_a_lineage_that_cannot_be_stored(
    new_store, monkeypatch, caplog, fault: str
) -> None:
    """A manifest holding a NaN cannot be stored as JSON, and an insert the
    database refuses for good fails its savepoint alone. Either way the card
    lands with its block, entry and turn line, without a lineage row, and
    the failure is logged at error."""
    store = new_store()
    await _claimed(store)
    lineage = _card_lineage("cmp_two", count=1)
    if fault == "nan-in-a-manifest":
        lineage = dataclasses.replace(
            lineage,
            lineage=(ProvenanceInput("spec_a", "cv_a", {"coverage": {"share": float("nan")}}),),
        )
    else:
        monkeypatch.setattr(
            studio_store, "_INSERT_CARD_LINEAGE", text("INSERT INTO no_such_table VALUES (:pk)")
        )

    inserts: list[object] = []
    insert = studio_store._insert_card_lineage

    async def counted(*args: Any) -> None:
        inserts.append(args)
        await insert(*args)

    monkeypatch.setattr(studio_store, "_insert_card_lineage", counted)

    with caplog.at_level(logging.ERROR, logger="harness.studio_store"):
        landed = await store.change_canvas(
            "principal", CONVERSATION, _landing("cmp_two"), lineage=lineage
        )

    assert landed == "created"
    reloaded = await new_store().load("principal", CONVERSATION)
    assert [block.component_id for block in reloaded.composition.blocks] == ["cmp_two"]
    assert "cmp_two" in reloaded.components
    assert await store.card_lineage("principal", CONVERSATION, component_id="cmp_two") is None
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    # A NaN never reaches Postgres: the lineage is refused before the insert.
    assert len(inserts) == (0 if fault == "nan-in-a-manifest" else 1)


async def test_a_card_lineage_row_that_does_not_decode_leaves_the_next_match_to_serve(
    new_store, pg_engine
) -> None:
    """The card's own row comes first and does not decode, so the row of
    another card drawn from its frame serves; with that one broken too, none
    serves."""
    store = new_store()
    await _claimed(store)
    for component_id in ("cmp_one", "cmp_two"):
        await store.change_canvas(
            "principal",
            CONVERSATION,
            _landing(component_id),
            lineage=_card_lineage(component_id, count=1),
        )
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE conversation_card_lineage SET members = '{}' WHERE component_id = 'cmp_two'"
            )
        )

    read = await store.card_lineage(
        "principal", CONVERSATION, component_id="cmp_two", frame=("ops_f", "ocv_f")
    )
    assert read == _card_lineage("cmp_one", count=1)

    async with pg_engine.begin() as conn:
        await conn.execute(text("UPDATE conversation_card_lineage SET members = '[]'"))
    assert (
        await store.card_lineage(
            "principal", CONVERSATION, component_id="cmp_two", frame=("ops_f", "ocv_f")
        )
        is None
    )


async def test_a_card_write_commits_only_the_fields_its_change_made(new_store) -> None:
    """The turn's drive removes a card (`remove_cards`) while the stream's
    owner holds a pending interrupt it has not saved. The removal commits
    the canvas and the fields its change made, and leaves the pending
    interrupt for the owner's own write."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    state.pending["int_1"] = _pending()
    remove = studio_state.make_remove_cards(store)
    token = current_conversation_key.set(("principal", CONVERSATION))
    try:
        assert (await remove(component_ids=["cmp_two"], all_cards=False))[:3] == (
            ["cmp_two"],
            [],
            [],
        )
    finally:
        current_conversation_key.reset(token)

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.composition is not None and reloaded.composition.blocks == []
    assert reloaded.pending == {}
    assert list(state.pending) == ["int_1"]


async def test_a_stream_whose_segment_was_taken_over_lands_no_card(new_store, clock) -> None:
    """A's lease lapsed and B claimed the conversation. A's stream, still
    running, cannot land a card on the canvas B now owns."""
    store_a = new_store(clock=clock)
    state_a = await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    store_b = new_store(clock=clock)
    await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    with pytest.raises(SegmentLost):
        await store_a.change_canvas(
            "principal", CONVERSATION, lambda work: land_component(work, _CARD)
        )

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.composition is None
    assert state_a.composition is None
    assert state_a.components == {}


async def test_a_canvas_write_that_fails_leaves_the_live_state_as_it_was(new_store) -> None:
    """The card's digest entry, turn record and block reach the live state
    only once the canvas write commits, so a later closing write never
    stores a card the canvas does not hold."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    async def failing_write(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the database went away")

    store._write_composition = failing_write

    with pytest.raises(RuntimeError, match="went away"):
        await store.change_canvas(
            "principal", CONVERSATION, lambda work: land_component(work, _CARD)
        )

    assert state.composition is None
    assert state.components == {}
    assert state.turn_components == []


def _driver_error(wrapper: DBAPIError, orig: BaseException) -> DBAPIError:
    """`wrapper` raised from `orig`, as SQLAlchemy raises a driver's error."""
    wrapper.__cause__ = orig
    return wrapper


def _connection_reset(statement: str) -> DBAPIError:
    """The error SQLAlchemy raises when the connection resets under
    `statement`."""
    reset = ConnectionResetError("connection reset")
    return _driver_error(OperationalError(statement, {}, reset), reset)


def _dropping_connection(store: StudioStore, *, acknowledged: bool) -> list[int]:
    """Make the next transaction `store` opens fail as a dropped connection
    does: before its commit, or after it with the acknowledgement lost.
    Returns a list counting the faults raised."""
    real_transaction = store._transaction
    faults: list[int] = []

    def fault() -> OperationalError:
        faults.append(1)
        return _connection_reset("COMMIT")

    @asynccontextmanager
    async def dropping() -> Any:
        async with real_transaction() as conn:
            yield conn
            if not faults and not acknowledged:
                raise fault()
        if not faults and acknowledged:
            raise fault()

    store._transaction = dropping
    return faults


@pytest.mark.parametrize("acknowledged", [False, True], ids=["before-commit", "lost-ack"])
async def test_a_card_write_that_meets_a_dropped_connection_lands_the_card_once(
    new_store, monkeypatch, acknowledged: bool
) -> None:
    """A dropped connection during a card's write is retried. A retry after a
    commit whose acknowledgement was lost reads the block that commit wrote,
    so the card still has one block and one turn line."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    faults = _dropping_connection(store, acknowledged=acknowledged)

    landed = await store.change_canvas(
        "principal", CONVERSATION, lambda work: land_component(work, _CARD)
    )

    assert faults == [1]
    assert landed == "created"
    reloaded = await new_store().load("principal", CONVERSATION)
    assert [b.component_id for b in reloaded.composition.blocks] == ["cmp_two"]
    assert reloaded.composition == state.composition
    assert [line["component_id"] for line in reloaded.turn_components] == ["cmp_two"]
    assert reloaded.turn_components == state.turn_components


async def test_a_card_write_failing_past_the_lease_deadline_reports_the_database_unavailable(
    new_store, clock
) -> None:
    """A write that keeps failing stops retrying at the lease's deadline, ten
    seconds before the lease dated from the claim's load would lapse: past
    it, another instance may claim the conversation. No instance has, so
    the write reports the database unavailable, raised from the last fault,
    in place of a takeover."""
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    deadline = store.lease_deadline("principal", CONVERSATION)
    assert deadline == clock.at + studio_state.SEGMENT_TTL_S - studio_store.LEASE_MARGIN_S
    await clock.set(deadline)

    async def dropped(*args: Any, **kwargs: Any) -> None:
        raise _connection_reset("UPDATE")

    store._write_composition = dropped

    with pytest.raises(DatabaseUnavailable) as raised:
        await store.change_canvas(
            "principal", CONVERSATION, lambda work: land_component(work, _CARD)
        )

    assert isinstance(raised.value.__cause__, OperationalError)
    assert state.components == {}
    assert state.composition is None


@pytest.mark.parametrize(
    ("fault", "retried"),
    [
        pytest.param(
            _driver_error(
                OperationalError("UPDATE", {}, Exception("terminating connection")),
                asyncpg.AdminShutdownError("terminating connection due to administrator command"),
            ),
            True,
            id="connection-ended",
        ),
        pytest.param(
            OperationalError("UPDATE", {}, Exception("closed"), connection_invalidated=True),
            True,
            id="invalidated",
        ),
        pytest.param(OSError(errno.EHOSTUNREACH, "No route to host"), True, id="raw-os-error"),
        pytest.param(
            _driver_error(
                OperationalError("connect", {}, Exception("password authentication failed")),
                asyncpg.InvalidAuthorizationSpecificationError("password authentication failed"),
            ),
            False,
            id="auth-error",
        ),
        pytest.param(
            ssl.SSLCertVerificationError("certificate verify failed"),
            False,
            id="unverified-certificate",
        ),
        pytest.param(
            _driver_error(
                OperationalError(
                    "UPDATE", {}, Exception("canceling statement due to lock timeout")
                ),
                asyncpg.LockNotAvailableError("canceling statement due to lock timeout"),
            ),
            False,
            id="lock-timeout",
        ),
    ],
)
async def test_a_card_write_retries_a_lost_connection_alone(
    new_store, monkeypatch, fault: Exception, retried: bool
) -> None:
    """A write retries a fault the harness's classifier calls transient (a
    connection ended or invalidated, a network that cannot reach the
    database), and raises anything else at once."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    real_write = store._write_composition
    faults: list[Exception] = []

    async def failing_once(*args: Any, **kwargs: Any) -> None:
        if not faults:
            faults.append(fault)
            raise fault
        await real_write(*args, **kwargs)

    store._write_composition = failing_once

    if retried:
        await store.change_canvas(
            "principal", CONVERSATION, lambda work: land_component(work, _CARD)
        )
    else:
        with pytest.raises(type(fault)):
            await store.change_canvas(
                "principal", CONVERSATION, lambda work: land_component(work, _CARD)
            )

    reloaded = await new_store().load("principal", CONVERSATION)
    assert (reloaded.composition is not None) == retried


class _FailingCheckout:
    """An engine whose next `begin()` fails to check a connection out with
    `fault`; everything else goes to `engine`."""

    def __init__(self, engine: Any, fault: BaseException) -> None:
        self._engine = engine
        self.fault: BaseException | None = fault

    def begin(self) -> Any:
        if self.fault is None:
            return self._engine.begin()
        fault, self.fault = self.fault, None

        @asynccontextmanager
        async def failing() -> Any:
            raise fault
            yield  # pragma: no cover -- marks this a generator

        return failing()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


@pytest.mark.parametrize(
    ("fault", "retried"),
    [
        pytest.param(ConnectionRefusedError("refused"), True, id="refused"),
        pytest.param(SQLAlchemyTimeoutError("QueuePool limit reached"), True, id="pool-timeout"),
        pytest.param(
            ssl.SSLCertVerificationError("certificate verify failed"),
            False,
            id="unverified-certificate",
        ),
    ],
)
async def test_a_write_retries_a_connection_checkout_that_failed_alone(
    new_store, monkeypatch, fault: Exception, retried: bool
) -> None:
    """The connection checkout sits inside the retried attempt: a transient
    fault there is retried, and a permanent one raises at once."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    store._engine = _FailingCheckout(store._engine, fault)

    if retried:
        await store.change_canvas(
            "principal", CONVERSATION, lambda work: land_component(work, _CARD)
        )
    else:
        with pytest.raises(type(fault)):
            await store.change_canvas(
                "principal", CONVERSATION, lambda work: land_component(work, _CARD)
            )

    assert store._engine.fault is None
    reloaded = await new_store().load("principal", CONVERSATION)
    assert (reloaded.composition is not None) == retried


async def test_a_card_lineage_read_makes_one_attempt_and_leaves_the_lease_alone(
    new_store, monkeypatch
) -> None:
    """The read a write makes before it is sent is best effort: a transient
    fault raises at once, with no retry and no early lease renewal, and the
    caller goes on without the lineage."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    state.renew_now = asyncio.Event()
    await store.change_canvas(
        "principal",
        CONVERSATION,
        lambda work: land_component(work, _CARD),
        lineage=_card_lineage("cmp_two", count=1),
    )
    store._lineage_engine = _FailingCheckout(
        store._lineage_engine, ConnectionRefusedError("refused")
    )

    with pytest.raises(ConnectionRefusedError):
        await store.card_lineage("principal", CONVERSATION, component_id="cmp_two")
    assert not state.renew_now.is_set()


async def test_a_lineage_read_on_a_stalled_statement_gives_the_one_connection_back(
    pg_database_url: str, monkeypatch
) -> None:
    """Read A stalls in its statement on the lineage engine's one connection
    and nobody waits for it. The statement bound ends A, so read B, waiting
    for the connection, gets it and reads the card's lineage."""
    monkeypatch.setattr(engine_module, "_LINEAGE_COMMAND_TIMEOUT_S", 0.3)
    stalling = studio_store._SELECT_CARD_LINEAGE.text.replace(
        "SELECT component_id",
        "WITH stall AS (SELECT pg_sleep(CASE WHEN CAST(:spec_id AS text) = 'stall' "
        "THEN 30 ELSE 0 END)) SELECT component_id",
        1,
    ).replace("FROM conversation_card_lineage", "FROM conversation_card_lineage, stall", 1)
    monkeypatch.setattr(
        studio_store,
        "_SELECT_CARD_LINEAGE",
        text(stalling).bindparams(bindparam("skipped", type_=ARRAY(Text))),
    )
    engine = create_async_engine(pg_database_url)
    lineage_engine = engine_module.build_lineage_engine(pg_database_url)
    try:
        store = StudioStore(engine, lineage_engine=lineage_engine)
        await _claimed(store)
        lineage = _card_lineage("cmp_two", count=1)
        await store.change_canvas("principal", CONVERSATION, _landing("cmp_two"), lineage=lineage)

        stalled = asyncio.create_task(
            store.card_lineage(
                "principal", CONVERSATION, component_id="cmp_two", frame=("stall", "stall")
            )
        )
        await asyncio.sleep(0.05)
        read = await asyncio.wait_for(
            store.card_lineage("principal", CONVERSATION, component_id="cmp_two"), timeout=5
        )

        assert read == lineage
        with pytest.raises(TimeoutError):
            await stalled
    finally:
        await lineage_engine.dispose()
        await engine.dispose()


class _NoCheckout:
    """An engine that refuses every connection, standing for a session-store
    pool no lineage read may take one from."""

    def begin(self) -> Any:
        raise AssertionError("a lineage read took a session-store connection")

    connect = begin


async def test_a_card_lineage_read_takes_no_connection_from_the_session_store_pool(
    pg_engine,
) -> None:
    """Lineage reads run over the lineage engine alone, so the lease renewals
    and card writes keep every connection of the session-store pool."""
    store = StudioStore(pg_engine, lineage_engine=pg_engine)
    await _claimed(store)
    lineage = _card_lineage("cmp_two", count=1)
    await store.change_canvas("principal", CONVERSATION, _landing("cmp_two"), lineage=lineage)
    store._engine = _NoCheckout()

    assert await store.card_lineage("principal", CONVERSATION, component_id="cmp_two") == lineage


async def test_a_removal_whose_commit_lost_its_acknowledgement_reports_the_cards_it_removed(
    new_store, monkeypatch
) -> None:
    """The retry reads a canvas the card is already off. It still reports the
    card removed, and the stream still owes its `composition.updated`."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    faults = _dropping_connection(store, acknowledged=True)
    remove = studio_state.make_remove_cards(store)
    token = current_conversation_key.set(("principal", CONVERSATION))
    try:
        result = await remove(component_ids=["cmp_two"], all_cards=False)
    finally:
        current_conversation_key.reset(token)

    assert faults == [1]
    assert result[:3] == (["cmp_two"], [], [])
    assert state.take_unreported_removal() is True
    assert state.composition is not None and state.composition.blocks == []
    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.composition == state.composition


def _user_adds(component_id: str) -> Any:
    """`/composition`'s add of `component_id`, as the route builds it."""

    def mutate(
        manifest: CompositionManifest | None, components: dict[str, dict[str, Any]]
    ) -> CompositionManifest:
        applied = studio_state._apply_operations(
            components, manifest, [AddOp(op=MutationOp.ADD, component_id=component_id)]
        )
        assert isinstance(applied, tuple)
        return _next_manifest(manifest, applied[0], actor=CompositionActor.USER)

    return mutate


async def _canvas_writes(engine: AsyncEngine) -> list[str]:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT canvas_writes FROM conversations WHERE conversation_id = :c"),
                {"c": CONVERSATION},
            )
        ).scalar_one()


async def _remove_all(store: StudioStore) -> tuple[list[str], list[str]]:
    remove = studio_state.make_remove_cards(store)
    token = current_conversation_key.set(("principal", CONVERSATION))
    try:
        return await remove(component_ids=[], all_cards=True)
    finally:
        current_conversation_key.reset(token)


async def test_a_retried_removal_of_every_card_keeps_a_card_the_user_added_meanwhile(
    new_store, monkeypatch
) -> None:
    """The removal commits and its acknowledgement is lost. Before the retry,
    the user adds a card through another instance. The retry finds its own
    write recorded, takes nothing more off, and reports what the commit
    removed."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store_a, store_b = new_store(), new_store()
    state = await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    for card in ("cmp_one", "cmp_two"):
        await store_a.change_canvas(
            "principal",
            CONVERSATION,
            lambda work, card=card: land_component(work, _component(card)),
        )
    remove = studio_state.make_remove_cards(store_a)
    token = current_conversation_key.set(("principal", CONVERSATION))
    try:
        await remove(component_ids=["cmp_one"], all_cards=False)
    finally:
        current_conversation_key.reset(token)
    real_transaction = store_a._transaction
    faults: list[int] = []

    @asynccontextmanager
    async def losing_the_ack() -> Any:
        async with real_transaction() as conn:
            yield conn
        if not faults:
            faults.append(1)
            await store_b.mutate_composition(
                "principal", CONVERSATION, _user_adds("cmp_one"), component_ids={"cmp_one"}
            )
            raise _connection_reset("COMMIT")

    store_a._transaction = losing_the_ack

    result = await _remove_all(store_a)

    assert faults == [1]
    assert result[:3] == (["cmp_two"], [], [])
    assert state.take_unreported_removal() is True
    reloaded = await new_store().load("principal", CONVERSATION)
    assert [b.component_id for b in reloaded.composition.blocks] == ["cmp_one"]
    assert state.composition == reloaded.composition


async def test_overlapping_canvas_changes_each_land_once_when_the_first_lost_its_acknowledgement(
    new_store, pg_engine, monkeypatch
) -> None:
    """Two of a turn's canvas changes run at once. A's removal commits and
    its acknowledgement is lost; B lands a card before A retries. A's retry
    finds its own write recorded beside B's and takes B's card off no
    second time."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    real_transaction = store._transaction
    faults: list[int] = []
    landing: list[asyncio.Task[Any]] = []

    @asynccontextmanager
    async def losing_the_ack() -> Any:
        async with real_transaction() as conn:
            yield conn
        if not faults:
            faults.append(1)
            # B is queued on the write lock A holds, so it writes before A's retry.
            landing.append(
                asyncio.create_task(
                    store.change_canvas(
                        "principal",
                        CONVERSATION,
                        lambda work: land_component(work, _component("cmp_one")),
                    )
                )
            )
            await asyncio.sleep(0)
            raise _connection_reset("COMMIT")

    store._transaction = losing_the_ack

    result = await _remove_all(store)

    assert faults == [1]
    assert landing[0].done()
    assert result[:3] == (["cmp_two"], [], [])
    reloaded = await new_store().load("principal", CONVERSATION)
    assert [b.component_id for b in reloaded.composition.blocks] == ["cmp_one"]
    assert state.composition == reloaded.composition
    assert len(await _canvas_writes(pg_engine)) == 3


def _revise_card(work: ConversationState) -> bool:
    """The Edit Card route's revise of `_CARD` to revision 2."""
    _record_revision(work, component_id="cmp_two", component=_CARD, revision=2, title="Renamed")
    return _refresh_block(work, component_id="cmp_two", revision=2, title="Renamed")


def _land_cards(*component_ids: str) -> Callable[[ConversationState], None]:
    """One canvas change landing each of `component_ids` in turn."""

    def land(work: ConversationState) -> None:
        for component_id in component_ids:
            land_component(work, _component(component_id))

    return land


@pytest.mark.parametrize(
    ("first", "first_lines"),
    [
        pytest.param(_land_cards("cmp_one"), [("cmp_one", "created")], id="land-then-land"),
        pytest.param(
            _land_cards("cmp_one", "cmp_three", "cmp_four", "cmp_five"),
            [
                ("cmp_one", "created"),
                ("cmp_three", "created"),
                ("cmp_four", "created"),
                ("cmp_five", "created"),
            ],
            id="land-several-then-land",
        ),
        pytest.param(_revise_card, [("cmp_two", "revised")], id="revise-then-land"),
    ],
)
async def test_a_canvas_change_after_a_lost_acknowledgement_builds_on_the_commit(
    new_store,
    pg_engine,
    monkeypatch,
    first: Callable[[Any], Any],
    first_lines: list[tuple[str, str]],
) -> None:
    """A's canvas change commits and its acknowledgement is lost. B lands a
    card before A retries. B builds on A's commit, so the row and the live
    state each hold both changes' digest entries and turn lines, and the
    baseline equals the row."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    real_transaction = store._transaction
    faults: list[int] = []
    landing: list[asyncio.Task[Any]] = []

    @asynccontextmanager
    async def losing_the_ack() -> Any:
        async with real_transaction() as conn:
            yield conn
        if not faults:
            faults.append(1)
            # B is queued on the write lock A holds, so it writes before A's retry.
            landing.append(
                asyncio.create_task(
                    store.change_canvas(
                        "principal",
                        CONVERSATION,
                        lambda work: land_component(work, _component("cmp_new")),
                    )
                )
            )
            await asyncio.sleep(0)
            raise _connection_reset("COMMIT")

    store._transaction = losing_the_ack

    await store.change_canvas("principal", CONVERSATION, first)

    assert faults == [1]
    assert landing[0].done()
    lines = [("cmp_two", "created"), *first_lines, ("cmp_new", "created")]
    reloaded_store = new_store()
    reloaded = await reloaded_store.load("principal", CONVERSATION)
    for held in (reloaded, state):
        assert [(line["component_id"], line["action"]) for line in held.turn_components] == lines
        assert {cid for cid, _ in lines} <= set(held.components)
    assert state.components == reloaded.components
    assert (
        list(state.components)
        == list(reloaded.components)
        == list(dict.fromkeys(cid for cid, _ in lines))
    )
    assert state.composition == reloaded.composition
    assert (
        store._runtime[("principal", CONVERSATION)].baseline
        == reloaded_store._runtime[("principal", CONVERSATION)].baseline
    )
    assert len(await _canvas_writes(pg_engine)) == 3


async def test_a_landed_canvas_write_whose_settling_read_failed_is_applied_once(
    new_store, pg_engine, monkeypatch
) -> None:
    """A's card lands and its acknowledgement is lost. The retry's read of
    the row's items, which settles the landed write, meets a dropped
    connection too. The next retry settles it: the card has one turn line,
    and its write id is recorded once."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    faults = _dropping_connection(store, acknowledged=True)
    real_select = studio_store._ItemTable.select
    select_faults: list[int] = []

    def failing_once(self: Any) -> Any:
        if faults and not select_faults:
            select_faults.append(1)
            raise _connection_reset("SELECT")
        return real_select(self)

    monkeypatch.setattr(studio_store._ItemTable, "select", failing_once)

    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))

    assert (faults, select_faults) == ([1], [1])
    reloaded = await new_store().load("principal", CONVERSATION)
    for held in (reloaded, state):
        assert [line["component_id"] for line in held.turn_components] == ["cmp_two"]
    assert len(await _canvas_writes(pg_engine)) == 1


async def test_a_closing_write_keeps_the_line_of_a_card_whose_write_gave_up_after_it_landed(
    new_store, monkeypatch
) -> None:
    """A card lands and its acknowledgement is lost, and the write gives up
    without a retry. The stream's failure close records the turn from a live
    state missing the card's line. The closing write settles the landed
    write first, so the stored record and the live state hold the line
    once."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    faults = _dropping_connection(store, acknowledged=True)

    async def giving_up(entry: Any, conversation_id: str, attempt: Any) -> Any:
        return await attempt()

    monkeypatch.setattr(store, "_retrying", giving_up)
    with pytest.raises(OperationalError):
        await store.change_canvas(
            "principal", CONVERSATION, lambda work: land_component(work, _CARD)
        )
    assert faults == [1]
    assert state.turn_components == []
    state.close_turn(final_text="", error="internal")

    assert await store.close_segment("principal", CONVERSATION, segment_id) is SegmentClose.SAVED

    reloaded = await new_store().load("principal", CONVERSATION)
    for held in (reloaded, state):
        assert [line["component_id"] for line in held.turn_components] == ["cmp_two"]
        assert [line["component_id"] for line in held.turns[-1]["cards"]] == ["cmp_two"]
        assert held.turns[-1]["error"] == "internal"
    assert state.composition == reloaded.composition
    assert [block.component_id for block in reloaded.composition.blocks] == ["cmp_two"]


@pytest.mark.parametrize("turn_lease", [False, True], ids=["claim-save", "claim-turn-lease"])
async def test_a_claim_empties_the_canvas_writes_of_earlier_leases(
    new_store, pg_engine, turn_lease
) -> None:
    """Each canvas change records its write id; the next claim clears them,
    whether a claim save (`claim_segment`) or `/turn`'s `claim_turn_lease`."""
    store = new_store()
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    assert len(await _canvas_writes(pg_engine)) == 1
    await store.close_segment("principal", CONVERSATION, segment_id)
    assert len(await _canvas_writes(pg_engine)) == 1

    await store.load("principal", CONVERSATION)
    if turn_lease:
        await store.claim_turn_lease("principal", CONVERSATION)
    else:
        store.claim_segment("principal", CONVERSATION)
        await store.save("principal", CONVERSATION)

    assert await _canvas_writes(pg_engine) == []


async def test_a_block_refresh_whose_commit_lost_its_acknowledgement_mints_one_revision(
    new_store, monkeypatch
) -> None:
    """The retry finds the block already at the revision and title, and
    leaves the canvas as the lost commit wrote it."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    assert state.composition is not None
    landed = state.composition.revision
    faults = _dropping_connection(store, acknowledged=True)

    await store.change_canvas(
        "principal",
        CONVERSATION,
        lambda work: _refresh_block(work, component_id="cmp_two", revision=2, title="Renamed"),
    )

    assert faults == [1]
    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.composition.revision == landed + 1
    assert (reloaded.composition.blocks[0].revision, reloaded.composition.blocks[0].title) == (
        2,
        "Renamed",
    )


async def test_a_removal_that_cannot_reach_the_database_ends_the_stream_as_lost(
    new_store, clock
) -> None:
    """No close could be saved while the database is down. The removal wakes
    the stream's heartbeat (`renew_lease_now`), whose renewal past the
    lease's deadline ends the stream as lost, with the turn as stored."""
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    await clock.set(store.lease_deadline("principal", CONVERSATION))

    async def dropped(*args: Any, **kwargs: Any) -> None:
        raise _connection_reset("UPDATE")

    store._write_composition = dropped
    state.renew_now = asyncio.Event()
    remove = studio_state.make_remove_cards(store)
    token = current_conversation_key.set(("principal", CONVERSATION))
    try:
        with pytest.raises(DatabaseUnavailable):
            await remove(component_ids=["cmp_two"], all_cards=False)
    finally:
        current_conversation_key.reset(token)

    assert state.renew_now.is_set()


async def test_a_removal_fenced_out_by_a_takeover_ends_the_stream(new_store, clock) -> None:
    """`remove_cards` runs inside a model's tool call, where a raise would
    reach the model as a failed call. A removal another instance's claim
    fenced out wakes the stream's heartbeat (`renew_lease_now`), whose
    renewal finds the lease lost and ends the stream, before it propagates."""
    store_a = new_store(clock=clock)
    state_a = await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    await store_a.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    store_b = new_store(clock=clock)
    await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)
    state_a.renew_now = asyncio.Event()
    remove = studio_state.make_remove_cards(store_a)
    token = current_conversation_key.set(("principal", CONVERSATION))
    try:
        with pytest.raises(SegmentLost):
            await remove(component_ids=["cmp_two"], all_cards=False)
    finally:
        current_conversation_key.reset(token)

    assert state_a.renew_now.is_set()


async def test_a_composition_edit_reads_the_digest_entries_its_adds_name_alone(new_store) -> None:
    """A pin, move or remove reads no digest entry under the row lock, and an
    add reads the entries of the cards it names."""
    store = new_store()
    await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    for component_id in ("cmp_two", "cmp_three"):
        card = {**_CARD, "component_id": component_id}
        await store.change_canvas(
            "principal", CONVERSATION, lambda work, card=card: land_component(work, card)
        )
    read: list[dict[str, Any]] = []

    def mutate(manifest: CompositionManifest | None, components: Any) -> CompositionManifest:
        read.append(dict(components))
        assert manifest is not None
        return manifest

    await store.mutate_composition("principal", CONVERSATION, mutate)
    await store.mutate_composition(
        "principal", CONVERSATION, mutate, component_ids={"cmp_three", "cmp_gone"}
    )

    assert [sorted(components) for components in read] == [[], ["cmp_three"]]


async def test_mutate_composition_no_op_returns_the_same_manifest_without_appending(
    new_store,
) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)
    state = store.current("principal", CONVERSATION)
    await store.mutate_composition("principal", CONVERSATION, _one_block_manifest)

    real_write = store._write_composition
    calls = 0

    async def counting_write(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return await real_write(*args, **kwargs)

    store._write_composition = counting_write

    result = await store.mutate_composition(
        "principal", CONVERSATION, lambda manifest, _components: manifest
    )

    assert calls == 0
    assert result is state.composition


async def test_mutate_composition_refusal_returns_it_without_appending(new_store) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)
    refusal = JSONResponse(status_code=404, content={"error": "not_found"})

    result = await store.mutate_composition(
        "principal", CONVERSATION, lambda manifest, _components: refusal
    )

    assert result is refusal


# --- writes that run to the end under cancellation ---------------------------


@pytest.mark.parametrize("method", ["drop_lease", "renew"])
async def test_a_cancelled_segment_write_still_lands_then_reraises_the_cancel(
    new_store,
    method: str,
    clock,
) -> None:
    """A request's cancellation cancels the task running `drop_lease`, and
    the stream's end cancels the heartbeat running `renew`, while the write
    is opening its transaction. The write runs to the end and the caller's cancellation is
    raised once it has."""
    store = new_store(clock=clock)
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await clock.advance(30)
    reading = asyncio.Event()
    real_transaction = store._transaction

    @asynccontextmanager
    async def slow_transaction() -> Any:
        reading.set()
        await asyncio.sleep(0.05)
        async with real_transaction() as conn:
            yield conn

    store._transaction = slow_transaction
    writing = asyncio.ensure_future(getattr(store, method)("principal", CONVERSATION, segment_id))
    await reading.wait()
    writing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writing

    store._transaction = real_transaction
    reloaded = await new_store().load("principal", CONVERSATION)
    if method == "drop_lease":
        assert reloaded.segment is None
    else:
        assert reloaded.segment is not None
        assert reloaded.segment.expires_at == clock.at + studio_state.SEGMENT_TTL_S


async def test_a_write_whose_caller_is_cancelled_ends_in_the_cancellation_whatever_it_raises():
    """The heartbeat is cancelled while its renewal is still running, and
    the renewal then fails: the caller still gets its cancellation, so the
    heartbeat stops and the stream finishes unwinding."""
    started, proceed = asyncio.Event(), asyncio.Event()

    async def failing() -> None:
        started.set()
        await proceed.wait()
        raise TimeoutError("the renewal timed out")

    caller = asyncio.ensure_future(studio_store.run_to_completion(failing()))
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    proceed.set()

    with pytest.raises(asyncio.CancelledError):
        await caller


def test_a_pending_plan_saved_before_step_labels_were_kept_loads_with_none() -> None:
    entry = {"adk_id": "adk_1", "kind": "plan", "step_order": ["step_a"]}

    pending = studio_store._deserialize_pending({"int_1": entry})["int_1"]

    assert pending.step_order == ("step_a",)
    assert pending.step_labels == {}
    assert (pending.questions, pending.legacy) == ((), False)


def test_a_turns_choice_baseline_reloads_on_another_instance() -> None:
    """The labels `plan.approved` names as `changed_from` are durable, so an
    approve answered on another instance states them."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_choice_baseline = {"aid:sector": "health (CRS 120)"}
    reloaded = ConversationState(conversation_id=CONVERSATION)

    studio_store._apply_delta(reloaded, studio_store._serialize_durable(state))

    assert reloaded.turn_choice_baseline == state.turn_choice_baseline


def test_a_pending_ask_saved_before_questions_were_kept_loads_as_one_legacy_question() -> None:
    """Its flat `option_ids` become question `q1`'s options with no labels,
    and the pause is skippable as a whole."""
    entry = {"adk_id": "clarify", "kind": "elicitation", "option_ids": ["spec_a", "o2"]}

    pending = studio_store._deserialize_pending({"int_1": entry})["int_1"]

    assert pending.questions == (
        _PendingQuestion(id="q1", question="", options={"spec_a": None, "o2": None}),
    )
    assert (pending.skippable, pending.legacy) == (True, True)


def test_a_recorded_identity_keeps_the_set_its_places_were_asked_for_as():
    """A Spec over Africa's countries is recorded with Africa beside its identity, so
    a later turn's pin of it asks Search for the set again."""
    block = json.loads(
        (Path(__file__).parent / "fixtures" / "search_spec_region_members.json").read_text()
    )["spec"]
    spec_id = block["spec_id"]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_readings = [
        SearchReading.from_part(
            {"variable": "x", "status": "definite", "spec_id": spec_id}
        ).model_dump(mode="json")
    ]
    state.turn_specs = {spec_id: block}

    recorded = state.recorded_identity(spec_id)

    assert recorded is not None
    assert [(g.role, g.label, g.ref) for g in recorded.groups] == [
        ("observationAbout", "Africa", "africa")
    ]


def test_a_plan_retrieving_a_sets_places_records_the_set_into_the_next_turn():
    """A plan whose source binds Africa's countries keeps Africa on its plan record
    (`PlanSource.groups`) and on the identity a later turn's pin resolves, through
    the turn's close."""
    block = json.loads(
        (Path(__file__).parent / "fixtures" / "search_spec_region_members.json").read_text()
    )["spec"]
    spec_id = block["spec_id"]
    plan_json = _plan_json_retrieving(source_from_spec_block(block))
    (source,) = plan_json["sources"]
    assert source["groups"] == [{"role": "observationAbout", "label": "Africa", "ref": "africa"}]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_plans = [plan_json]

    state.close_turn(final_text="Done.")
    state.begin_turn()

    recorded = state.recorded_identity(spec_id)
    assert recorded is not None
    assert [(g.role, g.label, g.ref) for g in recorded.groups] == [
        ("observationAbout", "Africa", "africa")
    ]


@pytest.mark.parametrize("corrupt", [False, True], ids=["verified", "hash-mismatch"])
def test_translator_only_plan_preserves_verified_sibling_definitions(corrupt: bool):
    block = json.loads(
        (
            Path(__file__).parents[2] / "data-mcp/tests/fixtures/resolve_aid_france_kenya.json"
        ).read_text()
    )["parts"][0]["spec"]
    if corrupt:
        block["resolved"]["alternative_measures"][0]["definition"]["identity"] = {}
    sibling = block["resolved"]["alternative_measures"][0]["definition"]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turn_plans = [_plan_json_retrieving(source_from_spec_block(block))]

    assert state.recorded_identity(block["spec_id"]) is not None
    assert (state.recorded_identity(sibling["spec_id"]) is not None) is not corrupt
    state.close_turn(final_text="Done.")
    state.begin_turn()
    assert (state.recorded_identity(sibling["spec_id"]) is not None) is not corrupt
    if not corrupt:
        assert state.recorded_identity(sibling["spec_id"]).identity == sibling["identity"]


@pytest.mark.parametrize("origin", ["search", "translator"])
def test_source_constraints_survive_a_saved_alternative_definition(origin):
    block = json.loads(
        (
            Path(__file__).parents[2] / "data-mcp/tests/fixtures/resolve_aid_france_kenya.json"
        ).read_text()
    )["parts"][0]["spec"]
    block["_resolved_sources"] = [{"text": "DAC", "ref": None, "polarity": "include"}]
    source = source_from_spec_block(block)
    sibling = block["resolved"]["alternative_measures"][0]["definition"]["spec_id"]
    state = ConversationState(conversation_id=CONVERSATION)
    if origin == "search":
        state.turn_specs = {source.spec_id: block}
    else:
        state.turn_plans = [_plan_json_retrieving(source)]
    assert state.recorded_identity(sibling).resolved_sources == source.resolved_sources
    state.close_turn(final_text="Done.")
    state.begin_turn()
    assert state.recorded_identity(sibling).resolved_sources == source.resolved_sources


# --- the `studio.turn` line and outcome of each turn close ------------------


def _turn_log_lines(caplog, status: str | None = "closed") -> list[dict[str, Any]]:
    """The JSON fields of every `studio.turn` line logged with `status`, a
    close's by default and every line with `None`."""
    return [
        fields
        for record in caplog.records
        if isinstance(fields := getattr(record, JSON_FIELDS, None), dict)
        and fields.get("event") == "studio.turn"
        and status in (None, fields.get("status"))
    ]


def _slow_commit(store: StudioStore) -> tuple[asyncio.Event, asyncio.Event]:
    """Make the next transaction `store` opens wait, once its statements
    ran, until the second event is set before it commits. The first event is
    set when it starts waiting."""
    real_transaction = store._transaction
    committing, proceed = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def slow() -> Any:
        async with real_transaction() as conn:
            yield conn
            if not committing.is_set():
                committing.set()
                await proceed.wait()

    store._transaction = slow
    return committing, proceed


@pytest.mark.parametrize("write", ["save", "close_segment"])
async def test_a_closed_turn_writes_its_outcome_once_its_write_commits(
    new_store, monkeypatch, write
) -> None:
    """`close_turn` writes nothing. The write that commits the record, a
    save or a segment's closing write, writes one outcome and starts its
    export, however many paths closed the turn and however many saves
    follow, carrying the turn's question and answer and no error."""
    written = _spy_outcomes(monkeypatch)
    flushes: list[str] = []
    monkeypatch.setattr(studio_store, "start_telemetry_flush", lambda: flushes.append("flush"))
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = (
        store.claim_segment("principal", CONVERSATION) if write == "close_segment" else None
    )
    state.begin_turn(TurnUsage(account=_ACCOUNT, surface="studio"))
    state.turn_user_message = "life expectancy in Kenya"
    if segment_id is not None:
        await store.save("principal", CONVERSATION)
    state.close_turn(final_text="done")
    state.close_turn(final_text="done again")  # the resume protocol can call this twice
    assert written == []

    if segment_id is None:
        assert await store.save("principal", CONVERSATION) is True
    else:
        assert (
            await store.close_segment("principal", CONVERSATION, segment_id) is SegmentClose.SAVED
        )
    assert await store.save("principal", CONVERSATION) is False
    assert flushes == ["flush"]

    assert written == [
        {
            "trace_id": state.trace_id,
            "session_id": CONVERSATION,
            "user_id": _ACCOUNT,
            "tags": ["studio"],
            "request": "life expectancy in Kenya",
            "response": "done",
            "error": None,
            "paused": False,
        }
    ]


def test_a_closed_line_s_reply_joins_only_the_parts_its_record_holds() -> None:
    """A record with no answer gives the code's notes alone as the reply,
    with no blank paragraph before them."""
    line = studio_store._closed_line(
        {
            "trace_id": "tr_1",
            "user": "q",
            "assistant": "",
            "match_notes": "notes",
            "basis_line": "b",
        }
    )
    assert line.response == "notes\n\nb"


async def test_an_abandoned_turn_writes_its_outcome_on_its_own_trace(
    new_store, monkeypatch
) -> None:
    """A turn that never reached a terminal frame is closed by the next
    turn's `begin_turn`, whose claim save writes its outcome on the old
    turn's trace id although it mints the new one, at the level a newer
    turn's close takes."""
    written = _spy_outcomes(monkeypatch)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()  # turn 1, never closed
    turn_1_trace_id = state.trace_id
    await store.save("principal", CONVERSATION)
    work = store.draft("principal", CONVERSATION)
    work.begin_turn()  # turn 2: closes turn 1 as abandoned
    await store.save("principal", CONVERSATION, work)

    assert [(w["trace_id"], w["error"]) for w in written] == [(turn_1_trace_id, "abandoned")]
    assert state.trace_id != turn_1_trace_id


@pytest.mark.parametrize("closed_by", ["close_turn", "begin_turn"])
async def test_a_turn_whose_record_build_raises_keeps_a_minimal_record(
    new_store, monkeypatch, caplog, closed_by
):
    """A record build that raises, in `close_turn` or in the next
    `begin_turn`'s abandoned close, keeps a minimal record naming no sources
    and the turn's trace id, whose save writes its line and outcome, and
    leaves nothing open, so the turn is built once and the conversation goes
    on. The record keeps the close's own code: none for a clean close,
    `abandoned` for the abandoned one."""
    written = _spy_outcomes(monkeypatch)
    builds: list[str] = []

    def fail(*_args, **_kwargs):
        builds.append("built")
        raise RuntimeError("bad plan record")

    monkeypatch.setattr(studio_state, "_turn_identities", fail)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    state.turn_user_message = "aid to kenya"
    turn_1_trace_id = state.trace_id

    with caplog.at_level(logging.INFO):
        if closed_by == "close_turn":
            state.close_turn(final_text="done")
        state.begin_turn()
        await store.save("principal", CONVERSATION)

    error = None if closed_by == "close_turn" else "abandoned"
    assistant = "done" if closed_by == "close_turn" else ""
    stored = await new_store().load("principal", CONVERSATION)
    record = stored.turns[0]
    assert {key: record.get(key) for key in ("turn", "user", "assistant", "error", "trace_id")} == {
        "turn": 1,
        "user": "aid to kenya",
        "assistant": assistant,
        "error": error,
        "trace_id": turn_1_trace_id,
    }
    assert set(record) == {"turn", "user", "assistant", "trace_id", "usage"} | (
        {"error"} if error else set()
    )
    assert builds == ["built"]
    assert [(w["trace_id"], w["error"]) for w in written] == [(turn_1_trace_id, error)]
    assert [(line["trace_id"], line["error"]) for line in _turn_log_lines(caplog)] == [
        (turn_1_trace_id, error)
    ]
    (warning,) = [r for r in caplog.records if "could not be built" in r.getMessage()]
    assert warning.exc_info is not None
    assert warning.exc_info[0] is RuntimeError


async def test_a_closed_turn_logs_one_studio_turn_line_once_its_closing_write_commits(
    new_store, monkeypatch, caplog
) -> None:
    """Closing a turn logs nothing. The closing write that commits its record
    writes one `studio.turn` line, however many paths close the turn and
    however many saves follow, a save rewriting the stored record included,
    naming who asked, when the turn began, the deployed revision and each
    spec used once."""
    monkeypatch.setenv("K_REVISION", "dfo-dev-ai-00042-abc")
    caller = "conversation:" + "c" * 32
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn(TurnUsage(caller=caller, account=_ACCOUNT, visitor=_VISITOR))
    await store.save("principal", CONVERSATION)
    for use, spec_id in [("fetched", "spec_a"), ("fetched", "spec_a"), ("drawn", "spec_b")]:
        state.note_spec_use(use, spec_id)

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        state.close_turn(final_text="done", error="cancelled")
        state.close_turn(final_text="done again")
        assert _turn_log_lines(caplog, status=None) == []
        await store.close_segment("principal", CONVERSATION, segment_id)
        await store.save("principal", CONVERSATION)
        # A write that rewrites a record the row already holds closes nothing.
        state.turns[-1] = {**state.turns[-1], "assistant": "rewritten"}
        assert await store.save("principal", CONVERSATION) is False

    [line] = _turn_log_lines(caplog, status=None)
    started_at = datetime.fromisoformat(line.pop("started_at"))
    assert started_at.tzinfo is not None
    assert line == {
        "event": "studio.turn",
        "trace_id": state.trace_id,
        "status": "closed",
        "caller": caller,
        "account": _ACCOUNT,
        "visitor": _VISITOR,
        "surface": None,
        "release": "dfo-dev-ai-00042-abc",
        "error": "cancelled",
        "fetched_specs": ["spec_a"],
        "drawn_specs": ["spec_b"],
    }


async def test_a_pause_line_comes_from_the_pause_alone_and_a_close_beside_it_writes_its_own(
    new_store, monkeypatch, caplog
) -> None:
    """A Studio write that stores an interrupt the turn waits on writes no
    `paused` line: the turn's cursor write does, through
    `write_pause_line`, with the text streamed so far. The closing write of
    a turn that closes with an interrupt stored writes the close's line
    alone."""
    written = _spy_outcomes(monkeypatch)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    state.turn_user_message = "aid to kenya"
    await store.save("principal", CONVERSATION)

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        state.turn_final_text = "Here is a plan."
        state.pending["int_1"] = _pending()
        await store.save("principal", CONVERSATION)
        assert _turn_log_lines(caplog, status=None) == []
        studio_store.write_pause_line(CONVERSATION, state)
        state.pending["int_2"] = _pending()
        state.close_turn(final_text="done")
        await store.close_segment("principal", CONVERSATION, segment_id)

    lines = _turn_log_lines(caplog, status=None)
    assert [(line["status"], line["trace_id"], line["error"]) for line in lines] == [
        ("paused", state.trace_id, None),
        ("closed", state.trace_id, None),
    ]
    assert [(w["paused"], w["response"]) for w in written] == [
        (True, "Here is a plan."),
        (False, "done"),
    ]


async def test_a_save_whose_write_raises_logs_no_line(new_store, monkeypatch, caplog) -> None:
    """A save whose write raises before it commits commits nothing, so the
    turn it closes logs no line and writes no outcome."""
    written = _spy_outcomes(monkeypatch)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)

    async def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("database unavailable")

    store._write = unavailable
    state.close_turn(final_text="done")
    with (
        caplog.at_level(logging.INFO, logger="harness.studio_store"),
        pytest.raises(RuntimeError, match="database unavailable"),
    ):
        await store.save("principal", CONVERSATION)

    assert _turn_log_lines(caplog, status=None) == []
    assert written == []


@pytest.mark.parametrize("cancelled_by", ["caller", "close-timeout"])
async def test_a_write_cancelled_during_its_commit_still_logs_the_record_it_commits(
    new_store, caplog, monkeypatch, cancelled_by
) -> None:
    """A request cancelled while its save commits, and a closing write whose
    timeout fires while it commits, still land the record and write its one
    line in the task that commits it. The request then ends in the
    cancellation, and the closing write returns as saved. The row holds the
    record the line names."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    committing, proceed = _slow_commit(store)
    state.close_turn(final_text="done")

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        if cancelled_by == "caller":
            state.segment = None
            task = asyncio.ensure_future(store.save("principal", CONVERSATION))
            await asyncio.wait_for(committing.wait(), timeout=5)
            task.cancel()
            await asyncio.sleep(0)
            proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        else:
            # A budget no clock reaches, pulled in once the write waits at its
            # COMMIT, so the timeout fires there however slow the statements.
            budget = 3600.0
            real_timeout = asyncio.timeout
            scopes: list[asyncio.Timeout] = []

            def capture(delay: float | None) -> asyncio.Timeout:
                scope = real_timeout(delay)
                if delay == budget:
                    scopes.append(scope)
                return scope

            monkeypatch.setattr(asyncio, "timeout", capture)
            closing = asyncio.ensure_future(
                store.close_segment("principal", CONVERSATION, segment_id, timeout=budget)
            )
            await asyncio.wait_for(committing.wait(), timeout=5)
            (scope,) = scopes
            scope.reschedule(asyncio.get_running_loop().time())
            while not scope.expired():  # the cancel reaches the write at COMMIT
                await asyncio.sleep(0)
            proceed.set()
            assert await asyncio.wait_for(closing, timeout=5) is SegmentClose.SAVED

    stored = await new_store().load("principal", CONVERSATION)
    assert stored is not None
    assert [(record["trace_id"], record.get("error")) for record in stored.turns] == [
        (state.trace_id, None)
    ]
    assert [(line["trace_id"], line["error"]) for line in _turn_log_lines(caplog)] == [
        (state.trace_id, None)
    ]


@pytest.mark.parametrize("acknowledged", [False, True], ids=["before-commit", "lost-ack"])
async def test_a_closing_write_that_meets_a_dropped_connection_logs_its_close_once(
    new_store, clock, caplog, acknowledged
) -> None:
    """A closing write that meets a dropped connection is not retried.
    Before its commit, it logs no line and the row holds the turn open, and
    the next turn on another instance closes it `abandoned` with the one
    line. With the commit's acknowledgement lost, the read back finds the
    write's transaction committed (`pg_xact_status`), so the close logs its
    one line then, with no second write, and the next turn finds the turn
    recorded and logs no `abandoned` line."""
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    closed_trace_id = state.trace_id
    state.close_turn(final_text="done")
    faults = _dropping_connection(store, acknowledged=acknowledged)

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        closed = await store.close_segment("principal", CONVERSATION, segment_id)
        at_close = [(line["trace_id"], line["error"]) for line in _turn_log_lines(caplog)]
        await clock.advance(studio_state.SEGMENT_TTL_S + 1)
        other = new_store(clock=clock)
        await other.load("principal", CONVERSATION)
        work = other.draft("principal", CONVERSATION)
        other.claim_segment("principal", CONVERSATION, work)
        work.begin_turn()
        await other.save("principal", CONVERSATION, work)

    assert faults == [1]
    assert closed is (SegmentClose.SAVED if acknowledged else SegmentClose.FAILED)
    assert at_close == ([(closed_trace_id, None)] if acknowledged else [])
    stored = await new_store().load("principal", CONVERSATION)
    assert [(record["trace_id"], record.get("error")) for record in stored.turns] == [
        (closed_trace_id, None if acknowledged else "abandoned")
    ]
    assert [(line["trace_id"], line["error"]) for line in _turn_log_lines(caplog)] == [
        (closed_trace_id, None if acknowledged else "abandoned")
    ]


async def test_a_closing_write_whose_bookkeeping_raises_after_it_landed_returns_it_saved(
    new_store, monkeypatch, caplog
) -> None:
    """A closing write whose COMMIT landed and whose bookkeeping then raised
    returns as saved, with the close's one line: the row and the line both
    hold the close, so the stream does not end it unsaved. The live state no
    longer matches the row, so it is dropped and the next request loads the
    row. The dropped state the stream holds keeps the canvas the write
    committed, which its ending frame sends."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    await store.change_canvas("principal", CONVERSATION, lambda work: land_component(work, _CARD))
    committed_canvas = state.composition
    # The stream's copy lags the canvas the row holds.
    state.composition = None
    state.close_turn(final_text="done")

    def assign_raising(*_args: Any) -> None:
        raise RuntimeError("the bookkeeping failed")

    monkeypatch.setattr(studio_store, "_assign_changed", assign_raising)
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        closed = await store.close_segment("principal", CONVERSATION, segment_id)

    assert closed is SegmentClose.SAVED
    assert [line["trace_id"] for line in _turn_log_lines(caplog)] == [state.trace_id]
    assert store.current("principal", CONVERSATION) is None
    assert state.composition == committed_canvas is not None
    stored = await new_store().load("principal", CONVERSATION)
    assert [record["trace_id"] for record in stored.turns] == [state.trace_id]


@pytest.mark.parametrize(
    ("acknowledged", "between", "result"),
    [
        pytest.param(False, "another-write", SegmentClose.FAILED, id="rolled-back-then-written"),
        pytest.param(True, "another-write", SegmentClose.SAVED, id="landed-then-written"),
        pytest.param(True, "in-progress", SegmentClose.SAVED, id="landed-in-progress"),
        pytest.param(
            True, "unreadable", SegmentClose.SAVED, id="landed-status-unreadable-at-first"
        ),
    ],
)
async def test_a_closing_write_is_read_back_by_its_own_transaction(
    new_store, pg_engine, monkeypatch, caplog, acknowledged, between, result
) -> None:
    """The read back of a closing write whose COMMIT raised asks Postgres
    whether the write's own transaction committed. A write by another
    instance that moves the row to the version the close would have set
    does not make a rolled-back close count as landed, nor one past it a
    landed close as lost. A COMMIT the server is still finishing is read
    again until it settles, and so is one whose status reads fail at first,
    as in the outage that broke the COMMIT. A close that landed logs its one
    line, and one that did not logs none."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    state.close_turn(final_text="done")
    faults = _dropping_connection(store, acknowledged=acknowledged)
    real_read_back = store._transaction_committed
    real_status = store._xact_status
    statuses: list[str | None] = []

    async def read_back(*args: Any) -> bool:
        if between == "another-write":
            async with pg_engine.begin() as conn:
                await conn.execute(
                    text(
                        "UPDATE conversations SET version = version + 1 "
                        "WHERE conversation_id = :conversation_id"
                    ),
                    {"conversation_id": CONVERSATION},
                )
        return await real_read_back(*args)

    async def status(xact: str) -> str | None:
        statuses.append("in progress" if between == "in-progress" and len(statuses) < 2 else None)
        return statuses[-1] or await real_status(xact)

    async def unreadable(xact: str) -> str | None:
        statuses.append("raised")
        if len(statuses) < 3:
            raise _connection_reset("SELECT")
        return await real_status(xact)

    monkeypatch.setattr(store, "_transaction_committed", read_back)
    if between == "in-progress":
        monkeypatch.setattr(store, "_xact_status", status)
    elif between == "unreadable":
        monkeypatch.setattr(store, "_xact_status", unreadable)
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        closed = await store.close_segment("principal", CONVERSATION, segment_id)

    assert faults == [1]
    assert closed is result
    assert [line["trace_id"] for line in _turn_log_lines(caplog)] == (
        [state.trace_id] if result is SegmentClose.SAVED else []
    )


class _StatusReadEscaped(BaseException):
    """A status read that no bound reached."""


@pytest.mark.parametrize("read", ["cancellation-turned-into-error", "too-old"])
async def test_a_status_read_back_that_cannot_tell_returns_false_and_logs_it(
    caplog, read: str
) -> None:
    """A status read back returns `False`, with one ERROR record naming the
    write, when it cannot tell whether the COMMIT landed. A read cancelled
    at its bound that raises another error in place of the cancellation, as
    a driver rolling back a broken connection does, counts as unknown, and
    the read back still ends at its limit: every read is bounded, so none
    runs past it. A status too old to read (`None`) ends it at once."""
    loop = asyncio.get_running_loop()
    started = loop.time()

    async def status() -> str | None:
        if read == "too-old":
            return None
        try:
            await asyncio.sleep(max(started + 2 - loop.time(), 0))
        except asyncio.CancelledError:
            raise asyncpg.InterfaceError("cannot perform operation") from None
        raise _StatusReadEscaped

    with caplog.at_level(logging.ERROR, logger="harness.db.commit"):
        assert not await transaction_committed(status, what="a write", limit_s=0.3)

    assert loop.time() - started < 1
    [record] = [r for r in caplog.records if r.name == "harness.db.commit"]
    assert record.levelno == logging.ERROR
    assert "a write" in record.getMessage()


class _LostAck:
    """An engine whose next `begin` commits and then raises as a lost
    acknowledgement does. Everything else is `engine`'s."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._first = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def begin(self) -> Any:
        if not self._first:
            return self._engine.begin()
        self._first = False
        return self._lost()

    @asynccontextmanager
    async def _lost(self) -> Any:
        async with self._engine.begin() as conn:
            yield conn
        raise _connection_reset("COMMIT")


async def test_a_close_and_a_pause_read_a_raised_commit_back_for_their_own_limits(
    new_store, pg_engine, monkeypatch, caplog
) -> None:
    """A pause's read back of a COMMIT that raised gives up sooner than a
    closing write's (`PAUSE_READ_BACK_LIMIT_S`, `COMMIT_READ_BACK_LIMIT_S`,
    here 0.1 s and 1 s). Both COMMITs land with their acknowledgement lost
    and their status reads `in progress` for 0.4 s: the pause raises and
    runs no `committed`, and the close logs its one line and returns as
    saved."""
    monkeypatch.setattr(turn_store_module, "PAUSE_READ_BACK_LIMIT_S", 0.1)
    monkeypatch.setattr(studio_store, "COMMIT_READ_BACK_LIMIT_S", 1.0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    turns = TurnStore(pg_engine)
    cursor = await turns.open_turn(
        "principal",
        CONVERSATION,
        segment_id=segment_id,
        turn_index=state.turn_index,
        node="ask",
        inputs={},
        state={},
    )
    loop = asyncio.get_running_loop()

    def settling(real: Callable[[str], Any]) -> Callable[[str], Any]:
        asked: list[float] = []

        async def status(xact: str) -> str | None:
            asked.append(loop.time())
            if loop.time() - asked[0] < 0.4:
                return "in progress"
            return await real(xact)

        return status

    turns._engine = _LostAck(pg_engine)  # type: ignore[assignment]
    monkeypatch.setattr(turns, "_xact_status", settling(turns._xact_status))
    noted: list[Any] = []
    with pytest.raises(DBAPIError):
        await turns.pause(
            cursor,
            segment_id=segment_id,
            node="ask",
            node_input={},
            kind="ask",
            interrupt_id="int_1",
            payload={},
            state={},
            lineage=None,
            committed=noted.append,
        )
    assert noted == []
    assert (await TurnStore(pg_engine).turn(cursor.id)).status is TurnStatus.PAUSED

    state.close_turn(final_text="done")
    faults = _dropping_connection(store, acknowledged=True)
    monkeypatch.setattr(store, "_xact_status", settling(store._xact_status))
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        closed = await store.close_segment("principal", CONVERSATION, segment_id)

    assert faults == [1]
    assert closed is SegmentClose.SAVED
    assert [line["trace_id"] for line in _turn_log_lines(caplog)] == [state.trace_id]


@pytest.mark.parametrize("write", ["abandoning-claim-save", "cancel-save", "load-repair"])
async def test_a_write_that_records_a_turn_and_loses_its_acknowledgement_logs_it_once(
    new_store, clock, pg_engine, caplog, write
) -> None:
    """A write that records a turn closed, whose COMMIT lands with its
    acknowledgement lost, is confirmed by its transaction's status: a claim
    save that closes the open turn `abandoned`, a `/cancel` save that closes
    it `cancelled`, and `load`'s record of a turn whose cursor closed. The
    call returns as for a COMMIT that returned, with the turn's one line,
    and a fresh instance's load finds the turn recorded and writes none."""
    caplog.set_level(logging.INFO, logger="harness.studio_store")
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    closed_trace_id = state.trace_id
    await store.save("principal", CONVERSATION)

    if write == "abandoning-claim-save":
        await clock.advance(studio_state.SEGMENT_TTL_S + 1)
        other = new_store(clock=clock)
        await other.load("principal", CONVERSATION)
        work = other.draft("principal", CONVERSATION)
        other.claim_segment("principal", CONVERSATION, work)
        work.begin_turn()
        faults = _dropping_connection(other, acknowledged=True)
        assert await other.save("principal", CONVERSATION, work)
        error = "abandoned"
    elif write == "cancel-save":
        work = store.draft("principal", CONVERSATION)
        work.close_turn(final_text="", error=ErrorCode.CANCELLED.value)
        faults = _dropping_connection(store, acknowledged=True)
        assert await store.save("principal", CONVERSATION, work)
        error = ErrorCode.CANCELLED.value
    else:
        turns = TurnStore(pg_engine)
        opened = await turns.open_turn(
            "principal",
            CONVERSATION,
            segment_id=segment_id,
            turn_index=1,
            node="seed",
            inputs={},
            state={},
        )
        await turns.finish(opened, segment_id=segment_id, status=TurnStatus.CANCELLED)
        store.evict("principal", CONVERSATION)
        await clock.advance(studio_state.SEGMENT_TTL_S + 1)
        other = new_store(clock=clock)
        faults = _dropping_connection(other, acknowledged=True)
        loaded = await other.load("principal", CONVERSATION)
        assert loaded is not None and loaded.turn_recorded
        error = ErrorCode.CANCELLED.value

    assert faults == [1]
    assert [(line["trace_id"], line["error"]) for line in _turn_log_lines(caplog)] == [
        (closed_trace_id, error)
    ]
    fresh = await new_store(clock=clock).load("principal", CONVERSATION)
    assert fresh is not None
    assert [record["trace_id"] for record in fresh.turns] == [closed_trace_id]
    assert len(_turn_log_lines(caplog)) == 1


async def test_a_closing_write_whose_commit_raises_after_its_timeout_still_reads_back(
    new_store, caplog
) -> None:
    """A closing write whose COMMIT lands and then raises after the caller's
    timeout fired is still read back, in the COMMIT's own task, so the
    landed close logs its one line and returns as saved."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    state.close_turn(final_text="done")
    real_transaction = store._transaction

    @asynccontextmanager
    async def late_lost_ack() -> Any:
        store._transaction = real_transaction
        async with real_transaction() as conn:
            yield conn
        await asyncio.sleep(0.3)
        raise _connection_reset("COMMIT")

    store._transaction = late_lost_ack
    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        closed = await store.close_segment("principal", CONVERSATION, segment_id, timeout=0.1)

    assert closed is SegmentClose.SAVED
    assert [line["trace_id"] for line in _turn_log_lines(caplog)] == [state.trace_id]


async def test_a_load_that_read_before_a_lease_claim_committed_keeps_the_claim(
    new_store, monkeypatch
) -> None:
    """A load whose read of the row began before this instance's lease claim
    committed copies the row into the live state before the claim commits,
    so the live state ends naming the lease the instance holds."""
    store = new_store()
    await store.create("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    real_read = store._read
    read, resume = asyncio.Event(), asyncio.Event()

    async def read_then_wait(*args: Any) -> Any:
        stored = await real_read(*args)
        read.set()
        await resume.wait()
        return stored

    monkeypatch.setattr(store, "_read", read_then_wait)
    loading = asyncio.ensure_future(store.load("principal", CONVERSATION))
    await read.wait()
    monkeypatch.setattr(store, "_read", real_read)
    lock = store._runtime[("principal", CONVERSATION)].write_lock
    real_acquire = lock.acquire
    asked = asyncio.Event()

    async def acquire() -> bool:
        asked.set()
        return await real_acquire()

    monkeypatch.setattr(lock, "acquire", acquire)
    claiming = asyncio.ensure_future(store.claim_turn_lease("principal", CONVERSATION))
    # The claim waits on the write lock the load holds.
    await asked.wait()
    resume.set()
    await loading
    lease = await claiming

    live = store.current("principal", CONVERSATION)
    assert live is not None and live.segment is not None
    assert live.segment.id == lease.segment_id
    assert store._runtime[("principal", CONVERSATION)].held == lease.segment_id


async def test_a_cancelled_write_that_commits_keeps_its_bookkeeping(new_store) -> None:
    """A cancel of a save while it commits lets the commit land with the
    bookkeeping of what it committed, so the next save writes nothing it
    already wrote."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)
    state.turn_final_text = "streamed so far"
    committing, proceed = _slow_commit(store)

    writing = asyncio.ensure_future(store.save("principal", CONVERSATION))
    await asyncio.wait_for(committing.wait(), timeout=5)
    writing.cancel()
    await asyncio.sleep(0)
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await writing

    real_write = store._write
    deltas: list[set[str]] = []

    async def spying(conn: Any, pk: int, delta: dict[str, Any], **kwargs: Any) -> Any:
        deltas.append(set(delta))
        return await real_write(conn, pk, delta, **kwargs)

    store._write = spying
    assert await store.save("principal", CONVERSATION) is False

    assert deltas == []
    stored = await new_store().load("principal", CONVERSATION)
    assert stored.turn_final_text == "streamed so far"


@pytest.mark.parametrize("failure", ["write-error", "taken-over"])
async def test_a_turn_the_store_never_saw_closed_logs_once_as_abandoned_under_its_trace_id(
    new_store, clock, monkeypatch, caplog, failure
) -> None:
    """A close whose closing write fails, on a write error or because another
    instance took the segment over, logs no line, and the row still holds
    the turn open. The next turn, on another instance, closes it as
    abandoned, writing the one line under the turn's trace id and caller."""
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    first, second = "conversation:" + "1" * 32, "conversation:" + "2" * 32
    state = await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    state.begin_turn(TurnUsage(caller=first))
    closed_trace_id = state.trace_id
    await store_a.save("principal", CONVERSATION)

    async def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("database unavailable")

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        if failure == "taken-over":
            await clock.advance(studio_state.SEGMENT_TTL_S + 1)
            await store_b.load("principal", CONVERSATION)
            store_b.claim_segment("principal", CONVERSATION)
            await store_b.save("principal", CONVERSATION)
        else:
            store_a._write = unavailable
        state.close_turn(final_text="done")
        closed = await store_a.close_segment("principal", CONVERSATION, segment_id)
        assert closed is (SegmentClose.LOST if failure == "taken-over" else SegmentClose.FAILED)
        assert _turn_log_lines(caplog, status=None) == []

        await clock.advance(studio_state.SEGMENT_TTL_S + 1)
        later = await store_b.load("principal", CONVERSATION)
        assert later is not None
        work = store_b.draft("principal", CONVERSATION)
        store_b.claim_segment("principal", CONVERSATION, work)
        work.begin_turn(TurnUsage(caller=second))
        await store_b.save("principal", CONVERSATION, work)

    assert [
        (line["trace_id"], line["caller"], line["error"]) for line in _turn_log_lines(caplog)
    ] == [(closed_trace_id, first, "abandoned")]
    assert later.trace_id != closed_trace_id


async def test_two_claims_over_an_open_turn_log_one_abandoned_line(new_store, caplog) -> None:
    """Two instances load a conversation whose turn is open and both claim
    the next turn, each closing the open one as abandoned. The claim whose
    save lands writes the one line. The stale one commits nothing and writes
    none."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    open_trace_id = state.trace_id
    await store.save("principal", CONVERSATION)
    store_a, store_b = new_store(), new_store()
    await store_a.load("principal", CONVERSATION)
    await store_b.load("principal", CONVERSATION)

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        claim_a = store_a.draft("principal", CONVERSATION)
        claim_b = store_b.draft("principal", CONVERSATION)
        claim_a.begin_turn()
        claim_b.begin_turn()
        await store_a.save("principal", CONVERSATION, claim_a)
        with pytest.raises(ConversationStale):
            await store_b.save("principal", CONVERSATION, claim_b)

    assert [(line["trace_id"], line["error"]) for line in _turn_log_lines(caplog)] == [
        (open_trace_id, "abandoned")
    ]


async def test_a_turn_loaded_from_a_row_with_no_usage_keeps_the_identity_it_takes(
    new_store, pg_engine, caplog
) -> None:
    """A paused turn whose row stores no `turn_usage` holds a usage of its
    own once loaded, so the account `take_identity` fills in survives the
    claim save and reaches the turn's line."""
    store = new_store()
    paused = await store.create("principal", CONVERSATION)
    paused.begin_turn()
    await store.save("principal", CONVERSATION)
    async with pg_engine.begin() as conn:
        await conn.execute(text("UPDATE conversations SET turn_state = turn_state - 'turn_usage'"))
    other = new_store()
    state = await other.load("principal", CONVERSATION)
    assert state.turn_usage == TurnUsage(trace_id=paused.trace_id)
    work = other.draft("principal", CONVERSATION)
    segment_id = other.claim_segment("principal", CONVERSATION, work)
    work.take_identity(account=_ACCOUNT, visitor=None, caller="conversation:" + "c" * 32)
    await other.save("principal", CONVERSATION, work)

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        state.close_turn(final_text="done")
        await other.close_segment("principal", CONVERSATION, segment_id)

    assert [(line["trace_id"], line["account"]) for line in _turn_log_lines(caplog)] == [
        (paused.trace_id, _ACCOUNT)
    ]


@pytest.mark.parametrize(
    ("stored_trace_id", "caller", "started_at"),
    [
        ("tr_current", "conversation:" + "c" * 32, "2026-11-19T10:00:00+00:00"),
        ("tr_earlier", None, None),
    ],
    ids=["this-turn", "another-turn"],
)
def test_a_stored_usage_reads_only_for_the_turn_it_names(
    stored_trace_id, caller, started_at
) -> None:
    """A row whose usage names another turn's trace id reads as unset."""
    stored = {
        "trace_id": "tr_current",
        "turn_usage": {
            "trace_id": stored_trace_id,
            "started_at": "2026-11-19T10:00:00+00:00",
            "caller": "conversation:" + "c" * 32,
        },
    }
    state = ConversationState(conversation_id=CONVERSATION)

    studio_store._apply_delta(state, stored)

    assert state.turn_usage == TurnUsage(
        trace_id="tr_current", caller=caller, started_at=started_at
    )


async def test_a_reload_keeps_the_specs_a_turn_this_instance_holds_noted_since_its_save(
    new_store,
) -> None:
    """Specs the open turn noted after its last save, during a pause, survive
    a reload of the same turn, after the ones the row holds."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn(TurnUsage(fetched_specs=["spec_a"]))
    await store.save("principal", CONVERSATION)
    state.note_spec_use("fetched", "spec_b")
    state.note_spec_use("drawn", "spec_b")

    reloaded = await store.load("principal", CONVERSATION)

    assert reloaded is state
    assert reloaded.turn_usage.fetched_specs == ["spec_a", "spec_b"]
    assert reloaded.turn_usage.drawn_specs == ["spec_b"]


async def test_a_spec_noted_on_a_working_copy_reaches_the_live_turn_only_once_its_save_commits(
    new_store,
) -> None:
    """A working copy (`draft`) holds a usage of its own, so what a request
    notes on it before its claim save leaves the live state as loaded until
    the save commits."""
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    await store.save("principal", CONVERSATION)

    work = store.draft("principal", CONVERSATION)
    work.take_identity(account=_ACCOUNT, visitor=None, caller=None)
    work.note_spec_use("fetched", "spec_a")
    assert state.turn_usage.account is None
    assert state.turn_usage.fetched_specs == []

    await store.save("principal", CONVERSATION, work)

    assert state.turn_usage.account == _ACCOUNT
    assert state.turn_usage.fetched_specs == ["spec_a"]
