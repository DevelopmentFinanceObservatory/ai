"""`ConversationState`'s telemetry span lifecycle: `begin_turn` opens the
turn's root span on the freshly minted `trace_id`, and `close_turn` closes
it, exactly once, however the turn ends. `harness.studio_state` only calls
through to `common.telemetry`'s `start_turn_span`/`end_turn_span`; those
functions' own backend selection and error handling are covered by
`packages/common/tests/test_telemetry.py`, so here they are faked to prove
the wiring, not the telemetry backend itself.

The canvas and conversation digests the models read carry a stated maximum
size (`CANVAS_DIGEST_MAX_BYTES`, `CONVERSATION_DIGEST_MAX_BYTES`); the tests
below feed each one oversized input and assert the maximum holds.

The canvas digest entry is pinned here too: what a card shows (its entities
and span) travels in the digest the models read, from the names the
component's own fact summary and stored intent carry.

The store (`StudioStore.create`/`load`/`save`/`checkpoint`/`release`) and
the segment lease (`claim_segment`/`renew`/`request_cancel`) are exercised
at the bottom of this file against Postgres (the `pg_engine` fixture), each
test in a schema of its own.
"""

from __future__ import annotations

import asyncio
import copy
import errno
import json
import logging
import ssl
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
from compiler.sources import source_from_spec_block
from fastapi.responses import JSONResponse
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from harness import studio_state, studio_store
from harness.approval import ApprovalDecision, ApprovalRecord
from harness.conversation import current_conversation_key
from harness.graph.state import ORCHESTRATOR_SPECS_KEY
from harness.studio_edit import _refresh_block
from harness.studio_state import (
    CANVAS_DIGEST_MAX_BYTES,
    CONVERSATION_DIGEST_MAX_BYTES,
    ConversationState,
    DatabaseUnavailable,
    Segment,
    SegmentBusy,
    SegmentLost,
    _digest_entry,
    _next_manifest,
    _PendingInterrupt,
    _PendingQuestion,
    land_component,
    make_recall_part,
    remove_cards,
)
from harness.studio_store import SEQ_BLOCK, ConversationStale, StudioStore
from harness.turn_records import PlanRecord, RecordedIdentity, SearchReading
from harness.wire import (
    AddOp,
    ComponentType,
    CompositionActor,
    CompositionBlock,
    CompositionManifest,
    MutationOp,
    PlanProposed,
)
from planner.plan import PlanStatus
from planner.workflow import SelectedSource
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.ext.asyncio import AsyncEngine

from . import drawn_cards
from .resolved_revision import resolved_revision
from .store_clock import StoreClock
from .test_turn_records import _scoped_part, _top_donors
from .turn_driver import NO_ADK

CONVERSATION = "conv_01test0000"


@pytest.mark.parametrize("method", ["begin_turn", "ensure_turn_span"])
def test_a_turn_span_opens_on_the_new_trace_id_in_the_conversations_session(monkeypatch, method):
    fake_span = MagicMock()
    started_with = []
    monkeypatch.setattr(
        studio_state,
        "start_turn_span",
        lambda trace_id, *, session_id=None: (
            started_with.append((trace_id, session_id)),
            fake_span,
        )[1],
    )

    state = ConversationState(conversation_id=CONVERSATION)
    getattr(state, method)()

    assert started_with == [(state.trace_id, CONVERSATION)]
    assert state._telemetry_span is fake_span


def test_close_turn_ends_the_span_exactly_once(monkeypatch):
    fake_span = MagicMock()
    ended = []
    monkeypatch.setattr(
        studio_state, "start_turn_span", lambda trace_id, *, session_id=None: fake_span
    )
    monkeypatch.setattr(
        studio_state,
        "end_turn_span",
        lambda handle, *, request, response, error: ended.append(
            (handle, request, response, error)
        ),
    )

    state = ConversationState(conversation_id=CONVERSATION)
    state.begin_turn()
    state.turn_user_message = "life expectancy in Kenya"
    state.close_turn(final_text="done")
    state.close_turn(final_text="done again")  # the resume protocol can call this twice

    # The span ends once, carrying the turn's question and answer and no error.
    assert ended == [(fake_span, "life expectancy in Kenya", "done", None)]
    assert state._telemetry_span is None


def test_abandoned_turn_closes_its_span_before_the_next_one_opens(monkeypatch):
    """A turn that never reached a terminal frame gets its span closed by
    the NEXT turn's `begin_turn`, before that next turn's own span opens --
    `begin_turn` mints the new `trace_id` first, but the abandoned-turn
    cleanup must still act on the span belonging to the OLD turn."""
    turn_1_span, turn_2_span = MagicMock(name="turn_1_span"), MagicMock(name="turn_2_span")
    remaining = [turn_1_span, turn_2_span]
    opened = []
    ended = []
    monkeypatch.setattr(
        studio_state,
        "start_turn_span",
        lambda trace_id, *, session_id=None: (opened.append(trace_id), remaining.pop(0))[1],
    )
    monkeypatch.setattr(
        studio_state,
        "end_turn_span",
        lambda handle, *, error, **_: ended.append((handle, error)),
    )

    state = ConversationState(conversation_id=CONVERSATION)
    state.begin_turn()  # turn 1, never closed
    turn_1_trace_id = state.trace_id
    assert state._telemetry_span is turn_1_span
    state.begin_turn()  # turn 2: triggers begin_turn's own abandoned-turn cleanup
    turn_2_trace_id = state.trace_id

    assert opened == [turn_1_trace_id, turn_2_trace_id]
    assert ended == [(turn_1_span, "abandoned")]
    assert state._telemetry_span is turn_2_span
    assert state.turns[-1] == {"user": "", "assistant": "", "error": "abandoned"}


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
    digest = state.canvas_digest_state()
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
            state.canvas_digest_state(),
            state.canvas_digest_state(focus=card_id),
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


def _canvas(entries: dict[str, dict[str, Any]]) -> ConversationState:
    """A conversation whose composition holds every entry, in order."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.components.update(entries)
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

    digest = state.canvas_digest_state()

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

    digest = state.canvas_digest_state()

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

    digest = state.canvas_digest_state()

    assert digest is not None
    title = digest["blocks"][0]["title"]
    assert title.endswith(studio_state._CUT_MARKER)
    assert title.startswith("ttt")
    assert _serialised(title) <= studio_state._TITLE_BYTE_CAP


def test_a_title_of_escaped_characters_is_cut_by_its_serialised_size():
    state = _canvas({"cmp_01": _entry(ASTRAL * 1_000, _fact_summary())})

    digest = state.canvas_digest_state()

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

    digest = state.canvas_digest_state()

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

    digest = state.canvas_digest_state()

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

    digest = state.canvas_digest_state()

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

        digest = state.canvas_digest_state()

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

    digest = state.canvas_digest_state()

    assert digest is not None
    kept, cut = digest["blocks"]
    assert kept["fact_summary"] == at_cap
    assert set(cut["fact_summary"]) == {"cut"}


def test_the_canvas_digest_stays_under_its_maximum_with_every_field_at_its_cap():
    title = "t" * (studio_state._TITLE_BYTE_CAP - 2)
    summary = "s" * (studio_state._FACT_SUMMARY_BYTE_CAP - 2)
    shows = _shows(
        "n" * (studio_state._DIGEST_NAME_BYTE_CAP - 2),
        "y" * (studio_state._DIGEST_TIME_BYTE_CAP - 2),
    )
    entries = {
        f"cmp_{index:058d}": _entry(title, summary, revision=9_999_999_999, shows=shows)
        for index in range(studio_state._DIGEST_BLOCK_CAP)
    }
    state = _canvas(entries)

    digest = state.canvas_digest_state(focus=next(reversed(entries)))

    assert digest is not None
    assert len(digest["blocks"]) == studio_state._DIGEST_BLOCK_CAP
    size = _serialised(digest)
    assert size <= CANVAS_DIGEST_MAX_BYTES
    assert size > CANVAS_DIGEST_MAX_BYTES * 0.9  # the maximum is within reach


def test_the_canvas_digest_stays_under_its_maximum_however_large_the_inputs():
    entries = {
        f"cmp_{index:016d}": _entry(
            ASTRAL * BIG,
            {"series": ["x" * BIG]},
            shows=_shows(ASTRAL * BIG, ASTRAL * BIG, count=3 * studio_state._DIGEST_SHOW_CAP),
        )
        for index in range(3 * studio_state._DIGEST_BLOCK_CAP)
    }
    state = _canvas(entries)

    digest = state.canvas_digest_state()

    assert digest is not None
    assert len(digest["blocks"]) == studio_state._DIGEST_BLOCK_CAP
    assert _serialised(digest) <= CANVAS_DIGEST_MAX_BYTES


def test_a_canvas_emptied_by_removals_is_stated_empty_in_the_digest():
    """A canvas whose cards were all removed reads as empty, not as no canvas,
    so the model is not left with the earlier turns' cards as its only view."""
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.composition = _next_manifest(state.composition, [], actor=CompositionActor.USER)

    digest = state.canvas_digest_state()
    assert digest is not None
    assert digest["blocks"] == []
    assert [card["component_id"] for card in digest["hidden"]] == ["cmp_01"]
    assert ConversationState(conversation_id=CONVERSATION).canvas_digest_state() is None


def test_the_digest_marks_a_pinned_block():
    state = _canvas({"cmp_01": _entry("Aid", None), "cmp_02": _entry("Debt", None)})
    assert state.composition is not None
    state.composition.blocks[1].pinned = True

    digest = state.canvas_digest_state()

    assert digest is not None
    assert "pinned" not in digest["blocks"][0]
    assert digest["blocks"][1]["pinned"] is True


def test_the_digest_counts_the_blocks_its_cap_leaves_out():
    cap = studio_state._DIGEST_BLOCK_CAP
    state = _canvas({f"cmp_{index:02d}": _entry("Aid", None) for index in range(cap + 3)})

    digest = state.canvas_digest_state()

    assert digest is not None
    assert len(digest["blocks"]) == cap
    assert digest["blocks_not_shown"] == 3
    one = _canvas({"cmp_01": _entry("Aid", None)}).canvas_digest_state()
    assert one is not None
    assert "blocks_not_shown" not in one


def test_the_digest_with_every_block_pinned_and_some_cut_stays_under_its_maximum():
    title = "t" * (studio_state._TITLE_BYTE_CAP - 2)
    summary = "s" * (studio_state._FACT_SUMMARY_BYTE_CAP - 2)
    shows = _shows(
        "n" * (studio_state._DIGEST_NAME_BYTE_CAP - 2),
        "y" * (studio_state._DIGEST_TIME_BYTE_CAP - 2),
    )
    entries = {
        f"cmp_{index:058d}": _entry(title, summary, revision=9_999_999_999, shows=shows)
        for index in range(studio_state._DIGEST_BLOCK_CAP + 2)
    }
    state = _canvas(entries)
    assert state.composition is not None
    for block in state.composition.blocks:
        block.pinned = True

    digest = state.canvas_digest_state(focus=next(reversed(entries)))

    assert digest is not None
    assert digest["blocks_not_shown"] == 2
    assert _serialised(digest) <= CANVAS_DIGEST_MAX_BYTES


def test_the_digest_lists_this_conversations_cards_that_are_not_on_the_canvas():
    """A card a new turn hid, or one removed, is named under `hidden`: id,
    type and title only, so the model can talk about it but not read it as
    on the canvas."""
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary())})
    state.components["cmp_old"] = _entry("Debt", _fact_summary())

    digest = state.canvas_digest_state()

    assert digest is not None
    assert [b["component_id"] for b in digest["blocks"]] == ["cmp_01"]
    assert digest["hidden"] == [{"component_id": "cmp_old", "type": "line_chart", "title": "Debt"}]


def test_a_canvas_with_only_hidden_cards_is_empty_and_names_them():
    state = _canvas({"cmp_01": _entry("Aid", None)})
    remove_cards(state, [], all_cards=True)

    digest = state.canvas_digest_state()

    assert digest == {
        "blocks": [],
        "hidden": [{"component_id": "cmp_01", "type": "line_chart", "title": "Aid"}],
    }


def test_the_hidden_list_keeps_the_newest_and_counts_the_rest():
    cap = studio_state._DIGEST_HIDDEN_CAP
    state = _canvas({"cmp_on": _entry("On", None)})
    for index in range(cap + 4):
        state.components[f"cmp_{index:02d}"] = _entry(f"Card {index}", None)

    digest = state.canvas_digest_state()

    assert digest is not None
    assert len(digest["hidden"]) == cap
    assert digest["hidden"][-1]["component_id"] == f"cmp_{cap + 3:02d}"
    assert digest["hidden_not_shown"] == 4


def test_a_focus_card_off_the_canvas_is_a_block_that_says_it_is_hidden():
    """#348 keeps the focus card in `blocks` whatever its state; a hidden one
    stays hidden, so the model does not read it as on the user's canvas."""
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.components["cmp_off"] = _entry("Off the canvas", None)

    digest = state.canvas_digest_state(focus="cmp_off")

    assert digest is not None
    assert digest["blocks"][-1]["component_id"] == "cmp_off"
    assert digest["blocks"][-1]["on_canvas"] is False
    assert "on_canvas" not in digest["blocks"][0]
    assert [card["component_id"] for card in digest["hidden"]] == ["cmp_off"]


def test_the_digest_with_hidden_cards_at_their_caps_stays_under_its_maximum():
    title = "t" * (studio_state._TITLE_BYTE_CAP - 2)
    summary = "s" * (studio_state._FACT_SUMMARY_BYTE_CAP - 2)
    shows = _shows(
        "n" * (studio_state._DIGEST_NAME_BYTE_CAP - 2),
        "y" * (studio_state._DIGEST_TIME_BYTE_CAP - 2),
    )
    entries = {
        f"cmp_{index:058d}": _entry(title, summary, revision=9_999_999_999, shows=shows)
        for index in range(studio_state._DIGEST_BLOCK_CAP + 2)
    }
    state = _canvas(entries)
    assert state.composition is not None
    for block in state.composition.blocks:
        block.pinned = True
    for index in range(studio_state._DIGEST_HIDDEN_CAP + 2):
        hidden = _entry("h" * BIG, None)
        hidden["type"] = "x" * BIG
        state.components["cmp_h" + "i" * BIG + str(index)] = hidden

    digest = state.canvas_digest_state(focus=next(reversed(entries)))

    assert digest is not None
    assert digest["hidden_not_shown"] == 2
    assert _serialised(digest) <= CANVAS_DIGEST_MAX_BYTES


def test_the_orchestrators_canvas_summary_counts_the_cards_and_names_none():
    """The always-on summary is two counts; the listing is a recall away."""
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


def test_recall_without_a_component_lists_the_canvas_and_the_hidden_cards():
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary()), "cmp_02": _entry("Debt", None)})
    assert state.composition is not None
    state.composition.blocks[0].pinned = True
    state.components["cmp_old"] = _entry("Old", _fact_summary())

    listing = state.recall_part(part="canvas")

    assert listing["part"] == "canvas"
    assert [card["component_id"] for card in listing["on_canvas"]] == ["cmp_01", "cmp_02"]
    assert listing["on_canvas"][0]["pinned"] is True
    assert [card["component_id"] for card in listing["hidden"]] == ["cmp_old"]
    assert listing["hidden"][0]["fact_summary"] is not None


def test_the_listing_names_cards_past_its_cap_compactly_so_each_can_be_found():
    """Past the newest 16 a card is still named (id, type, title), so the
    model can recall or remove it by id; past that compact cap they are
    counted."""
    full, compact = studio_state._DIGEST_BLOCK_CAP, studio_state._LISTING_COMPACT_CAP
    state = _canvas({f"cmp_{n:03d}": _entry(f"Card {n}", None) for n in range(full + compact + 2)})

    listing = state.canvas_listing_state()

    assert len(listing["on_canvas"]) == full
    assert len(listing["on_canvas_older"]) == compact
    assert set(listing["on_canvas_older"][0]) == {"component_id", "type", "title"}
    assert listing["on_canvas_older"][-1]["component_id"] == f"cmp_{compact + 1:03d}"
    assert listing["on_canvas_not_shown"] == 2


def test_the_listing_at_every_cap_stays_under_its_maximum():
    title = "t" * (studio_state._TITLE_BYTE_CAP - 2)
    summary = "s" * (studio_state._FACT_SUMMARY_BYTE_CAP - 2)
    shows = _shows(
        "n" * (studio_state._DIGEST_NAME_BYTE_CAP - 2),
        "y" * (studio_state._DIGEST_TIME_BYTE_CAP - 2),
    )
    many = studio_state._DIGEST_BLOCK_CAP + studio_state._LISTING_COMPACT_CAP + 3
    entries = {
        f"cmp_{index:058d}": _entry(title, summary, revision=9_999_999_999, shows=shows)
        for index in range(many)
    }
    state = _canvas(entries)
    assert state.composition is not None
    for block in state.composition.blocks:
        block.pinned = True
    for index in range(many):
        state.components[f"cmp_h{index:057d}"] = _entry(
            title, summary, revision=9_999_999_999, shows=shows
        )

    listing = state.canvas_listing_state()

    assert listing["hidden_not_shown"] == 3
    assert _serialised({"part": "canvas", **listing}) <= studio_state.CANVAS_LISTING_MAX_BYTES


def test_recall_says_whether_a_component_is_still_on_the_canvas():
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.components["cmp_off"] = _entry("Removed", None)

    assert state.recall_part(part="canvas", component_id="cmp_01")["on_canvas"] is True
    assert state.recall_part(part="canvas", component_id="cmp_off")["on_canvas"] is False


def test_removing_cards_takes_the_named_blocks_off_and_reports_the_rest():
    state = _canvas({f"cmp_0{n}": _entry("Aid", None) for n in (1, 2, 3)})
    before = state.composition

    removed, missing = remove_cards(state, ["cmp_02", "cmp_99"], all_cards=False)

    assert (removed, missing) == (["cmp_02"], ["cmp_99"])
    assert state.composition is not None and before is not None
    assert [b.component_id for b in state.composition.blocks] == ["cmp_01", "cmp_03"]
    assert state.composition.revision == before.revision + 1
    assert state.composition.actor == CompositionActor.MODEL
    # The stream owes one update for it, once.
    assert state.take_unreported_removal() is True
    assert state.take_unreported_removal() is False


def test_a_card_already_off_the_canvas_is_not_reported_missing():
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.components["cmp_hidden"] = _entry("Hidden", None)

    assert remove_cards(state, ["cmp_hidden"], all_cards=False) == ([], [])
    assert state.take_unreported_removal() is False


def test_removing_every_card_leaves_an_empty_canvas_the_digest_states():
    state = _canvas({f"cmp_0{n}": _entry("Aid", None) for n in (1, 2)})

    removed, missing = remove_cards(state, [], all_cards=True)

    assert (removed, missing) == (["cmp_01", "cmp_02"], [])
    digest = state.canvas_digest_state()
    assert digest is not None and digest["blocks"] == []
    assert remove_cards(state, [], all_cards=True) == ([], [])


def test_a_focus_that_names_no_stored_component_stays_out_of_the_digest():
    """`focus` is the client's `component_ref.component_id`, an unbounded string."""
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary())})

    digest = state.canvas_digest_state(focus="F" * 1_000_000)

    assert digest is not None
    assert "focus_component_id" not in digest
    assert [block["component_id"] for block in digest["blocks"]] == ["cmp_01"]
    assert _serialised(digest) <= CANVAS_DIGEST_MAX_BYTES


def test_a_focus_that_names_a_stored_component_is_named_in_the_digest():
    state = _canvas({"cmp_01": _entry("Aid", _fact_summary()), "cmp_02": _entry("Debt", None)})

    digest = state.canvas_digest_state(focus="cmp_01")

    assert digest is not None
    assert digest["focus_component_id"] == "cmp_01"


def test_a_focus_on_a_component_off_the_canvas_adds_its_block_and_names_it():
    state = _canvas({"cmp_01": _entry("Aid", None)})
    state.components["cmp_off"] = _entry("Off the canvas", None)

    digest = state.canvas_digest_state(focus="cmp_off")

    assert digest is not None
    assert digest["focus_component_id"] == "cmp_off"
    assert digest["blocks"][-1]["component_id"] == "cmp_off"


def test_a_focus_card_older_than_the_block_cap_stays_in_the_digest():
    """The card the user asks about is the one block the digest must keep."""
    entries = {f"cmp_{index:02d}": _entry(f"Card {index}", None) for index in range(17)}
    state = _canvas(entries)

    digest = state.canvas_digest_state(focus="cmp_00")

    assert digest is not None
    ids = [block["component_id"] for block in digest["blocks"]]
    assert len(ids) == studio_state._DIGEST_BLOCK_CAP
    assert "cmp_00" in ids
    assert digest["focus_component_id"] == "cmp_00"


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

    digest = state.canvas_digest_state()

    assert digest is not None
    assert _serialised(digest) <= CANVAS_DIGEST_MAX_BYTES
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


# --- the ledger-backed store: create / load / save / checkpoint / release ---
# --- and the segment lease: claim_segment / renew / request_cancel ----------


def session_turn_specs(
    service: InMemorySessionService, *, app_name: str
) -> Callable[[str, str], Awaitable[dict[str, Any]]]:
    """A `StudioStore` `turn_specs` reader over `service`: the spec blocks
    turn session `session_id` holds, as stored."""

    async def turn_specs(principal: str, session_id: str) -> dict[str, Any]:
        session = await service.get_session(
            app_name=app_name,
            user_id=principal,
            session_id=session_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        return dict((session.state.get(ORCHESTRATOR_SPECS_KEY) or {}) if session else {})

    return turn_specs


@pytest.fixture
def turn_sessions() -> InMemorySessionService:
    """The per-turn ADK sessions, where an open turn's spec blocks live."""
    return InMemorySessionService()


@pytest.fixture
def new_store(
    pg_engine: AsyncEngine, turn_sessions: InMemorySessionService
) -> Callable[..., StudioStore]:
    """A fresh `StudioStore` over the test's Postgres schema and turn
    sessions: each call stands for another instance. With `clock`, the
    store reads its leases' time from that clock instead of the database's."""

    def build(*, clock: StoreClock | None = None) -> StudioStore:
        return StudioStore(
            pg_engine,
            adk=NO_ADK,
            turn_specs=session_turn_specs(turn_sessions, app_name="app"),
            **(clock.store_kwargs if clock is not None else {}),
        )

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
    new_store, turn_sessions, pg_engine
) -> None:
    """A checkpoint after a search writes no spec block to the conversation's
    rows; closing the turn on another instance stores the block of the source
    its plan used, read from the turn's own session."""
    used = _PLAN.sources[0].spec_id
    candidate = "spec_" + "d" * 32
    blocks = {used: {"spec_id": used, "note": "used block"}, candidate: {"note": "unused block"}}
    store_a = new_store()
    state = await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store_a.save("principal", CONVERSATION)
    await turn_sessions.create_session(
        app_name="app",
        user_id="principal",
        session_id=state.session_id,
        state={ORCHESTRATOR_SPECS_KEY: blocks},
    )
    state.turn_specs = dict(blocks)
    state.turn_plans = [_PLAN_JSON]

    await store_a.checkpoint("principal", CONVERSATION, segment_id)

    written = await _stored_rows_text(pg_engine)
    assert "used block" not in written and "unused block" not in written

    resumed = await new_store().load("principal", CONVERSATION)
    assert resumed is not None
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
        ),
        skippable=True,
        step_order=("step_b", "step_a"),
        step_labels={"step_b": "Fetch aid to Kenya", "step_a": "Total aid by year"},
        resuming_segment="seg_1",
        answer={"decision": "approve", "option_id": "spec_a"},
    )
    state.segment = Segment(id="seg_1", owner="owner_a", expires_at=1_700_000_045.0, cancel=True)
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
    )
    assert (pending.skippable, pending.legacy) == (True, False)
    assert pending.step_order == ("step_b", "step_a")
    assert pending.step_labels == {"step_b": "Fetch aid to Kenya", "step_a": "Total aid by year"}
    assert pending.resuming_segment == "seg_1"
    assert pending.answer == {"decision": "approve", "option_id": "spec_a"}
    assert loaded.awaiting_answer is False  # its only entry is resuming
    assert loaded.segment == Segment(
        id="seg_1", owner="owner_a", expires_at=1_700_000_045.0, cancel=True
    )
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
    assert loaded._telemetry_span is None


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
    pg_engine, turn_sessions
) -> None:
    """B's own clock runs a day ahead of A's. A's lease is live by the
    database's clock, so B can neither claim it nor miss it when cancelling."""
    store_a = StudioStore(
        pg_engine,
        adk=NO_ADK,
        turn_specs=session_turn_specs(turn_sessions, app_name="app"),
    )
    store_b = StudioStore(
        pg_engine,
        adk=NO_ADK,
        turn_specs=session_turn_specs(turn_sessions, app_name="app"),
        clock=lambda: time.time() + 86_400,
    )
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
    earlier segment used, and a segment past its reservation's headroom
    reserves the next block before it sends more. A new turn numbers from 0
    again."""
    store_a = new_store(clock=clock)
    state_a = await store_a.create("principal", CONVERSATION)
    state_a.begin_turn()
    first = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    assert state_a.seq == 0
    state_a.seq = 37
    await store_a.release("principal", CONVERSATION, first)

    store_b = new_store(clock=clock)
    state_b = await store_b.load("principal", CONVERSATION)
    assert state_b.seq == 37
    second = store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)
    assert state_b.seq == SEQ_BLOCK

    state_b.seq = 2 * SEQ_BLOCK - 10  # inside the headroom of [256, 512)
    await store_b.reserve_frames("principal", CONVERSATION, second)
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)  # B dies mid-segment

    store_c = new_store(clock=clock)
    state_c = await store_c.load("principal", CONVERSATION)
    store_c.claim_segment("principal", CONVERSATION)
    await store_c.save("principal", CONVERSATION)
    assert state_c.seq == 3 * SEQ_BLOCK - 10
    state_b.seq = 3 * SEQ_BLOCK - 20  # B again inside its headroom, now fenced out
    with pytest.raises(SegmentLost):
        await store_b.reserve_frames("principal", CONVERSATION, second)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    state_c.segment = None
    await store_c.save("principal", CONVERSATION)
    state_c.begin_turn()
    store_c.claim_segment("principal", CONVERSATION)
    await store_c.save("principal", CONVERSATION)
    assert state_c.seq == 0


async def test_a_resuming_claim_starts_past_numbers_a_lapsed_stream_reserved_after_the_load(
    new_store, clock
) -> None:
    """B loads the conversation once A's lease has lapsed and claims the turn
    in memory. A's heartbeat then brings its lease back, A reserves another
    block and lapses again before B's claim commits. B numbers from the end
    A reserved, past every number A may have sent."""
    store_a = new_store(clock=clock)
    state_a = await store_a.create("principal", CONVERSATION)
    state_a.begin_turn()
    first = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)

    store_b = new_store(clock=clock)
    state_b = await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    assert await store_a.renew("principal", CONVERSATION, first) == "ok"
    state_a.seq = SEQ_BLOCK - 10  # inside the headroom of [0, 256)
    await store_a.reserve_frames("principal", CONVERSATION, first)
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    await store_b.save("principal", CONVERSATION)

    assert state_b.seq == 2 * SEQ_BLOCK - 10


async def test_a_stream_whose_lease_lapsed_reserves_no_more_frame_numbers(new_store, clock) -> None:
    """Another instance may already have loaded the reserved end to claim the
    turn from, so a lapsed stream's reservation is refused and the stored end
    stays where it was."""
    store = new_store(clock=clock)
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    state.seq = SEQ_BLOCK - 10
    await clock.advance(studio_state.SEGMENT_TTL_S + 1)

    with pytest.raises(SegmentLost):
        await store.reserve_frames("principal", CONVERSATION, segment_id)

    resumed = new_store(clock=clock)
    state_resumed = await resumed.load("principal", CONVERSATION)
    resumed.claim_segment("principal", CONVERSATION)
    assert state_resumed.seq == SEQ_BLOCK


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

    assert await store_a.renew("principal", CONVERSATION, segment_id) == "lost"


async def test_renew_reports_ok_and_pushes_the_ttl_forward(new_store, clock) -> None:
    store = new_store(clock=clock)
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    await clock.advance(30)
    assert await store.renew("principal", CONVERSATION, segment_id) == "ok"

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.segment is not None
    assert reloaded.segment.cancel is False
    assert reloaded.segment.expires_at == clock.at + studio_state.SEGMENT_TTL_S


async def test_renew_reports_cancelled_and_still_pushes_the_ttl_forward(new_store, clock) -> None:
    """`request_cancel`, from another instance, sets the segment's `cancel`
    flag; the owner's next `renew` reports it -- and still renews, so the
    flag keeps being observed until the owner reacts to it."""
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    assert await store_b.request_cancel("principal", CONVERSATION) is True

    await clock.advance(30)
    assert await store_a.renew("principal", CONVERSATION, segment_id) == "cancelled"

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.segment is not None
    assert reloaded.segment.cancel is True
    assert reloaded.segment.expires_at == clock.at + studio_state.SEGMENT_TTL_S


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


# --- checkpoint: fencing on the segment ------------------------------------


async def test_checkpoint_raises_segment_lost_after_a_takeover_and_writes_nothing(
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
    with pytest.raises(SegmentLost):
        await store_a.checkpoint("principal", CONVERSATION, segment_id)

    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.turn_final_text == ""


async def test_checkpoint_raises_segment_lost_when_the_ledger_is_gone(new_store, pg_engine) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    async with pg_engine.begin() as conn:
        await conn.execute(text("DELETE FROM conversations"))

    store.current("principal", CONVERSATION).turn_final_text = "an answer with no ledger"
    with pytest.raises(SegmentLost):
        await store.checkpoint("principal", CONVERSATION, segment_id)


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

    # The create's write token reaches the live state before the card lands
    # (`_RecordingExecutor._keep_write_token`).
    state_a.write_tokens["cmp_two"] = "token-two"

    landed = await store_a.change_canvas(
        "principal", CONVERSATION, lambda work: land_component(work, _CARD)
    )

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
    only once the canvas write commits, so a later checkpoint never stores a
    card the canvas does not hold."""
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


async def test_a_release_whose_commit_lost_its_acknowledgement_is_recorded_as_saved(
    new_store, monkeypatch, caplog
) -> None:
    """The retry finds the lease the lost commit cleared. The store records
    the write as made, with the row's version, and warns of no takeover."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    state.turn_final_text = "the segment's last words"
    faults = _dropping_connection(store, acknowledged=True)

    with caplog.at_level(logging.INFO, logger=studio_store.__name__):
        await store.release("principal", CONVERSATION, segment_id)

    assert faults == [1]
    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == [logging.WARNING]
    async with store.engine.connect() as conn:
        version = (await conn.execute(text("SELECT version FROM conversations"))).scalar_one()
    assert store._runtime[("principal", CONVERSATION)].version == version
    reloaded = await new_store().load("principal", CONVERSATION)
    assert (reloaded.segment, reloaded.turn_final_text) == (None, "the segment's last words")


async def test_a_frame_reservation_that_meets_a_dropped_connection_is_retried(
    new_store, monkeypatch
) -> None:
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store = new_store()
    state = await store.create("principal", CONVERSATION)
    state.begin_turn()
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    state.seq = SEQ_BLOCK - 10
    faults = _dropping_connection(store, acknowledged=False)

    await store.reserve_frames("principal", CONVERSATION, segment_id)

    assert faults == [1]
    async with store.engine.connect() as conn:
        reserved = (await conn.execute(text("SELECT seq_reserved FROM conversations"))).scalar_one()
    assert reserved == 2 * SEQ_BLOCK - 10


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


async def test_a_release_that_cannot_reach_the_database_logs_it_saved_nothing(
    new_store, clock, caplog
) -> None:
    """The release names the database, not a takeover, as why nothing was
    saved."""
    store = new_store(clock=clock)
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    await clock.set(store.lease_deadline("principal", CONVERSATION))

    async def dropped(*args: Any, **kwargs: Any) -> None:
        raise _connection_reset("UPDATE")

    store._write = dropped

    with caplog.at_level(logging.WARNING, logger=studio_store.__name__):
        await store.release("principal", CONVERSATION, segment_id)

    (record,) = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.levelno == logging.ERROR
    assert "could not reach the database" in record.getMessage()


async def test_a_release_retried_after_another_instance_claimed_and_released_is_fenced_out(
    new_store, monkeypatch, caplog
) -> None:
    """A's release commits and its acknowledgement is lost. Before the retry,
    B claims and releases. The row holds no lease, but at B's version, so the
    retry knows its own commit is not the last one and records nothing."""
    monkeypatch.setattr(studio_store, "_RETRY_FIRST_DELAY_S", 0)
    store_a = new_store()
    state_a = await store_a.create("principal", CONVERSATION)
    state_a.begin_turn()
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)
    state_a.turn_final_text = "A's last words"
    store_b = new_store()
    real_transaction = store_a._transaction
    faults: list[int] = []

    @asynccontextmanager
    async def losing_the_ack() -> Any:
        async with real_transaction() as conn:
            yield conn
        if not faults:
            faults.append(1)
            state_b = await store_b.load("principal", CONVERSATION)
            state_b.turn_final_text = "B's words"
            b_segment = store_b.claim_segment("principal", CONVERSATION)
            await store_b.save("principal", CONVERSATION)
            await store_b.release("principal", CONVERSATION, b_segment)
            raise _connection_reset("COMMIT")

    store_a._transaction = losing_the_ack
    version_before = store_a._runtime[("principal", CONVERSATION)].version

    with caplog.at_level(logging.WARNING, logger=studio_store.__name__):
        await store_a.release("principal", CONVERSATION, segment_id)

    assert faults == [1]
    assert any("taken over" in r.getMessage() for r in caplog.records)
    assert store_a._runtime[("principal", CONVERSATION)].version == version_before
    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.turn_final_text == "B's words"


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
    assert result == (["cmp_two"], [])
    assert state.take_unreported_removal() is True
    assert state.composition is not None and state.composition.blocks == []
    reloaded = await new_store().load("principal", CONVERSATION)
    assert reloaded.composition == state.composition


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


# --- release: drop resuming entries, clear the segment, checkpoint ----------


async def test_release_drops_resuming_entries_clears_the_segment_and_persists(
    new_store,
) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    state = store.current("principal", CONVERSATION)
    state.pending["int_resumed"] = _pending(resuming_segment=segment_id)
    state.pending["int_other"] = _pending(resuming_segment="seg_other")
    state.turn_final_text = "done"

    await store.release("principal", CONVERSATION, segment_id)

    reloaded = await new_store().load("principal", CONVERSATION)
    assert set(reloaded.pending) == {"int_other"}
    assert reloaded.segment is None
    assert reloaded.turn_final_text == "done"


@pytest.mark.parametrize("method", ["release", "renew"])
async def test_a_cancelled_segment_write_still_lands_then_reraises_the_cancel(
    new_store,
    method: str,
    clock,
) -> None:
    """A disconnect cancels the task running `release`, and the stream's end
    cancels the heartbeat running `renew`, while the write is opening its
    transaction. The write runs to the end and the caller's cancellation is
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
    if method == "release":
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

    caller = asyncio.ensure_future(studio_store._run_to_completion(failing()))
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    proceed.set()

    with pytest.raises(asyncio.CancelledError):
        await caller


async def test_release_never_raises_on_a_db_error(new_store) -> None:
    store = new_store()
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    @asynccontextmanager
    async def broken_transaction() -> Any:
        raise RuntimeError("db is down")
        yield

    store._transaction = broken_transaction

    await store.release("principal", CONVERSATION, segment_id)  # must not raise


async def test_release_swallows_a_takeover_and_still_never_raises(new_store, clock) -> None:
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)  # A's segment has expired
    await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    await store_a.release("principal", CONVERSATION, segment_id)  # must not raise


@pytest.mark.parametrize(("paused", "ended"), [(False, ["segment_lost"]), (True, [])])
async def test_a_release_that_lost_its_segment_ends_a_running_turn_s_span_alone(
    new_store, clock, monkeypatch, paused: bool, ended: list[str]
) -> None:
    """A's release is fenced out by B's takeover. A turn that was running
    ends A's span as lost; a turn whose pause A had stored keeps it for the
    answer that resumes it."""
    calls: list[str | None] = []

    def recording_end(handle: Any, **kwargs: Any) -> None:
        if handle is not None:
            calls.append(kwargs.get("error"))

    monkeypatch.setattr(studio_state, "end_turn_span", recording_end)
    store_a = new_store(clock=clock)
    store_b = new_store(clock=clock)
    state = await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    if paused:
        state.pending["int_1"] = _PendingInterrupt(adk_id="adk_1", kind="plan")
    await store_a.save("principal", CONVERSATION)
    state._telemetry_span = object()

    await clock.advance(studio_state.SEGMENT_TTL_S + 1)
    await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)
    await store_a.release("principal", CONVERSATION, segment_id)

    assert calls == ended


async def test_a_release_past_its_timeout_cancels_the_checkpoint_once(new_store) -> None:
    """The timeout cancels a stuck checkpoint once and lets the driver's own
    cleanup finish. A timeout that re-cancels at every await (anyio's
    `move_on_after`) cancels that cleanup too."""
    store = new_store()
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    cleanup_cancels = 0
    cleaned_up = False

    @asynccontextmanager
    async def stuck_transaction() -> Any:
        nonlocal cleanup_cancels, cleaned_up
        try:
            await asyncio.Event().wait()  # a write waiting on a row lock
        finally:
            try:
                await asyncio.sleep(0.05)  # the rollback and connection return
            except asyncio.CancelledError:
                cleanup_cancels += 1
                raise
            cleaned_up = True
        yield

    store._transaction = stuck_transaction

    await store.release("principal", CONVERSATION, segment_id, timeout=0.05)

    assert cleanup_cancels == 0
    assert cleaned_up


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
