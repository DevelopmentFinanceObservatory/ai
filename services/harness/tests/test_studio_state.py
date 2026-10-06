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

The canvas digest entry is pinned here too: what a card shows (its variable,
entities and span) travels in the digest the models read, from the names the
component's own fact summary and stored intent carry.

The ledger-backed store (`StudioStore.create`/`load`/`save`/`checkpoint`/
`release`) and the segment lease (`claim_segment`/`renew`/`request_cancel`)
are exercised at the bottom of this file, parametrised over
`InMemorySessionService` and `DatabaseSessionService(sqlite+aiosqlite)` for
tests whose behaviour is the same on either backend
(`InMemorySessionService.append_event` has no compare-and-swap check at
all). The stale-retry and concurrent-write tests run only over the database
backend, where two `StudioStore`s sharing one `DatabaseSessionService` can
actually race.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from compiler.sources import source_from_spec_block
from fastapi.responses import JSONResponse
from google.adk.errors import StaleSessionError
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from harness import studio_state
from harness.approval import ApprovalDecision, ApprovalRecord
from harness.graph import ORCHESTRATOR_SPECS_KEY
from harness.studio_state import (
    CANVAS_DIGEST_MAX_BYTES,
    CONVERSATION_DIGEST_MAX_BYTES,
    ConversationState,
    Segment,
    SegmentBusy,
    SegmentLost,
    StudioStore,
    _digest_entry,
    _next_manifest,
    _PendingInterrupt,
    land_component,
)
from harness.turn_records import PlanRecord, RecordedIdentity, SearchReading
from harness.wire import ComponentType, CompositionActor, CompositionBlock, CompositionManifest
from planner.workflow import SelectedSource

from .resolved_revision import resolved_revision
from .sqlite_session import sqlite_session_service
from .test_turn_records import _top_donors

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


# --- land_component: the one place the canvas decision is made and done ----


def _component(component_id: str, *, revision: int = 1, kind: str = "line_chart") -> dict[str, Any]:
    """A minimal `write_component` result: just what `land_component` and
    `_digest_entry` read (`component_id`, `type`, `revision`) -- the rest
    of a real platform payload is exercised in `test_studio_translate.py`'s
    `_written_component`."""
    return {"component_id": component_id, "revision": revision, "type": kind}


def test_land_component_replaces_the_named_block_in_place() -> None:
    """`revises` naming the canvas's only block: the new component takes
    that block's slot, the manifest still holds one block, the
    turn_components entry says `replaced`, and the return value does too."""
    state = ConversationState(conversation_id=CONVERSATION)
    block = studio_state.CompositionBlock(
        component_id="cmp_nigeria", revision=1, type="line_chart", title="Nigeria", pinned=False
    )
    state.composition = studio_state._next_manifest(
        None, [block], actor=studio_state.CompositionActor.MODEL
    )

    action = studio_state.land_component(state, _component("cmp_two"), revises="cmp_nigeria")

    assert action == "replaced"
    assert [b.component_id for b in state.composition.blocks] == ["cmp_two"]
    assert state.turn_components[-1]["action"] == "replaced"
    assert "cmp_two" in state.components


def test_land_component_appends_when_revises_names_nothing_on_canvas() -> None:
    """A `revises` target already gone (or never composed) cannot be
    swapped, so the write appends instead."""
    state = ConversationState(conversation_id=CONVERSATION)

    action = studio_state.land_component(state, _component("cmp_two"), revises="cmp_gone")

    assert action == "created"
    assert [b.component_id for b in state.composition.blocks] == ["cmp_two"]
    assert state.turn_components[-1]["action"] == "created"


def test_land_component_appends_when_revises_is_none() -> None:
    state = ConversationState(conversation_id=CONVERSATION)

    action = studio_state.land_component(state, _component("cmp_two"), revises=None)

    assert action == "created"
    assert [b.component_id for b in state.composition.blocks] == ["cmp_two"]


def test_land_component_first_card_op_replays_onto_none_identically() -> None:
    """The op `land_component` records for the very first card on a fresh
    canvas must be a pure function of its input: replayed against `None`
    (a fresh checkpoint fetch that also found no composition yet), it
    reproduces the exact same manifest, `composition_id` included, rather
    than minting a second, different id."""
    state = ConversationState(conversation_id=CONVERSATION)

    action = studio_state.land_component(state, _component("cmp_one"), revises=None)

    assert action == "created"
    live = state.composition
    assert len(state._composition_ops) == 1

    replayed = state._composition_ops[0](None)

    assert replayed == live
    assert replayed.composition_id == live.composition_id


def test_land_component_held_when_auto_add_is_off() -> None:
    """`auto_add=False` means there is no block for anything to swap or
    append: the digest entry still gets written (the card is still
    findable by `recall`), but the composition itself is untouched and the
    return is `"held"`."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.auto_add = False

    action = studio_state.land_component(state, _component("cmp_two"), revises="cmp_nigeria")

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
            "indicators": [
                {"id": "dcid:LifeExpectancy_Person", "name": "Life Expectancy", "description": ""}
            ],
            "attribution": {
                "label": "WDI · 1 indicator(s)",
                "source_count": 1,
                "indicator_count": 1,
            },
            "window": {"start": "1960", "end": "2023"},
        },
    }


def test_digest_entry_carries_the_variable_entities_and_span_the_card_shows() -> None:
    entry = _digest_entry(_two_entity_component())

    assert entry["type"] == "line_chart"
    assert entry["title"] == "Life expectancy in Guatemala and Nigeria"
    assert entry["variable"] == ["Life Expectancy"]
    assert entry["entities"] == ["Guatemala", "Nigeria"]
    assert entry["time"] == "1960-2023"


def test_digest_entry_takes_nothing_but_the_title_from_the_resolved_revision() -> None:
    """The digest's names come from the wire fact summary alone. A summary
    without labels leaves `variable`, `entities` and `time` out, although
    the resolved revision in `payload` names its own series and window."""
    component = _two_entity_component()
    component["fact_summary"] = {
        "component_id": "cmp_two",
        "revision": 1,
        "series": [],
        "indicators": [],
        "attribution": {"label": "WDI", "source_count": 1, "indicator_count": 1},
    }
    assert component["payload"]["component"]["series"][0]["label"]

    entry = _digest_entry(component)

    assert entry["title"] == "Life expectancy in Guatemala and Nigeria"
    assert "variable" not in entry
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
    assert "variable_truncated" not in block


def test_digest_entry_omits_what_a_component_does_not_show() -> None:
    """A note card names no variable, no entity and no span: it keeps none of
    the three keys rather than carrying empty lists. It carries no title
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
    assert "variable" not in entry
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
        "indicators": [
            {
                "id": f"dcid:Count_Person_{index}",
                "name": "Grants, Excluding Technical Cooperation (current US$)",
                "description": "Grants are legally binding commitments.",
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


def _shows(name: str, time: str, *, count: int = studio_state._DIGEST_SHOW_CAP) -> dict[str, Any]:
    """The `variable` / `entities` / `time` keys `_digest_shows` adds to an entry."""
    return {"variable": [name] * count, "entities": [name] * count, "time": time}


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
            "indicators": summary["indicators"],
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


def test_an_oversized_fact_summary_becomes_a_stub_that_says_it_was_cut():
    """A summary that does not fit even with no series, or that is not a
    summary at all, becomes a stub."""
    no_series_fit = {**_fact_summary(), "indicators": _fact_summary(BIG // 100)["indicators"]}
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


def test_a_component_id_a_type_and_a_title_of_any_size_are_cut_to_their_caps():
    """The digest's maximum holds for every field of a block.

    The id and the type come from the platform and the title from an edit the
    model wrote (`studio_edit._revised_frames` stores `patch["title"]` as given),
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

    served = state.recall_state(scope="canvas", query="ttt")

    assert served["cards"] == [{"component_id": "cmp_01", **_entry(title, summary)}]
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


# The definitions dfo-apis served for Guatemala's nominal GDP over every year
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


def test_the_digest_skips_a_stored_plan_record_that_no_longer_validates(
    caplog: pytest.LogCaptureFixture,
):
    state = ConversationState(conversation_id=CONVERSATION)
    malformed = {**_PLAN_JSON, "unknown_field": 1}
    state.turns = [
        {"user": "q1", "assistant": "a1", "plans": [malformed, _PLAN_JSON]},
        {"user": "q2", "assistant": "a2"},
    ]

    with caplog.at_level(logging.WARNING, logger="harness.studio_state"):
        digest = state.conversation_digest_state()

    assert digest is not None
    assert digest["turns"][0]["plans"] == [_PLAN.digest_line()]
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
            "plans": [{**_PLAN_JSON, **_MALFORMED}, _PLAN_JSON],
            "readings": [{**_READING_JSON, **_MALFORMED}, _READING_JSON],
        }
    ]
    state.turn_plans = [{**_PLAN_JSON, **_MALFORMED}, _PLAN_JSON]
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


def test_a_ledger_turn_stored_before_search_stopped_listing_members_still_loads():
    """A conversation's ledger holds turns stored when Search listed each source's
    members: every plan source and recorded identity carries `stat_vars`. The
    turn still reads as its plan and its identities, the list left behind."""
    plan = copy.deepcopy(_PLAN_JSON)
    for source in plan["sources"]:
        source["stat_vars"] = [] if source["members"] else [_GDP_STAT_VAR]
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


def test_recall_leaves_out_a_turns_plans_readings_blocks_and_identities():
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        {
            "user": "q",
            "assistant": "a",
            "plans": [_PLAN_JSON],
            "readings": [],
            "specs": {},
            "identities": {
                _GDP_ALL_YEARS["spec_id"]: {
                    "identity": _GDP_ALL_YEARS["identity"],
                    "stat_vars": [_GDP_STAT_VAR],
                }
            },
        }
    ]

    assert state.recall_state(scope="conversation", query="")["turns"] == [
        {"turn": 1, "user": "q", "assistant": "a"}
    ]


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
        **_entry("Aid", {"series": ["x"]}),
    }


@pytest.mark.parametrize(
    ("target", "legal"),
    [
        ({"part": "plan", "turn": 2}, {"turns": [1, 3]}),
        ({"part": "plan", "turn": 9}, {"turns": [1, 3]}),
        ({"part": "readings", "turn": 3}, {"turns": [1]}),
        ({"part": "canvas", "component_id": "cmp_99"}, {"component_ids": ["cmp_01"]}),
        ({"part": "canvas"}, {"component_ids": ["cmp_01"]}),
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


# --- anaphor_window: what "the same years" and a card-scoped follow-up's ---
# --- own window resolve to -------------------------------------------------


def _card(component_id: str, *, window: str | None = None) -> dict[str, Any]:
    """One `turns[n]["cards"]` entry, `_turn_component_entry`'s own shape."""
    card: dict[str, Any] = {
        "component_id": component_id,
        "type": "line_chart",
        "title": component_id,
        "action": "created",
    }
    if window:
        card["window"] = window
    return card


def test_anaphor_window_reads_the_newest_card_past_the_history_card_cap() -> None:
    """`_history_turn` keeps only a turn's FIRST `_HISTORY_CARD_CAP` cards
    for the conversation digest, dropping anything past it; `anaphor_window`
    reads the uncapped `turns` list instead, so the newest card's window --
    landed after the cap -- is still found."""
    capped = [_card(f"cmp_{index}") for index in range(studio_state._HISTORY_CARD_CAP)]
    newest = _card("cmp_newest", window="2021-2023")
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [_turn(0, cards=[*capped, newest])]

    assert state.anaphor_window(None) == "2021 to 2023"


def test_anaphor_window_finds_a_window_in_a_turn_dropped_from_the_digest() -> None:
    """`conversation_digest_state` keeps only the newest `_HISTORY_TURN_CAP`
    turns; `anaphor_window` reads the full `turns` list, so a window landed
    further back than that cap is still found when no later turn landed
    one of its own."""
    oldest = _turn(0, cards=[_card("cmp_old", window="2015-2020")])
    later = [
        _turn(index, cards=[_card(f"cmp_{index}")])
        for index in range(1, 1 + studio_state._HISTORY_TURN_CAP)
    ]
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [oldest, *later]

    assert state.anaphor_window(None) == "2015 to 2020"


def test_anaphor_window_prefers_the_focus_cards_own_window_over_the_newest() -> None:
    """A card-scoped follow-up names its focus card in `canvas_digest`'s
    `focus_component_id`; that card's own window wins even when a newer
    card elsewhere on the canvas carries a different one."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.components["cmp_a"] = _entry("A", None, shows=_shows("gdp", "2015-2020"))
    state.turns = [
        _turn(0, cards=[_card("cmp_a", window="2015-2020")]),
        _turn(1, cards=[_card("cmp_b", window="2021-2023")]),
    ]

    assert state.anaphor_window("cmp_a") == "2015 to 2020"


def test_anaphor_window_falls_back_to_the_newest_when_the_focus_card_has_no_window() -> None:
    """A focus card that names no window of its own (a note, a table with
    no resolved span) falls back to the most recently landed card's,
    the same as a question with no focus at all."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.components["cmp_a"] = _entry("A", None)
    state.turns = [
        _turn(0, cards=[_card("cmp_a")]),
        _turn(1, cards=[_card("cmp_b", window="2021-2023")]),
    ]

    assert state.anaphor_window("cmp_a") == "2021 to 2023"


def test_anaphor_window_prefers_the_newer_of_two_turns_each_carrying_a_window() -> None:
    """Two turns each land a card with its own window: the newer turn's
    window wins, proving the fallback walks `reversed(self.turns)` -- newest
    turn first -- and not forward turn order, where the older turn's window
    would be found first instead."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        _turn(0, cards=[_card("cmp_a", window="2015-2020")]),
        _turn(1, cards=[_card("cmp_b", window="2021-2023")]),
    ]

    assert state.anaphor_window(None) == "2021 to 2023"


def test_anaphor_window_prefers_the_last_of_several_windowed_cards_in_one_turn() -> None:
    """One turn lands several cards, each with its own window: the last
    landed card's window wins, proving the fallback walks a turn's own
    cards newest-first, not the order `land_component` appended them in."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        _turn(
            0,
            cards=[
                _card("cmp_a", window="2015-2020"),
                _card("cmp_b", window="2021-2023"),
            ],
        )
    ]

    assert state.anaphor_window(None) == "2021 to 2023"


def test_anaphor_window_reads_a_revised_cards_new_window_not_its_original() -> None:
    """The same component id landed again in a later turn -- a revision, not
    a new card -- carries its own new `turn_components` entry
    (`land_component` appends on every call, `create` or `replace` alike):
    the later turn's entry is what the fallback finds, not the original
    turn's now-stale window."""
    state = ConversationState(conversation_id=CONVERSATION)
    state.turns = [
        _turn(0, cards=[_card("cmp_a", window="2015-2020")]),
        _turn(1, cards=[_card("cmp_a", window="2021-2023")]),
    ]

    assert state.anaphor_window(None) == "2021 to 2023"


# --- awaiting_answer / drop_resuming: a pending interrupt marked resuming ---
# --- reads as no longer awaiting a client answer -----------------------------


def _pending(*, resuming_segment: str | None = None) -> _PendingInterrupt:
    return _PendingInterrupt(adk_id="adk_1", kind="plan", resuming_segment=resuming_segment)


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


def _database_service(tmp_path, name: str = "studio") -> DatabaseSessionService:
    return sqlite_session_service(tmp_path / f"{name}.db")


@pytest.fixture(params=["memory", "database"])
def session_service(request, tmp_path) -> Any:
    """Both backends the ledger runs over. A test whose behaviour does not
    depend on staleness enforcement (round trip, delta-only save, a missing
    ledger, a non-JSON value) runs against both; a test of the stale-retry
    path builds a `DatabaseSessionService` of its own instead, since
    `InMemorySessionService.append_event` has no compare-and-swap check."""
    if request.param == "memory":
        return InMemorySessionService()
    return _database_service(tmp_path)


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


async def test_the_ledger_stores_a_turns_blocks_only_when_the_turn_closes(session_service) -> None:
    """A checkpoint after a search writes no spec block to the ledger; closing
    the turn on another instance stores the block of the source its plan
    used, read from the turn's own session."""
    used = _PLAN.sources[0].spec_id
    candidate = "spec_" + "d" * 32
    blocks = {used: {"spec_id": used, "note": "used block"}, candidate: {"note": "unused block"}}
    store_a = StudioStore(session_service, app_name="app")
    state = await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    state.begin_turn()
    await store_a.save("principal", CONVERSATION)
    await session_service.create_session(
        app_name="app",
        user_id="principal",
        session_id=state.session_id,
        state={ORCHESTRATOR_SPECS_KEY: blocks},
    )
    state.turn_specs = dict(blocks)
    state.turn_plans = [_PLAN_JSON]

    await store_a.checkpoint("principal", CONVERSATION, segment_id)

    ledger = await session_service.get_session(
        app_name="app.ledger", user_id="principal", session_id=CONVERSATION
    )
    written = json.dumps([ledger.state, *(e.actions.state_delta for e in ledger.events)])
    assert "used block" not in written and "unused block" not in written

    resumed = await StudioStore(session_service, app_name="app").load("principal", CONVERSATION)
    assert resumed is not None
    resumed.close_turn(final_text="The top donor was the United States.")

    assert resumed.turns[-1]["specs"] == {used: blocks[used]}


async def test_round_trip_of_every_durable_field(session_service) -> None:
    """Every durable field survives a save on one store and a load on a
    SECOND store over the same ledger -- the actual cross-instance case,
    not just a read of the same in-process object."""
    store_a = StudioStore(session_service, app_name="app")
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
        option_ids=frozenset({"spec_b", "spec_a"}),
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
    state.composition = _next_manifest(None, [block], actor=CompositionActor.MODEL)

    await store_a.save("principal", CONVERSATION)

    store_b = StudioStore(session_service, app_name="app")
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
    assert pending.option_ids == frozenset({"spec_a", "spec_b"})
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
    assert loaded.cancel_event is None
    assert loaded._telemetry_span is None


async def test_load_returns_none_for_a_conversation_the_ledger_does_not_hold(
    session_service,
) -> None:
    store = StudioStore(session_service, app_name="app")

    assert await store.load("principal", "conv_unknown") is None


async def test_a_conversation_id_shaped_like_a_per_turn_session_id_does_not_collide(
    session_service,
) -> None:
    """A per-turn ADK session already exists, literally named "A#t1" (the
    session `StudioRunner._ensure_session` mints for conversation "A"'s
    first turn), under the same app name a `StudioStore` built over this
    session service uses for its own per-turn sessions. `create`ing a
    brand-new conversation literally named "A#t1" must not collide with it
    -- the ledger lives under its own, disjoint app name."""
    await session_service.create_session(app_name="app", user_id="principal", session_id="A#t1")
    store = StudioStore(session_service, app_name="app")

    state = await store.create("principal", "A#t1")

    assert state.turn_index == 0


async def test_load_does_not_read_the_ledgers_event_history(session_service) -> None:
    """`GetSessionConfig(num_recent_events=0)`: several saves accumulate
    several event rows, and a later `load` must not read any of them back."""
    store = StudioStore(session_service, app_name="app")
    await store.create("principal", CONVERSATION)
    state = store.current("principal", CONVERSATION)
    for turn_index in range(1, 4):
        state.turn_index = turn_index
        await store.save("principal", CONVERSATION)

    await store.load("principal", CONVERSATION)

    entry = store._runtime[("principal", CONVERSATION)]
    assert entry.session is not None
    assert entry.session.events == []


async def test_save_appends_only_the_keys_that_changed(session_service) -> None:
    calls: list[dict[str, Any]] = []
    real_append_event = session_service.append_event

    async def recording_append_event(session, event):
        calls.append(dict(event.actions.state_delta or {}))
        return await real_append_event(session, event)

    session_service.append_event = recording_append_event
    store = StudioStore(session_service, app_name="app")
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
    assert calls[1] == {studio_state._KEY_TURN_INDEX: 2}

    # a save with nothing changed appends no event at all
    await store.save("principal", CONVERSATION)
    assert len(calls) == 2


async def test_a_non_json_value_raises_rather_than_being_silently_coerced(session_service) -> None:
    store = StudioStore(session_service, app_name="app")
    state = await store.create("principal", CONVERSATION)
    state.turn_components = [{"bad": object()}]

    with pytest.raises(TypeError):
        await store.save("principal", CONVERSATION)


async def test_second_claim_over_one_database_service_raises_stale(tmp_path) -> None:
    """Two `StudioStore`s share ONE `DatabaseSessionService`. Both load the
    same brand-new ledger; the first to save its claimed `turn_index` wins,
    and the second -- still holding its pre-claim snapshot -- raises
    `StaleSessionError` rather than silently overwriting the first claim."""
    service = _database_service(tmp_path)
    store_a = StudioStore(service, app_name="app")
    store_b = StudioStore(service, app_name="app")
    await store_a.create("principal", CONVERSATION)
    await store_b.load("principal", CONVERSATION)

    store_a.current("principal", CONVERSATION).turn_index = 1
    await store_a.save("principal", CONVERSATION)

    store_b.current("principal", CONVERSATION).turn_index = 1
    with pytest.raises(StaleSessionError):
        await store_b.save("principal", CONVERSATION)


# --- the segment lease: claim_segment / renew / request_cancel --------------


async def test_claim_segment_raises_segment_busy_while_a_live_segment_exists(
    session_service,
) -> None:
    clock = [1_000.0]
    store_a = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    await store_a.create("principal", CONVERSATION)
    store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    await store_b.load("principal", CONVERSATION)
    clock[0] += 10  # well inside the 45s TTL

    with pytest.raises(SegmentBusy) as excinfo:
        store_b.claim_segment("principal", CONVERSATION)

    assert excinfo.value.retry_after_seconds == studio_state.SEGMENT_TTL_S - 10


async def test_claim_segment_succeeds_once_the_live_segment_has_expired(
    session_service,
) -> None:
    clock = [1_000.0]
    store_a = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    await store_a.create("principal", CONVERSATION)
    first_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    await store_b.load("principal", CONVERSATION)
    clock[0] += studio_state.SEGMENT_TTL_S + 1

    second_id = store_b.claim_segment("principal", CONVERSATION)

    assert second_id != first_id


async def test_renew_reports_lost_after_a_takeover(session_service) -> None:
    clock = [1_000.0]
    store_a = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    clock[0] += studio_state.SEGMENT_TTL_S + 1  # A's segment has expired
    await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    assert await store_a.renew("principal", CONVERSATION, segment_id) == "lost"


async def test_renew_reports_ok_and_pushes_the_ttl_forward(session_service) -> None:
    clock = [1_000.0]
    store = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    clock[0] += 30
    assert await store.renew("principal", CONVERSATION, segment_id) == "ok"

    reloaded = await StudioStore(session_service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.segment is not None
    assert reloaded.segment.cancel is False
    assert reloaded.segment.expires_at == clock[0] + studio_state.SEGMENT_TTL_S


async def test_renew_reports_cancelled_and_still_pushes_the_ttl_forward(session_service) -> None:
    """`request_cancel`, from another instance, sets the segment's `cancel`
    flag; the owner's next `renew` reports it -- and still renews, so the
    flag keeps being observed until the owner reacts to it."""
    clock = [1_000.0]
    store_a = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    assert await store_b.request_cancel("principal", CONVERSATION) is True

    clock[0] += 30
    assert await store_a.renew("principal", CONVERSATION, segment_id) == "cancelled"

    reloaded = await StudioStore(session_service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.segment is not None
    assert reloaded.segment.cancel is True
    assert reloaded.segment.expires_at == clock[0] + studio_state.SEGMENT_TTL_S


async def test_request_cancel_returns_false_with_no_live_segment(session_service) -> None:
    store = StudioStore(session_service, app_name="app")
    await store.create("principal", CONVERSATION)

    assert await store.request_cancel("principal", CONVERSATION) is False


async def test_request_cancel_returns_false_once_the_segment_has_expired(session_service) -> None:
    clock = [1_000.0]
    store = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    await store.create("principal", CONVERSATION)
    store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    clock[0] += studio_state.SEGMENT_TTL_S + 1

    assert await store.request_cancel("principal", CONVERSATION) is False


# --- checkpoint: fetch-then-merge reconcile, fencing, and release -----------


async def test_checkpoint_merges_a_key_committed_by_another_writer_since_this_instance_last_saved(
    tmp_path,
) -> None:
    """B's checkpoint always fetches the ledger fresh before merging, so a
    key A committed after B's last save survives alongside B's own write."""
    service = _database_service(tmp_path)
    store_a = StudioStore(service, app_name="app")
    store_b = StudioStore(service, app_name="app")
    await store_a.create("principal", CONVERSATION)
    await store_b.load("principal", CONVERSATION)
    segment_id = store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    await store_a.load("principal", CONVERSATION)
    store_a.current("principal", CONVERSATION).turn_index = 1
    await store_a.save("principal", CONVERSATION)

    store_b.current("principal", CONVERSATION).turn_final_text = "b's answer"
    await store_b.checkpoint("principal", CONVERSATION, segment_id)

    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.turn_index == 1
    assert reloaded.turn_final_text == "b's answer"


async def test_checkpoint_keeps_a_composition_op_recorded_while_its_own_fetch_is_in_flight(
    tmp_path,
) -> None:
    """B pins the seed block and commits, independently of A; at the same
    moment, A's own live stream lands a new component on A's `entry.state`
    -- through a hook on the session service standing in for "instance B"
    committing concurrently while A's checkpoint's own ledger fetch is in
    flight. Fetch-then-diff order catches A's late op instead of losing it
    to a reload's wholesale overwrite of `entry.state`."""
    service = _database_service(tmp_path)
    store_a = StudioStore(service, app_name="app")
    store_b = StudioStore(service, app_name="app")
    await store_a.create("principal", CONVERSATION)

    seed = CompositionBlock(
        component_id="cmp_seed", revision=1, type=ComponentType.LINE_CHART, title="Seed"
    )
    state_a = store_a.current("principal", CONVERSATION)
    state_a.composition = _next_manifest(None, [seed], actor=CompositionActor.MODEL)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    state_b = await store_b.load("principal", CONVERSATION)

    real_get_session = service.get_session
    landed = []

    async def get_session_then_interfere(*args: Any, **kwargs: Any) -> Any:
        if not landed:
            landed.append(True)
            pinned = state_b.composition.blocks[0].model_copy(update={"pinned": True})
            state_b.composition = _next_manifest(
                state_b.composition, [pinned], actor=CompositionActor.USER
            )
            await store_b.save("principal", CONVERSATION)
            action = land_component(
                state_a,
                {"component_id": "cmp_added", "type": "line_chart", "revision": 1},
                revises=None,
            )
            assert action == "created"
        return await real_get_session(*args, **kwargs)

    service.get_session = get_session_then_interfere

    await store_a.checkpoint("principal", CONVERSATION, segment_id)

    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert [b.component_id for b in reloaded.composition.blocks] == ["cmp_seed", "cmp_added"]
    assert reloaded.composition.blocks[0].pinned is True


async def test_checkpoint_consumes_composition_ops_by_prefix_not_the_whole_list(
    tmp_path,
) -> None:
    """An op landed on `entry.state` after the fold point (`n` already
    captured) but before this checkpoint's append actually lands must
    survive -- only `ops[:n]`, the prefix folded into THIS checkpoint, is
    consumed, so it is not silently dropped nor replayed twice."""
    service = _database_service(tmp_path)
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    state = store.current("principal", CONVERSATION)
    action = land_component(
        state, {"component_id": "cmp_one", "type": "line_chart", "revision": 1}, revises=None
    )
    assert action == "created"
    assert len(state._composition_ops) == 1

    real_append_event = service.append_event

    async def append_event_then_land(session: Any, event: Any) -> Any:
        # A second op lands after `checkpoint` already captured `n` for
        # this attempt's fold, but before the append itself commits.
        land_component(
            state, {"component_id": "cmp_two", "type": "line_chart", "revision": 1}, revises=None
        )
        return await real_append_event(session, event)

    service.append_event = append_event_then_land

    await store.checkpoint("principal", CONVERSATION, segment_id)

    assert len(state._composition_ops) == 1
    assert state._composition_ops[0].keywords["component_id"] == "cmp_two"
    assert [b.component_id for b in state.composition.blocks] == ["cmp_one", "cmp_two"]


async def test_checkpoint_warns_when_a_key_changed_on_both_sides(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every durable key but the composition has exactly one writer by
    design; this forces the one case that violates it, to prove the
    checkpoint still writes this request's value (never drops it) and
    warns loudly naming the key, rather than resolving the conflict
    silently."""
    service = _database_service(tmp_path)
    store_a = StudioStore(service, app_name="app")
    store_b = StudioStore(service, app_name="app")
    await store_a.create("principal", CONVERSATION)
    await store_b.load("principal", CONVERSATION)
    segment_id = store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    await store_a.load("principal", CONVERSATION)
    store_a.current("principal", CONVERSATION).turn_final_text = "a's answer"
    await store_a.save("principal", CONVERSATION)

    store_b.current("principal", CONVERSATION).turn_final_text = "b's answer"
    with caplog.at_level(logging.WARNING, logger="harness.studio_state"):
        await store_b.checkpoint("principal", CONVERSATION, segment_id)

    assert "studio:turn_final_text" in caplog.text
    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.turn_final_text == "b's answer"


async def test_checkpoint_retries_through_stale_appends_and_keeps_the_write(tmp_path) -> None:
    """B always fetches fresh, but a third writer keeps landing a commit
    right after B's own fetch and before B's append lands, so the append
    itself goes stale -- the bounded retry loops past two such races and
    keeps B's write on the third attempt."""
    service = _database_service(tmp_path)
    store_b = StudioStore(service, app_name="app")
    store_c = StudioStore(service, app_name="app")
    await store_b.create("principal", CONVERSATION)
    segment_id = store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    store_b.current("principal", CONVERSATION).turn_final_text = "b's answer"

    real_append_event = service.append_event
    attempts = 0

    async def append_event_then_interfere(session: Any, event: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts in (1, 2):
            # Land a real interfering commit without recursing back into
            # this same mock (store_c's own save also calls append_event).
            service.append_event = real_append_event
            try:
                await store_c.load("principal", CONVERSATION)
                store_c.current("principal", CONVERSATION).seq = attempts
                await store_c.save("principal", CONVERSATION)
            finally:
                service.append_event = append_event_then_interfere
        return await real_append_event(session, event)

    service.append_event = append_event_then_interfere

    await store_b.checkpoint("principal", CONVERSATION, segment_id)

    assert attempts == 3
    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.turn_final_text == "b's answer"
    assert reloaded.seq == 2


async def test_checkpoint_raises_segment_lost_after_a_takeover_and_writes_nothing(
    tmp_path,
) -> None:
    service = _database_service(tmp_path)
    clock = [1_000.0]
    store_a = StudioStore(service, app_name="app", clock=lambda: clock[0])
    store_b = StudioStore(service, app_name="app", clock=lambda: clock[0])
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    clock[0] += studio_state.SEGMENT_TTL_S + 1  # A's segment has expired
    await store_b.load("principal", CONVERSATION)
    new_segment_id = store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)
    assert new_segment_id != segment_id

    store_a.current("principal", CONVERSATION).turn_final_text = "a's answer, too late"
    with pytest.raises(SegmentLost):
        await store_a.checkpoint("principal", CONVERSATION, segment_id)

    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.turn_final_text == ""


# --- mutate_composition: /composition's fetch-rebuild-mutate-append cycle ---


def _one_block_manifest(
    manifest: CompositionManifest | None, *, component_id: str = "cmp_one"
) -> CompositionManifest:
    block = CompositionBlock(
        component_id=component_id, revision=1, type=ComponentType.LINE_CHART, title="One"
    )
    return _next_manifest(manifest, [block], actor=CompositionActor.USER)


async def test_mutate_composition_appends_the_returned_manifest_and_consumes_its_ops(
    tmp_path,
) -> None:
    service = _database_service(tmp_path)
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    state = store.current("principal", CONVERSATION)
    land_component(
        state, {"component_id": "cmp_one", "type": "line_chart", "revision": 1}, revises=None
    )
    assert len(state._composition_ops) == 1

    def mutate(manifest: CompositionManifest | None) -> CompositionManifest:
        assert manifest is not None
        pinned = manifest.blocks[0].model_copy(update={"pinned": True})
        return _next_manifest(manifest, [pinned], actor=CompositionActor.USER)

    result = await store.mutate_composition("principal", CONVERSATION, mutate)

    assert result is not None and result.blocks[0].pinned is True
    assert state.composition is result
    assert state._composition_ops == []

    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.composition.blocks[0].pinned is True


async def test_mutate_composition_rebuilds_the_fetched_manifest_with_unsaved_ops_replayed(
    tmp_path,
) -> None:
    """A's own live stream has recorded (but not yet saved) a
    `land_component` op when `/composition` runs on the same instance
    mid-turn: `mutate` must see that op already folded into the manifest
    it is handed, not the bare (here, empty) ledger value."""
    service = _database_service(tmp_path)
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    state = store.current("principal", CONVERSATION)
    land_component(
        state, {"component_id": "cmp_one", "type": "line_chart", "revision": 1}, revises=None
    )

    seen: list[CompositionManifest | None] = []

    def mutate(manifest: CompositionManifest | None) -> CompositionManifest | None:
        seen.append(manifest)
        return manifest  # a no-op: nothing appended

    await store.mutate_composition("principal", CONVERSATION, mutate)

    assert seen[0] is not None
    assert [b.component_id for b in seen[0].blocks] == ["cmp_one"]


async def test_mutate_composition_no_op_returns_the_same_manifest_without_appending(
    tmp_path,
) -> None:
    service = _database_service(tmp_path)
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    state = store.current("principal", CONVERSATION)
    land_component(
        state, {"component_id": "cmp_one", "type": "line_chart", "revision": 1}, revises=None
    )

    real_append_event = service.append_event
    calls = 0

    async def counting_append(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return await real_append_event(*args, **kwargs)

    service.append_event = counting_append

    result = await store.mutate_composition("principal", CONVERSATION, lambda manifest: manifest)

    assert calls == 0
    assert result is state.composition
    assert len(state._composition_ops) == 1  # not consumed: nothing was appended


async def test_mutate_composition_refusal_returns_it_without_appending() -> None:
    service = InMemorySessionService()
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    refusal = JSONResponse(status_code=404, content={"error": "not_found"})

    result = await store.mutate_composition("principal", CONVERSATION, lambda manifest: refusal)

    assert result is refusal


async def test_mutate_composition_retries_through_stale_appends_and_keeps_the_write(
    tmp_path,
) -> None:
    """B always fetches fresh, but a third writer keeps landing a commit
    right after B's own fetch and before B's append lands, so the append
    itself goes stale -- the bounded retry loops past two such races and
    keeps B's write on the third attempt."""
    service = _database_service(tmp_path)
    store_b = StudioStore(service, app_name="app")
    store_c = StudioStore(service, app_name="app")
    await store_b.create("principal", CONVERSATION)

    real_append_event = service.append_event
    attempts = 0

    async def append_event_then_interfere(session: Any, event: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts in (1, 2):
            service.append_event = real_append_event
            try:
                await store_c.load("principal", CONVERSATION)
                store_c.current("principal", CONVERSATION).seq = attempts
                await store_c.save("principal", CONVERSATION)
            finally:
                service.append_event = append_event_then_interfere
        return await real_append_event(session, event)

    service.append_event = append_event_then_interfere

    result = await store_b.mutate_composition("principal", CONVERSATION, _one_block_manifest)

    assert attempts == 3
    assert result is not None
    assert [b.component_id for b in result.blocks] == ["cmp_one"]
    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.seq == 2
    assert [b.component_id for b in reloaded.composition.blocks] == ["cmp_one"]


async def test_mutate_composition_raises_after_exhausted_retries_and_keeps_the_rebuild(
    tmp_path,
) -> None:
    """Every attempt loses the race to a fourth writer: the bounded retry
    gives up and raises rather than looping forever, and live state is left
    holding the last rebuild -- here still empty, since the interfering
    writer never touches the composition -- not the caller's edit."""
    service = _database_service(tmp_path)
    store_b = StudioStore(service, app_name="app")
    store_c = StudioStore(service, app_name="app")
    await store_b.create("principal", CONVERSATION)
    state_b = store_b.current("principal", CONVERSATION)

    real_append_event = service.append_event
    attempts = 0

    async def always_interfere(session: Any, event: Any) -> Any:
        nonlocal attempts
        attempts += 1
        service.append_event = real_append_event
        try:
            await store_c.load("principal", CONVERSATION)
            store_c.current("principal", CONVERSATION).seq = attempts
            await store_c.save("principal", CONVERSATION)
        finally:
            service.append_event = always_interfere
        return await real_append_event(session, event)

    service.append_event = always_interfere

    with pytest.raises(StaleSessionError):
        await store_b.mutate_composition("principal", CONVERSATION, _one_block_manifest)

    assert attempts == studio_state._COMPOSITION_MUTATE_RETRY_ATTEMPTS
    assert state_b.composition is None


# --- release: drop resuming entries, clear the segment, checkpoint ----------


async def test_release_drops_resuming_entries_clears_the_segment_and_persists(
    tmp_path,
) -> None:
    service = _database_service(tmp_path)
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    state = store.current("principal", CONVERSATION)
    state.pending["int_resumed"] = _pending(resuming_segment=segment_id)
    state.pending["int_other"] = _pending(resuming_segment="seg_other")
    state.turn_final_text = "done"

    await store.release("principal", CONVERSATION, segment_id)

    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert set(reloaded.pending) == {"int_other"}
    assert reloaded.segment is None
    assert reloaded.turn_final_text == "done"


@pytest.mark.parametrize("method", ["release", "renew"])
async def test_a_cancelled_segment_write_still_lands_then_reraises_the_cancel(
    session_service: Any, method: str
) -> None:
    """A disconnect cancels the task running `release`, and the stream's end
    cancels the heartbeat running `renew`, while the write is on its read.
    The write runs to the end and the caller's cancellation is raised once
    it has."""
    clock = [1_000.0]
    store = StudioStore(session_service, app_name="app", clock=lambda: clock[0])
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    clock[0] += 30
    reading = asyncio.Event()
    real_get_session = session_service.get_session

    async def slow_get_session(*args: Any, **kwargs: Any) -> Any:
        reading.set()
        await asyncio.sleep(0.05)
        return await real_get_session(*args, **kwargs)

    session_service.get_session = slow_get_session
    writing = asyncio.ensure_future(getattr(store, method)("principal", CONVERSATION, segment_id))
    await reading.wait()
    writing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writing

    session_service.get_session = real_get_session
    reloaded = await StudioStore(session_service, app_name="app").load("principal", CONVERSATION)
    if method == "release":
        assert reloaded.segment is None
    else:
        assert reloaded.segment is not None
        assert reloaded.segment.expires_at == clock[0] + studio_state.SEGMENT_TTL_S


async def test_release_never_raises_on_a_db_error(tmp_path) -> None:
    service = _database_service(tmp_path)
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    async def broken_get_session(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("db is down")

    service.get_session = broken_get_session

    await store.release("principal", CONVERSATION, segment_id)  # must not raise


async def test_release_swallows_a_takeover_and_still_never_raises(tmp_path) -> None:
    service = _database_service(tmp_path)
    clock = [1_000.0]
    store_a = StudioStore(service, app_name="app", clock=lambda: clock[0])
    store_b = StudioStore(service, app_name="app", clock=lambda: clock[0])
    await store_a.create("principal", CONVERSATION)
    segment_id = store_a.claim_segment("principal", CONVERSATION)
    await store_a.save("principal", CONVERSATION)

    clock[0] += studio_state.SEGMENT_TTL_S + 1  # A's segment has expired
    await store_b.load("principal", CONVERSATION)
    store_b.claim_segment("principal", CONVERSATION)
    await store_b.save("principal", CONVERSATION)

    await store_a.release("principal", CONVERSATION, segment_id)  # must not raise


async def test_a_reader_left_open_does_not_stall_a_release(tmp_path) -> None:
    """A cancelled turn can leave a half-read `SELECT` holding its sqlite
    connection open (`sqlite_session`). The test databases run in WAL mode,
    so the release commits past it instead of waiting out sqlite's busy
    timeout."""
    service = _database_service(tmp_path)
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)

    with closing(sqlite3.connect(tmp_path / "studio.db")) as reader:
        half_read = reader.execute("SELECT name FROM sqlite_master")
        half_read.fetchone()  # the statement stays open, mid-result
        started = time.monotonic()
        await store.release("principal", CONVERSATION, segment_id)
        elapsed = time.monotonic() - started

    assert elapsed < 1
    reloaded = await StudioStore(service, app_name="app").load("principal", CONVERSATION)
    assert reloaded.segment is None


async def test_a_release_past_its_timeout_cancels_the_checkpoint_once() -> None:
    """The timeout cancels a stuck checkpoint once and lets the session
    service's own cleanup finish. A timeout that re-cancels at every await
    (anyio's `move_on_after`) cancels that cleanup too, and a SQLAlchemy
    connection torn down twice over aiosqlite leaves a task that never
    finishes."""
    service = InMemorySessionService()
    store = StudioStore(service, app_name="app")
    await store.create("principal", CONVERSATION)
    segment_id = store.claim_segment("principal", CONVERSATION)
    await store.save("principal", CONVERSATION)
    cleanup_cancels = 0
    cleaned_up = False

    async def stuck_get_session(*args: Any, **kwargs: Any) -> Any:
        nonlocal cleanup_cancels, cleaned_up
        try:
            await asyncio.Event().wait()  # a read waiting on a database lock
        finally:
            try:
                await asyncio.sleep(0.05)  # the rollback and connection close
            except asyncio.CancelledError:
                cleanup_cancels += 1
                raise
            cleaned_up = True

    service.get_session = stuck_get_session

    await store.release("principal", CONVERSATION, segment_id, timeout=0.05)

    assert cleanup_cancels == 0
    assert cleaned_up


def test_a_pending_plan_saved_before_step_labels_were_kept_loads_with_none() -> None:
    entry = {"adk_id": "adk_1", "kind": "plan", "step_order": ["step_a"]}

    pending = studio_state._deserialize_pending({"int_1": entry})["int_1"]

    assert pending.step_order == ("step_a",)
    assert pending.step_labels == {}
