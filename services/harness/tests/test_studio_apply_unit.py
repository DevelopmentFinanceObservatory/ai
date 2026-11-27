"""A card's unit change through `POST /api/studio/turn` with `apply_unit`.

The first turn searches France's aid to Kenya, a spec Search converts into US
dollars at current prices or at a base year from 2000 to 2023, fetches it and
draws one card on the retrieval's frame. A later `apply_unit` re-reads the spec
in the chosen unit through Search, fetches it with the card's frame arguments
and redraws the card in its slot, with no model call.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from common.live_apis import LiveApisError
from common.mcp_telemetry import WRITE_TOKEN_KEY
from compiler.surface import ProgressCallback, ToolCallResult, ToolId
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness import card_unit
from harness.card_unit import UNIT_GONE_NOTE, ChangeMove, UnitCard
from harness.chart_binding import LINE_CHART_MAX_VALUES
from harness.conversation import current_conversation_id
from harness.graph.state import frame_id_for
from harness.place_options import ApisPlaceOptions
from harness.provenance import CARD_LINEAGE_KEY, CardLineage, ProvenanceInput, ProvenanceStore
from harness.studio_edit import _changed_title, _composed_title, _redrawn_intent
from harness.studio_state import PLAN_LINK_KEY, DatabaseUnavailable
from harness.studio_store import StudioStore
from harness.wire import (
    CardChanges,
    CompositionManifest,
    PlaceOption,
    PlacesChange,
    PlacesOffer,
    PlanLink,
    WindowChange,
    WindowOffer,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .resolved_revision import resolved_revision
from .scripted_genai import call_part, text_part
from .scripted_tools import CannedSearch, chart, fetch_call, make_charts_call, search_call
from .studio_routes import AUTH
from .test_studio import (
    _VIEW,
    COMPONENT,
    PRINCIPAL,
    ScriptedExecutor,
    _app,
    _client,
    _frame,
    _france_kenya_converted,
    _interrupt_body,
    _mutation_body,
    _stream,
    _turn_body,
)
from .turn_driver import CONVERSATION

_ENVELOPE = _france_kenya_converted()
_SPEC = _ENVELOPE["parts"][0]["spec"]["spec_id"]
_REF = {"spec_id": _SPEC, "content_version": "cv_" + "4" * 32}
_SUMMED = {"format": "parquet", "bundle": False, "layout": "frame"} | {
    "groupby": {"by": ["time_period"], "op": "sum"}
}
_CONSTANT_2023 = {"currency": "USD", "prices": "constant", "base_year": 2023}
_TITLE = "France's aid to Kenya, current US dollars"


def _component(
    component_id: str,
    ref: Mapping[str, str],
    title: str,
    window: tuple[int, int] = (2015, 2023),
) -> dict[str, Any]:
    """The platform's card `component_id` on frame `ref`, titled `title`,
    drawing the years `window`."""
    component = copy.deepcopy(COMPONENT)
    component["component_id"] = component_id
    component["payload"] = resolved_revision(
        "stat_card", component_id=component_id, revision=1, frame_ref=dict(ref), title=title
    )
    component["fact_summary"]["component_id"] = component_id
    component["fact_summary"]["window"] = {"start": str(window[0]), "end": str(window[1])}
    # A stat card draws the last year of its window, as the platform's
    # assembler draws one.
    component["payload"]["component"]["period"] = str(window[1])
    component["provenance"]["inputs"] = [dict(ref)]
    return component


# The coverage entries of a France to Kenya retrieval.
_RECIPIENTS = {
    "label": "financingRecipient",
    "count": 1,
    "axis": "destination",
    "noun": {"singular": "recipient country", "plural": "recipient countries"},
}
_PROVIDERS = {"label": "financingProvider", "count": 1, "axis": "origin"}


def _written(
    component_id: str,
    ref: Mapping[str, str],
    title: str,
    window: tuple[int, int] = (2015, 2023),
    identity: Mapping[str, Any] | None = None,
    entity_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """A `write_component` outcome for that card, with the lineage of the one
    `get_data` retrieval it draws, which summed the spec's StatVars and
    served `identity` when given. Its coverage names the recipients' noun
    and no provider's."""
    definition: dict[str, Any] = {"spec_id": ref["spec_id"]}
    if identity is not None:
        definition["identity"] = dict(identity)
    manifest = {
        **({"entity_labels": dict(entity_labels)} if entity_labels is not None else {}),
        "definition": definition,
        "content_version": ref["content_version"],
        "artifacts": [{"rendering": _SUMMED}],
        "coverage": {"dimensions": [_RECIPIENTS, _PROVIDERS]},
    }
    lineage = CardLineage(
        component_id=component_id,
        frame=(ref["spec_id"], ref["content_version"]),
        lineage=(ProvenanceInput(ref["spec_id"], ref["content_version"], manifest),),
    )
    return {
        "outcome": "ok",
        "frame_refs": [],
        "component": _component(component_id, ref, title, window),
        CARD_LINEAGE_KEY: lineage,
        WRITE_TOKEN_KEY: "cmw_" + "t" * 22,
    }


# The type of the cards these tests draw: a chart over a range of years.
_TYPE = "line_chart"


def _described(
    component_id: str, ref: Mapping[str, str], title: str, type_: str = _TYPE
) -> dict[str, Any]:
    return {
        "outcome": "ok",
        "frame_refs": [],
        "component_body": {
            "component_id": component_id,
            "revision": 1,
            "stored_intent": {
                "type": type_,
                "title": title,
                "select": {"stat_var": "dcid:Aid"},
                "frame_ref": dict(ref),
            },
        },
    }


class _Search:
    """Search's resolve: the France to Kenya part again, under the identity it
    is posted and that identity's spec_id."""

    def __init__(self) -> None:
        self.posted: list[dict[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Any, on_progress: ProgressCallback | None = None
    ) -> ToolCallResult:
        del tool, on_progress
        self.posted.append(dict(arguments))
        part = copy.deepcopy(_ENVELOPE["parts"][0])
        part["spec"]["spec_id"] = arguments["spec_id"]
        part["spec"]["definition"] = {
            "schema_version": 1,
            "spec_id": arguments["spec_id"],
            "identity": arguments["identity"],
        }
        envelope = {**_ENVELOPE, "parts": [part]}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: envelope})


def _first_turn() -> tuple[Any, ...]:
    return (
        search_call("aid from France to Kenya"),
        fetch_call(_SPEC, stat_vars="sum"),
        make_charts_call(chart(frame_id_for(_REF), type_=_TYPE, title=_TITLE)),
        text_part("Here is the card."),
    )


def _apply_body(component_id: str, spec_id: str, unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "turn.request",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        # The client's own count; the route reads no turn from it.
        "apply_unit": {"component_id": component_id, "spec_id": spec_id, "turn": 7, "unit": unit},
    }


def _first_results() -> list[dict[str, Any]]:
    return [
        {"outcome": "ok", "frame_refs": [_REF], "summary": "fetched"},
        _VIEW,
        _written("cmp_01unit0001", _REF, _TITLE),
    ]


async def test_a_card_drawn_on_a_converted_spec_offers_its_units(pg_engine: AsyncEngine) -> None:
    """The card's frame offers its units, and a state read gives its stored
    card the same `unit_change`, with no `replaces`."""
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        engine=pg_engine,
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 0}, headers=AUTH
        )

    created = _frame(frames, "component.created")
    assert created["unit_change"]["spec_id"] == created["provenance"]["inputs"][0]["spec_id"]
    assert created["unit_change"]["offer"]["start"] == {"currency": "USD", "prices": "current"}
    assert [c["code"] for c in created["unit_change"]["offer"]["currencies"]] == ["USD"]
    ((card,),) = [closed["cards"] for closed in read.json()["closed_since"]]
    assert card["component_id"] == created["component_id"]
    assert card["unit_change"] == created["unit_change"]
    assert "replaces" not in card


async def test_a_unit_change_redraws_the_card_in_its_slot_and_records_the_new_spec(
    pg_engine: AsyncEngine,
) -> None:
    search = _Search()
    executor = ScriptedExecutor(_first_results())
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=executor,
        tool_surface=search,  # ty: ignore[invalid-argument-type] -- the one tool the change reads
        engine=pg_engine,
    )
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        new_spec: str | None = None
        first_calls = len(executor.calls)
        # As a card drawn on a plan step's frame stores it.
        plan = PlanLink(plan_version_id="plan_a", plan_step_id="sum")
        async with pg_engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE conversation_components SET entry = entry || CAST(:link AS jsonb) "
                    "WHERE component_id = 'cmp_01unit0001'"
                ),
                {"link": json.dumps({PLAN_LINK_KEY: plan})},
            )

        async def answer(*, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
            nonlocal new_spec
            executor.calls.append({"tool": tool, "args": args})
            if tool == "describe_component":
                return _described("cmp_01unit0001", _REF, _TITLE)
            if tool == "get_data":
                new_spec = args["spec_id"]
                return {"outcome": "ok", "frame_refs": [_new_ref(new_spec)], "summary": "fetched"}
            assert tool == "write_component"
            assert new_spec is not None
            return _written("cmp_01unit0002", _new_ref(new_spec), args["intent"]["title"])

        app.state.studio.recorder._executor = answer
        frames = await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )
        read = await client.get(
            "/api/studio/state",
            params={"conversation_id": CONVERSATION, "since": 1},
            headers=AUTH,
        )

    assert new_spec is not None and new_spec != _SPEC
    calls = executor.calls[first_calls:]
    assert [call["tool"] for call in calls] == [
        "describe_component",
        "get_data",
        "write_component",
    ]
    # The frame is fetched as the card's was, its StatVars summed.
    assert calls[1]["args"] == {"spec_id": new_spec, "stat_vars": "sum"}
    assert calls[2]["args"]["intent"]["frame_ref"] == _new_ref(new_spec)
    assert calls[2]["args"]["intent"]["title"] == "France's aid to Kenya, constant 2023 US dollars"
    assert search.posted[0]["identity"]["facets"]["unit"] == {
        "state": "value",
        "request": _CONSTANT_2023,
    }

    created = _frame(frames, "component.created")
    assert created["unit_change"]["spec_id"] == new_spec
    assert created["unit_change"]["offer"]["start"] == _CONSTANT_2023
    produced = [f for f in frames if f["event"] == "activity.step" and f.get("produced")]
    assert [f["step_id"] for f in produced] == ["u3"]
    # The redrawn card names the plan step the card it replaced came from.
    assert created["created_from"] == {"step_id": "u3", **plan}
    manifest = _frame(frames, "done")["composition"]
    assert [block["component_id"] for block in manifest["blocks"]] == ["cmp_01unit0002"]
    assert [f["event"] for f in frames[-2:]] == ["suggestions", "done"]
    assert "message.complete" not in [f["event"] for f in frames]

    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert conversation.components["cmp_01unit0002"][PLAN_LINK_KEY] == plan
    record = conversation.turns[-1]
    assert record["user"] == (f"Changed {_TITLE} to constant 2023 US dollars ({new_spec}).")
    assert record["assistant"] == "The card now shows constant 2023 US dollars."
    assert [r["spec_id"] for r in record["readings"]] == [new_spec]
    assert new_spec in record["specs"]
    # The model reads the new card with no word of the swap.
    recalled = conversation.recall_part(part="canvas", component_id="cmp_01unit0002")
    assert "replaces" not in recalled

    # A state read gives the turn as its stream showed it: no reply, and the
    # change in words with no spec_id.
    shown = f"Change {_TITLE} to constant 2023 US dollars."
    (closed,) = read.json()["closed_since"]
    assert (closed["user_message"], closed["reply"]) == (shown, "")
    assert read.json()["turn"]["user_message"] == shown
    # Its stored card names the card it replaced and carries the unit offer
    # its frame did, so a page that never got the frame draws the same card.
    (card,) = closed["cards"]
    assert (card["component_id"], card["replaces"]) == ("cmp_01unit0002", "cmp_01unit0001")
    assert card["unit_change"] == created["unit_change"]


async def test_a_held_card_offers_no_unit_change(pg_engine: AsyncEngine) -> None:
    """A card kept off the canvas (`auto_add` false) gets no `unit_change`:
    the route refuses a card that holds no block."""
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        engine=pg_engine,
    )
    async with _client(app) as client:
        frames = await _stream(
            client, "/api/studio/turn", {**_turn_body("France's aid to Kenya"), "auto_add": False}
        )

    assert "unit_change" not in _frame(frames, "component.created")


async def test_a_redraw_the_platform_refuses_leaves_the_card_unchanged(
    pg_engine: AsyncEngine,
) -> None:
    """A refused write of the redrawn card ends the turn on the documented
    lead, and the canvas keeps the card it showed."""
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool the change reads
        engine=pg_engine,
    )

    async def answer(*, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
        if tool == "describe_component":
            return _described("cmp_01unit0001", _REF, _TITLE)
        if tool == "get_data":
            return {"outcome": "ok", "frame_refs": [_new_ref(args["spec_id"])], "summary": "ok"}
        return {"outcome": "refused", "refusal": {"code": "intent_invalid", "message": "no"}}

    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = answer
        frames = await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )

    assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")
    failed = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert [f["step_id"] for f in failed] == ["u3"]
    manifest = _frame(frames, "done")["composition"]
    assert [block["component_id"] for block in manifest["blocks"]] == ["cmp_01unit0001"]


async def test_a_second_unit_change_works_on_a_reopened_conversation(
    pg_engine: AsyncEngine,
) -> None:
    """The redrawn card changes unit again from another instance, which holds
    none of the first instance's process state."""
    search = _Search()
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=search,  # ty: ignore[invalid-argument-type] -- the one tool the change reads
        engine=pg_engine,
    )
    specs: list[str] = []
    ids = iter(["cmp_01unit0002", "cmp_01unit0003"])

    async def answer(*, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
        if tool == "describe_component":
            ref = _new_ref(specs[-1]) if specs else _REF
            return _described(args["component_id"], ref, "France's aid to Kenya")
        if tool == "get_data":
            specs.append(args["spec_id"])
            return {"outcome": "ok", "frame_refs": [_new_ref(args["spec_id"])], "summary": "ok"}
        return _written(next(ids), _new_ref(specs[-1]), args["intent"]["title"])

    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = answer
        first = await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )
    redrawn = _frame(first, "component.created")

    other = _app(
        resolver=CannedSearch(),
        tool_surface=search,  # ty: ignore[invalid-argument-type] -- the one tool the change reads
        engine=pg_engine,
        store=StudioStore(pg_engine),
    )
    other.state.studio.recorder._executor = answer
    unit = {"currency": "USD", "prices": "constant", "base_year": 2010}
    async with _client(other) as client:
        second = await _stream(
            client,
            "/api/studio/turn",
            _apply_body(redrawn["component_id"], redrawn["unit_change"]["spec_id"], unit),
        )

    created = _frame(second, "component.created")
    assert created["component_id"] == "cmp_01unit0003"
    assert created["unit_change"]["offer"]["start"] == unit
    manifest = _frame(second, "done")["composition"]
    assert [block["component_id"] for block in manifest["blocks"]] == ["cmp_01unit0003"]


def _new_ref(spec_id: str) -> dict[str, str]:
    return {"spec_id": spec_id, "content_version": "cv_" + "5" * 32}


_CURRENT = {"currency": "USD", "prices": "current"}


@pytest.mark.parametrize(
    ("decision", "answers", "unit"),
    [
        ("answer", {"q1": {"kind": "unit", **_CONSTANT_2023}}, _CONSTANT_2023),
        ("skip", None, _CONSTANT_2023),
        ("answer", {"q1": {"kind": "unit", **_CURRENT}}, _CURRENT),
    ],
    ids=["unit-answer", "skip", "served-unit"],
)
async def test_a_card_drawn_after_a_unit_question_offers_its_units(
    pg_engine: AsyncEngine,
    decision: str,
    answers: dict[str, Any] | None,
    unit: dict[str, Any],
) -> None:
    """A turn pauses on a unit question and resumes on its answer, or on a
    skip, which goes on in the unit the question started on. A unit the spec
    is not served in is re-read through Search, and the agent draws a card on
    the spec it is told to fetch. The card's `unit_change` names that spec and
    starts on that unit, and a unit change on it applies."""
    search = _Search()

    def spec_id() -> str:
        """The spec the answer tells the agent to fetch: the re-read, if any."""
        return search.posted[-1]["spec_id"] if search.posted else _SPEC

    def draw(_: Any) -> Any:
        return make_charts_call(
            chart(frame_id_for(_new_ref(spec_id())), type_="stat_card", title=_TITLE)
        )

    async def answer(*, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
        if tool == "describe_frame":
            return _VIEW
        if tool == "describe_component":
            return _described(args["component_id"], _new_ref(spec_id()), _TITLE)
        if tool == "get_data":
            return {"outcome": "ok", "frame_refs": [_new_ref(args["spec_id"])], "summary": "ok"}
        assert tool == "write_component"
        component_id = "cmp_01unit0002" if "unit_card" in drawn else "cmp_01unit0001"
        drawn["unit_card"] = component_id
        return _written(component_id, args["intent"]["frame_ref"], args["intent"]["title"])

    drawn: dict[str, str] = {}
    app = _app(
        search_call("aid from France to Kenya"),
        call_part("ask_user", {"questions": [{"question": "In which prices?", "unit_of": _SPEC}]}),
        lambda _: fetch_call(spec_id(), stat_vars="sum"),
        draw,
        text_part("Here is the card."),
        resolver=CannedSearch(_ENVELOPE),
        # The answer's re-read and the later unit change both resolve a definition.
        definition_surface=search,
        tool_surface=search,  # ty: ignore[invalid-argument-type] -- the one tool the change reads
        engine=pg_engine,
    )
    app.state.studio.recorder._executor = answer
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        elicitation = _frame(first, "elicitation.request")
        frames = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], decision, answers=answers),
        )
        read, reread = (
            spec_id(),
            [p["identity"]["facets"]["unit"]["request"] for p in search.posted],
        )
        created = _frame(frames, "component.created")
        changed = await _stream(
            client,
            "/api/studio/turn",
            _apply_body(
                created["component_id"], read, _CURRENT if unit != _CURRENT else _CONSTANT_2023
            ),
        )

    assert reread == ([] if unit == _CURRENT else [unit])
    assert created["unit_change"]["spec_id"] == created["provenance"]["inputs"][0]["spec_id"]
    assert created["unit_change"]["spec_id"] == read
    assert created["unit_change"]["offer"]["start"] == unit
    redrawn = _frame(changed, "component.created")
    assert redrawn["component_id"] == "cmp_01unit0002"
    assert redrawn["unit_change"]["offer"]["start"] != unit


@pytest.mark.parametrize(
    ("component_id", "spec_id", "unit", "constraint", "pointer"),
    [
        ("cmp_01gone0000", _SPEC, _CONSTANT_2023, "scope_choice_stale", "/apply_unit"),
        (
            "cmp_01unit0001",
            "spec_" + "f" * 32,
            _CONSTANT_2023,
            "scope_choice_stale",
            "/apply_unit",
        ),
        (
            "cmp_01unit0001",
            _SPEC,
            {"currency": "EUR", "prices": "current"},
            "unit_not_offered",
            "/apply_unit/unit",
        ),
        (
            "cmp_01unit0001",
            _SPEC,
            {"currency": "USD", "prices": "current"},
            "unit_not_offered",
            "/apply_unit/unit",
        ),
    ],
    ids=["card-gone", "other-spec", "currency-not-offered", "unit-it-shows"],
)
async def test_a_unit_change_that_cannot_apply_is_refused_before_the_stream(
    pg_engine: AsyncEngine,
    component_id: str,
    spec_id: str,
    unit: dict[str, Any],
    constraint: str,
    pointer: str,
) -> None:
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results(), strict=True),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- never called here
        engine=pg_engine,
    )
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        response = await client.post(
            "/api/studio/turn", json=_apply_body(component_id, spec_id, unit), headers=AUTH
        )

    assert response.status_code == 422
    body = response.json()
    assert (body["constraint_id"], body["pointer"]) == (constraint, pointer)
    assert len(app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns) == 1


_SIBLING_TITLE = "France's aid to Kenya by sector"
# The title the user gave the card through Edit Card: the platform's stored intent holds it.
_EDITED_TITLE = "What France gives Kenya, current US dollars"


def _two_cards_app(pg_engine: AsyncEngine) -> Any:
    """An app whose first turn draws two cards on one frame, the second a sibling."""
    return _app(
        search_call("aid from France to Kenya"),
        fetch_call(_SPEC, stat_vars="sum"),
        make_charts_call(
            chart(frame_id_for(_REF), type_=_TYPE, title=_TITLE),
            chart(frame_id_for(_REF), type_=_TYPE, title=_SIBLING_TITLE),
        ),
        text_part("Here are the cards."),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [_REF], "summary": "fetched"},
                _VIEW,
                _written("cmp_01unit0001", _REF, _TITLE),
                _written("cmp_01unit0009", _REF, _SIBLING_TITLE),
            ]
        ),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool the change reads
        engine=pg_engine,
    )


def _redraw(
    *, before_fetch: Any = None, before_write: Any = None, calls: list[str] | None = None
) -> Any:
    """The executor of a unit change on card cmp_01unit0001, whose stored title
    is the user's edit, redrawn as cmp_01unit0002; `before_fetch` and
    `before_write` run first when the new unit is fetched and when the redraw
    is written. `calls` collects the tools called."""

    async def answer(*, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
        if calls is not None:
            calls.append(tool)
        if tool == "describe_component":
            return _described("cmp_01unit0001", _REF, _EDITED_TITLE)
        if tool == "get_data":
            if before_fetch is not None:
                await before_fetch()
            return {"outcome": "ok", "frame_refs": [_new_ref(args["spec_id"])], "summary": "ok"}
        assert tool == "write_component"
        if before_write is not None:
            await before_write()
        return _written("cmp_01unit0002", args["intent"]["frame_ref"], args["intent"]["title"])

    return answer


@pytest.mark.parametrize(
    ("hint", "kept"),
    [
        ("row", [{"type": "row", "members": ["cmp_01unit0002", "cmp_01unit0009"]}]),
        ("comparison", []),
    ],
    ids=["row-names-the-new-card", "comparison-drops-it"],
)
async def test_a_unit_change_keeps_the_sibling_and_the_cards_own_canvas_state(
    pg_engine: AsyncEngine, hint: str, kept: list[dict[str, Any]]
) -> None:
    """Of two cards drawn by one answer, the one whose unit changes is
    replaced in its slot: `component.created` names the card it replaces, the
    new card keeps the old one's pin, note and the user's title, each in the
    new unit where it named the old, a row hint over both names the new card,
    a comparison hint, which aligns units, drops it and with it the hint, and
    the sibling stays as it was."""
    app = _two_cards_app(pg_engine)
    note = {
        "subtitle": "Current US dollars",
        "description": "Gross disbursements.",
        "y_label": "USD, current prices",
    }
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        manifest = _frame(first, "done")["composition"]
        arranged = await client.post(
            "/api/studio/composition",
            json=_mutation_body(
                manifest,
                [
                    {"op": "pin", "component_id": "cmp_01unit0001"},
                    {"op": "set_card_meta", "component_id": "cmp_01unit0001", "meta": note},
                    {
                        "op": "set_layout_hints",
                        "layout_hints": [
                            {"type": hint, "members": ["cmp_01unit0001", "cmp_01unit0009"]}
                        ],
                    },
                ],
            ),
            headers=AUTH,
        )
        assert arranged.status_code == 200
        app.state.studio.recorder._executor = _redraw()
        frames = await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )

    created = _frame(frames, "component.created")
    assert (created["component_id"], created["replaces"]) == ("cmp_01unit0002", "cmp_01unit0001")
    before = {block["component_id"]: block for block in arranged.json()["blocks"]}
    after = _frame(frames, "done")["composition"]
    new, sibling = after["blocks"]
    assert sibling == before["cmp_01unit0009"]
    assert new["component_id"] == "cmp_01unit0002"
    assert (new["pinned"], new["meta"]) == (
        True,
        {
            "subtitle": "Constant 2023 US dollars",
            "description": "Gross disbursements.",
            "y_label": "USD, constant 2023 prices",
        },
    )
    assert new["title"] == "What France gives Kenya, constant 2023 US dollars"
    assert after["layout_hints"] == kept


async def test_a_unit_change_whose_canvas_write_fails_publishes_no_card(
    pg_engine: AsyncEngine,
) -> None:
    app = _two_cards_app(pg_engine)
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = _redraw()

        async def unavailable(*_: Any, **__: Any) -> Any:
            raise DatabaseUnavailable("the canvas could not be written")

        app.state.studio.store.change_canvas = unavailable
        frames = await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )

    events = [frame["event"] for frame in frames]
    assert "component.created" not in events
    assert not [f for f in frames if f["event"] == "activity.step" and f.get("produced")]
    assert events[-1] == "error"


async def test_a_card_removed_while_its_unit_changes_is_refused_and_nothing_is_published(
    pg_engine: AsyncEngine,
) -> None:
    """The user removes the card while its new unit is read: the swap finds
    no slot under the canvas lock, so the turn ends on the stale refusal's
    sentence, and neither the redrawn card nor its digest entry appears."""
    app = _two_cards_app(pg_engine)
    store: StudioStore = app.state.studio.store

    async def remove() -> None:
        def without(manifest: CompositionManifest | None, _: Any) -> CompositionManifest:
            assert manifest is not None
            blocks = [b for b in manifest.blocks if b.component_id != "cmp_01unit0001"]
            return manifest.model_copy(update={"blocks": blocks, "revision": manifest.revision + 1})

        await store.mutate_composition(PRINCIPAL, CONVERSATION, without)

    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = _redraw(before_write=remove)
        frames = await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )

    assert "component.created" not in [frame["event"] for frame in frames]
    failed = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert [(f["step_id"], f["detail"]) for f in failed] == [("u3", UNIT_GONE_NOTE)]
    assert _frame(frames, "message.complete")["text"] == UNIT_GONE_NOTE
    blocks = _frame(frames, "done")["composition"]["blocks"]
    assert [block["component_id"] for block in blocks] == ["cmp_01unit0009"]
    conversation = store.current(PRINCIPAL, CONVERSATION)
    assert conversation is not None
    assert "cmp_01unit0002" not in conversation.components
    # The card written off the canvas stays the conversation's.
    assert "cmp_01unit0002" in conversation.write_tokens


async def test_a_card_removed_before_its_redraw_is_refused_and_never_written(
    pg_engine: AsyncEngine,
) -> None:
    """The user removes the card while its new unit is fetched: the change
    ends on the stale refusal's sentence and `write_component` is never
    called."""
    app = _two_cards_app(pg_engine)
    store: StudioStore = app.state.studio.store

    async def remove() -> None:
        def without(manifest: CompositionManifest | None, _: Any) -> CompositionManifest:
            assert manifest is not None
            blocks = [b for b in manifest.blocks if b.component_id != "cmp_01unit0001"]
            return manifest.model_copy(update={"blocks": blocks, "revision": manifest.revision + 1})

        await store.mutate_composition(PRINCIPAL, CONVERSATION, without)

    calls: list[str] = []
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = _redraw(before_fetch=remove, calls=calls)
        frames = await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )

    assert calls == ["describe_component", "get_data"]
    assert "component.created" not in [frame["event"] for frame in frames]
    failed = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert [(f["step_id"], f["detail"]) for f in failed] == [("u3", UNIT_GONE_NOTE)]
    assert _frame(frames, "message.complete")["text"] == UNIT_GONE_NOTE


_SRC = Path(__file__).resolve().parents[1] / "src"
# The frames a Studio turn pauses on: a plan card, a new version of one, and a question.
_PAUSING_FRAMES = {"PlanProposed", "PlanVersioned", "ElicitationRequest"}


def _replaces_set_in(tree: ast.Module) -> set[str]:
    """Each function of `tree` that sets a `replaces`: by a keyword, a dict
    key, a subscript store (`x["replaces"] = ...`) or an attribute store
    (`x.replaces = ...`)."""
    found: set[str] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(function):
            keyed = isinstance(node, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "replaces" for key in node.keys
            )
            stored = (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "replaces"
            ) or (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and node.attr == "replaces"
            )
            if keyed or stored or (isinstance(node, ast.keyword) and node.arg == "replaces"):
                found.add(function.name)
    return found


@pytest.mark.parametrize(
    "source",
    [
        "def f(e):\n    return e.model_copy(update={'replaces': 'c1'})",
        "def f(e):\n    return Created(replaces='c1')",
        "def f(e):\n    e['replaces'] = 'c1'",
        "def f(e):\n    e.replaces = 'c1'",
    ],
    ids=["dict-key", "keyword", "subscript-store", "attribute-store"],
)
def test_the_replaces_guard_finds_every_way_of_setting_it(source: str) -> None:
    """`_replaces_set_in` finds a `replaces` set each way Python can set one,
    so the guard below cannot miss a new route setting it."""
    assert _replaces_set_in(ast.parse(source)) == {"f"}


def test_a_card_replaces_another_only_on_the_edit_routes_which_never_pause() -> None:
    """`component.created` names the card it `replaces`, whose slot the new
    card takes. Only two functions of `harness.studio_edit` set it: `_swapped`,
    which builds the card swap of a unit change (`apply_unit`) and of an Edit
    Card rebuild, and `_swap_frames`, which writes that swap's frames. That
    module names no frame a turn pauses on. So no card the agent removed
    before a pause is replaced once the turn resumes, and the client holds
    no removal across a pause. `_swapped` also stores the replaced id with
    the new card, and the state read's `harness.studio._stored_card` gives
    that stored id back, setting nothing new."""
    setting = {
        f"{path.relative_to(_SRC)}:{name}"
        for path in sorted(_SRC.rglob("*.py"))
        for name in _replaces_set_in(ast.parse(path.read_text()))
    }
    edit = ast.parse((_SRC / "harness" / "studio_edit.py").read_text())
    named = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(edit)
        if isinstance(node, ast.Name | ast.Attribute)
    } | {
        alias.name
        for node in ast.walk(edit)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert setting == {
        "harness/studio.py:_stored_card",
        "harness/studio_edit.py:_swap_frames",
        "harness/studio_edit.py:_swapped",
    }
    assert not named & _PAUSING_FRAMES


# --- `apply_change`: a card's years and places, through the unit's own flow --


def _change_body(component_id: str, spec_id: str, change: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "turn.request",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "apply_change": {"component_id": component_id, "spec_id": spec_id, "change": dict(change)},
    }


_YEARS = {"kind": "window", "start_year": 2010, "end_year": 2024}


class _Changer:
    """The executor of a chain of changes on one card: each reads the card it
    is asked of, fetches a frame of its own for each `get_data` call, and
    redraws the card as the next id, drawing the years the fetch's scope
    names, else the years the card drew."""

    def __init__(self, *, rows: int = 12) -> None:
        self.calls: list[dict[str, Any]] = []
        self.rows = rows
        self.drawn = 1
        self.title = _TITLE
        self.type = _TYPE
        self.window = (2015, 2023)

    async def __call__(self, *, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.calls.append({"tool": tool, "args": args})
        if tool == "describe_component":
            return _described(args["component_id"], _REF, self.title, self.type)
        if tool == "get_data":
            window = (args.get("scope") or {}).get("window")
            if window is not None:
                self.window = (window["start_year"], window["end_year"])
            digest = hashlib.sha1(json.dumps(args, sort_keys=True).encode()).hexdigest()
            ref = {"spec_id": "spec_" + digest[:32], "content_version": "cv_" + "5" * 32}
            return {"outcome": "ok", "frame_refs": [ref], "summary": "ok", "rows": self.rows}
        assert tool == "write_component"
        self.drawn += 1
        self.title = args["intent"]["title"]
        return _written(
            f"cmp_01unit{self.drawn:04d}", args["intent"]["frame_ref"], self.title, self.window
        )

    def fetched(self) -> list[dict[str, Any]]:
        return [call["args"] for call in self.calls if call["tool"] == "get_data"]


def _changes_app(pg_engine: AsyncEngine) -> Any:
    return _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool a unit change reads
        engine=pg_engine,
    )


async def test_a_card_offers_its_years_and_a_change_of_years_redraws_it_over_them(
    pg_engine: AsyncEngine,
) -> None:
    """The card draws 2015 to 2023 of a spec Search serves from 1978 to 2024.
    A change of years fetches the spec over the new window, keeps the card's
    arguments, and redraws the card in its slot, retitled, with both offers."""
    app = _changes_app(pg_engine)
    changer = _Changer()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        offered = _frame(first, "component.created")["card_changes"]
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 1}, headers=AUTH
        )

    assert offered == {
        "spec_id": _SPEC,
        "window": {"start_year": 2015, "end_year": 2023, "first_year": 1978, "last_year": 2024},
    }
    assert changer.fetched() == [
        {
            "spec_id": _SPEC,
            "stat_vars": "sum",
            "scope": {"window": {"kind": "range", "start_year": 2010, "end_year": 2024}},
        }
    ]
    created = _frame(frames, "component.created")
    assert (created["component_id"], created["replaces"]) == ("cmp_01unit0002", "cmp_01unit0001")
    assert created["card_changes"]["window"]["start_year"] == 2010
    assert created["unit_change"]["spec_id"] == _SPEC
    assert [f["step_id"] for f in frames if f["event"] == "activity.step"] == ["u1", "u2", "u3"]
    assert [b["component_id"] for b in _frame(frames, "done")["composition"]["blocks"]] == [
        "cmp_01unit0002"
    ]
    ((card,),) = [closed["cards"] for closed in read.json()["closed_since"]]
    assert card["card_changes"] == created["card_changes"]


def _recorded_card_titles() -> list[str]:
    """Every card title the repository's JSON fixtures hold: the title of each
    object drawn as a card (a card type, or a `frame_ref`), with the titles
    this module's own tests and the edit tests draw cards under."""
    kinds = {"stat_card", "line_chart", "bar_chart", "table"}
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            title = node.get("title")
            title = title.get("text") if isinstance(title, dict) else title
            kind = node.get("type")
            drawn = (isinstance(kind, str) and kind in kinds) or "frame_ref" in node
            if drawn and isinstance(title, str):
                found.add(title)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in sorted(Path(__file__).parents[2].glob("*/tests/**/*.json")):
        with contextlib.suppress(ValueError):
            walk(json.loads(path.read_text()))
    return sorted(
        found
        | {
            _TITLE,
            "Aid 2020 to 2024, against 2023",
            "Aid in 2024",
            "Aid to Kenya, 2018-2024",
            "Aid to Malawi by donor (3 largest of 70 in 2024)",
            "Growth since 2019",
            "The top three donors in 2019 and in 2023",
            "Aid to Kenya, 2015/16",
            "Kenya's 2023 ODA receipts",
            "Aid to Kenya compared with Uganda",
        }
    )


_RECORDED_TITLES = _recorded_card_titles()


def _move(title: str, change: Any) -> ChangeMove:
    """A change of France's aid to Kenya, drawn 2015 to 2023, titled `title`."""
    block = _ENVELOPE["parts"][0]["spec"]
    shown = {"destination": ["country/KEN"], "origin": ["country/FRA"]}
    card = UnitCard(spec_id=_SPEC, args={"stat_vars": "sum"}, roles={"destination": False})
    offer = CardChanges(
        spec_id=_SPEC,
        window=WindowOffer(start_year=2015, end_year=2023, first_year=1978, last_year=2024),
        places=[
            PlacesOffer(
                role="destination",
                multiple=False,
                chosen=[PlaceOption(id="country/KEN", label="Kenya")],
            )
        ],
    )
    if isinstance(change, WindowChange):
        scope: dict[str, Any] = {
            "window": {
                "kind": "range",
                "start_year": change.start_year,
                "end_year": change.end_year,
            }
        }
        old: tuple[str, ...] = ()
        new: tuple[str, ...] = ()
    else:
        shown["destination"] = list(change.entities)
        scope = {"entities": {role: [f"dcid:{r}" for r in refs] for role, refs in shown.items()}}
        card = replace(card, labels={"country/UGA": "Uganda"})
        old, new = ("Kenya",), ("Uganda",)
    return ChangeMove(
        "cmp_01unit0001", title, card, change, scope, "", offer, block, old=old, new=new
    )


@pytest.mark.parametrize("title", _RECORDED_TITLES)
def test_a_changed_cards_title_names_only_its_new_years(title: str) -> None:
    """Over every recorded card title, a change to 2010 to 2024 ends the title
    on the new period, and the title names no other year."""
    retitled = _changed_title(
        title, _move(title, WindowChange(kind="window", start_year=2010, end_year=2024))
    )

    assert retitled.endswith(", 2010 to 2024")
    assert set(re.findall(r"(?<!\d)\d{4}(?!\d)", retitled)) == {"2010", "2024"}
    assert len(retitled) <= 120


@pytest.mark.parametrize("title", _RECORDED_TITLES)
def test_a_changed_cards_title_names_only_its_new_places(title: str) -> None:
    """Over every recorded card title, a change of Kenya to Uganda names
    Uganda and not Kenya, keeping the period the title ended on."""
    change = PlacesChange(kind="places", role="destination", entities=["country/UGA"])
    retitled = _changed_title(title, _move(title, change))

    assert "Uganda" in retitled
    assert "Kenya" not in retitled
    assert len(retitled) <= 120


@pytest.mark.parametrize(
    ("title", "change", "retitled"),
    [
        (
            "Gross disbursements to Nigeria for health, 2010 to 2024",
            WindowChange(kind="window", start_year=2015, end_year=2020),
            "Gross disbursements to Nigeria for health, 2015 to 2020",
        ),
        (
            "Aid to Kenya, 2015 to 2023",
            WindowChange(kind="window", start_year=2020, end_year=2020),
            "Aid to Kenya, 2020",
        ),
        (
            "Kenya's 2023 ODA receipts",
            WindowChange(kind="window", start_year=2010, end_year=2024),
            "Gross disbursements, France to Kenya, 2010 to 2024",
        ),
        (
            "Aid to Kenya, latest year",
            PlacesChange(kind="places", role="destination", entities=["country/UGA"]),
            "Aid to Uganda, latest year",
        ),
        (
            "Health aid to Kenya, 2015 to 2020",
            PlacesChange(kind="places", role="destination", entities=["country/UGA"]),
            "Health aid to Uganda, 2015 to 2020",
        ),
        (
            "Aid to Kenya compared with Uganda",
            PlacesChange(kind="places", role="destination", entities=["country/UGA"]),
            "Gross disbursements, France to Uganda, 2015 to 2023",
        ),
        (
            "Malaria grants to Ethiopia",
            WindowChange(kind="window", start_year=2010, end_year=2024),
            "Malaria grants to Ethiopia, 2010 to 2024",
        ),
        (
            "Guatemala's GDP, 1960\u20132024",
            WindowChange(kind="window", start_year=2000, end_year=2020),
            "Guatemala's GDP, 2000 to 2020",
        ),
    ],
    ids=[
        "range",
        "one-year",
        "year-inside",
        "latest-kept-on-places",
        "subject-kept-on-places",
        "place-already-named",
        "no-period",
        "dashed-range",
    ],
)
def test_a_changed_cards_title_is_built_from_its_parts(
    title: str, change: Any, retitled: str
) -> None:
    assert _changed_title(title, _move(title, change)) == retitled


class _Places:
    """Search's options door: the places of the kind each role binds, or a
    refusal when `down`."""

    def __init__(self, *, down: bool = False) -> None:
        self.asked: list[str] = []
        self.down = down

    async def __call__(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        del identity
        self.asked.append(role)
        if self.down:
            raise LiveApisError(status=503, code=None, retryable=True)
        listed = {
            "destination": (("country/KEN", "Kenya"), ("country/UGA", "Uganda")),
            "origin": (("country/FRA", "France"), ("country/DEU", "Germany")),
        }
        return listed[role], False


_UGANDA = {"kind": "places", "role": "destination", "entities": ["country/UGA"]}


def _unit_step(created: Mapping[str, Any]) -> dict[str, Any]:
    return _apply_body(created["component_id"], created["unit_change"]["spec_id"], _CONSTANT_2023)


def _years_step(created: Mapping[str, Any]) -> dict[str, Any]:
    return _change_body(created["component_id"], created["card_changes"]["spec_id"], _YEARS)


def _places_step(created: Mapping[str, Any]) -> dict[str, Any]:
    return _change_body(created["component_id"], created["card_changes"]["spec_id"], _UGANDA)


_STEPS = {"unit": _unit_step, "years": _years_step, "places": _places_step}


def _offers(created: Mapping[str, Any], step: str) -> bool:
    """Whether the card `created` announces offers the change `step`."""
    changes = created.get("card_changes") or {}
    return {
        "unit": "unit_change" in created,
        "years": "window" in changes,
        "places": any(p["role"] == "destination" for p in changes.get("places") or ()),
    }[step]


def _kept(fetched: Mapping[str, Any], step: str) -> bool:
    """Whether the retrieval `fetched` keeps what the change `step` set."""
    scope = fetched.get("scope") or {}
    return {
        "unit": fetched["spec_id"] != _SPEC,
        "years": scope.get("window") == {"kind": "range", "start_year": 2010, "end_year": 2024},
        "places": scope.get("entities")
        == {"destination": ["dcid:country/UGA"], "origin": ["dcid:country/FRA"]},
    }[step]


@pytest.mark.parametrize(
    ("first", "second"),
    [(a, b) for a in _STEPS for b in _STEPS if a != b],
)
async def test_a_second_change_keeps_the_first_and_is_offered_after_it(
    pg_engine: AsyncEngine, first: str, second: str
) -> None:
    """Each change carries the card's scope and spec on: years, places and unit
    each survive a change of another, and the card offers the second change
    after the first. The state read marks each change's turn with the card
    it named and what changed."""
    app = _changes_app(pg_engine)
    app.state.studio.place_options = _Places()
    changer = _Changer()
    named = []
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        for step in (first, second):
            created = _frame(frames, "component.created")
            assert _offers(created, step)
            named.append(created["component_id"])
            frames = await _stream(client, "/api/studio/turn", _STEPS[step](created))
        read = (
            await client.get(
                "/api/studio/state",
                params={"conversation_id": CONVERSATION, "since": 0},
                headers=AUTH,
            )
        ).json()

    marks = {"unit": {"kind": "unit"}, "years": {"kind": "window"}}
    marks["places"] = {"kind": "places", "role": "destination"}
    assert [turn.get("card_change") for turn in read["closed_since"]] == [
        None,
        {"component_id": named[0], **marks[first]},
        {"component_id": named[1], **marks[second]},
    ]
    assert read["turn"]["card_change"] == {"component_id": named[1], **marks[second]}

    first_fetch, last_fetch = changer.fetched()
    assert _kept(first_fetch, first)
    assert _kept(last_fetch, first)
    assert _kept(last_fetch, second)
    created = _frame(frames, "component.created")
    assert _offers(created, first)
    assert _offers(created, second)
    assert created["card_changes"]["spec_id"] == last_fetch["spec_id"]


async def test_a_card_offers_its_places_and_a_change_swaps_one(pg_engine: AsyncEngine) -> None:
    """France to Kenya binds one place in each of its roles, and Search is
    asked for none of them as it draws. A change of the recipient reads
    Search's places for that role once, resends the provider's place with it,
    and the title names the new recipient."""
    app = _changes_app(pg_engine)
    places = _Places()
    app.state.studio.place_options = places
    changer = _Changer()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        drawn = list(places.asked)
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _UGANDA)
        )

    assert drawn == []
    assert places.asked == ["destination"]
    # A line chart draws at most `LINE_CHART_MAX_VALUES` series.
    assert _frame(first, "component.created")["card_changes"]["places"] == [
        {
            "role": "destination",
            "multiple": True,
            "chosen": [{"id": "country/KEN", "label": "Kenya"}],
            "max": LINE_CHART_MAX_VALUES,
            "label": _RECIPIENTS["noun"],
        },
        {
            "role": "origin",
            "multiple": True,
            "chosen": [{"id": "country/FRA", "label": "France"}],
            "max": LINE_CHART_MAX_VALUES,
        },
    ]
    (fetched,) = changer.fetched()
    assert fetched["scope"] == {
        "entities": {"destination": ["dcid:country/UGA"], "origin": ["dcid:country/FRA"]}
    }
    created = _frame(frames, "component.created")
    assert created["replaces"] == "cmp_01unit0001"
    write = next(call for call in changer.calls if call["tool"] == "write_component")
    assert write["args"]["intent"]["title"] == "France's aid to Uganda, current US dollars"
    (destination, _) = created["card_changes"]["places"]
    assert destination["chosen"] == [{"id": "country/UGA", "label": "Uganda"}]


class _Asked(_Places):
    """Search's options door, keeping the identity each read posts."""

    def __init__(self) -> None:
        super().__init__()
        self.identities: list[dict[str, Any]] = []

    async def __call__(
        self, *, identity: Mapping[str, Any], role: str
    ) -> tuple[tuple[tuple[str, str], ...], bool]:
        self.identities.append(dict(identity))
        return await super().__call__(identity=identity, role=role)


async def test_a_cards_places_are_read_over_the_years_and_places_it_shows(
    pg_engine: AsyncEngine,
) -> None:
    """After a change of years, the places door is read for the card as it
    stands, by the picker and by a change of places alike: the years and
    places its scope names, where Search filters one role's places by the
    other's and by the years."""
    app = _changes_app(pg_engine)
    door = _Asked()
    app.state.studio.place_options = door
    changer = _Changer()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        years = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )
        created = _frame(years, "component.created")
        picker = await client.get(
            "/api/studio/places",
            params=_places_query(component_id=created["component_id"]),
            headers=AUTH,
        )
        await _stream(
            client, "/api/studio/turn", _change_body(created["component_id"], _SPEC, _UGANDA)
        )

    assert picker.status_code == 200
    window = {"kind": "range", "start_year": 2010, "end_year": 2024}
    assert [(i["requested_window"], i["entities"]) for i in door.identities] == [
        (window, {"destination": ["country/KEN"], "origin": ["country/FRA"]})
    ] * 2


class _ScopedSearch(_Search):
    """Search's resolve, whose answer for France to Uganda holds data from
    1999 to 2020 only, or, with `unknown`, states no years for it."""

    def __init__(self, *, unknown: bool = False) -> None:
        super().__init__()
        self.unknown = unknown

    async def call_tool(
        self, *, tool: ToolId, arguments: Any, on_progress: ProgressCallback | None = None
    ) -> ToolCallResult:
        result = await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        if (arguments["identity"].get("entities") or {}).get("destination") != ["country/UGA"]:
            return result
        envelope = copy.deepcopy((result.meta or {})[SEARCH_ENVELOPE_META_KEY])
        resolved = envelope["parts"][0]["spec"]["resolved"]
        resolved.pop("download", None)
        resolved.pop("coverage", None)
        if not self.unknown:
            resolved["years_with_data"] = {"start_year": 1999, "end_year": 2020}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: envelope})


@pytest.mark.parametrize(
    ("unknown", "window"), [(False, (1999, 2023)), (True, None)], ids=["read", "unknown"]
)
async def test_after_a_change_of_places_the_years_offered_are_the_new_places(
    pg_engine: AsyncEngine, unknown: bool, window: tuple[int, int] | None
) -> None:
    """Kenya's data runs from 1978, Uganda's from 1999 to 2020. After the card
    moves to Uganda, one Search read states Uganda's years, and the card
    offers them, widened to the years it draws; where Search states none, it
    offers no years."""
    search = _ScopedSearch(unknown=unknown)
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=search,  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
    )
    app.state.studio.place_options = _Places()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = _Changer()
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _UGANDA)
        )

    offered = _frame(frames, "component.created")["card_changes"].get("window")
    assert (
        (offered["first_year"], offered["last_year"]) if offered is not None else None
    ) == window
    assert [posted["identity"]["entities"]["destination"] for posted in search.posted] == [
        ["country/UGA"]
    ]


class _Overlap:
    """Two calls that each wait, up to a second, until the other has started:
    both finish only when they run at the same time."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.both = asyncio.Event()

    async def run(self, name: str) -> None:
        self.started.append(name)
        if len(self.started) == 2:
            self.both.set()
        await asyncio.wait_for(self.both.wait(), timeout=1)
        self.finished.append(name)


class _OverlappingSearch(_ScopedSearch):
    def __init__(self, overlap: _Overlap) -> None:
        super().__init__()
        self.overlap = overlap

    async def call_tool(
        self, *, tool: ToolId, arguments: Any, on_progress: ProgressCallback | None = None
    ) -> ToolCallResult:
        await self.overlap.run("years")
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


class _OverlappingChanger(_Changer):
    def __init__(self, overlap: _Overlap, *, rows: int = 12) -> None:
        super().__init__(rows=rows)
        self.overlap = overlap

    async def __call__(self, *, tool: str, args: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if tool == "get_data":
            await self.overlap.run("fetch")
        return await super().__call__(tool=tool, args=args, **kw)


@pytest.mark.parametrize("rows", [12, 0], ids=["fetched", "empty"])
async def test_a_change_of_places_reads_the_new_places_years_while_it_fetches(
    pg_engine: AsyncEngine, rows: int
) -> None:
    """A change of places reads the years Search holds for the new places
    alongside the data, so both start before either ends. A fetch that comes
    back empty ends the change, and the card keeps its offer."""
    overlap = _Overlap()
    search = _OverlappingSearch(overlap)
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=search,  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
    )
    app.state.studio.place_options = _Places()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = _OverlappingChanger(overlap, rows=rows)
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _UGANDA)
        )
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 0}, headers=AUTH
        )

    assert sorted(overlap.started) == ["fetch", "years"]
    assert sorted(overlap.finished) == ["fetch", "years"]
    if rows:
        assert _frame(frames, "component.created")["card_changes"]["window"]["first_year"] == 1999
        return
    assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")
    (card,) = read.json()["closed_since"][0]["cards"]
    assert card["component_id"] == "cmp_01unit0001"
    assert card["card_changes"] == _frame(first, "component.created")["card_changes"]


# The facets the data API elected for France's aid to Kenya, as the drawn
# series' coordinates name them.
_ELECTED = {
    "provenance": "provenance/OECDDAC2A",
    "measurementMethod": "GrantEquivalent",
    "unit": "CurrencyUSD",
}


def _faceted(outcome: dict[str, Any], facets: Mapping[str, str]) -> dict[str, Any]:
    """`outcome`, a `write_component` outcome, its series counted under `facets`."""
    for series in outcome["component"]["fact_summary"]["series"]:
        series["coordinates"] = {"destination": "country/KEN", **facets}
    return outcome


class _FacetedChanger(_Changer):
    """A `_Changer` whose redrawn cards count their series under `facets`."""

    def __init__(self, facets: Mapping[str, str]) -> None:
        super().__init__()
        self.facets = dict(facets)

    async def __call__(self, *, tool: str, args: dict[str, Any], **kw: Any) -> dict[str, Any]:
        outcome = await super().__call__(tool=tool, args=args, **kw)
        return _faceted(outcome, self.facets) if tool == "write_component" else outcome


@pytest.mark.parametrize(
    ("redrawn", "lands"),
    [(_ELECTED, True), ({**_ELECTED, "measurementMethod": "Flows"}, False)],
    ids=["same-facets", "another-method"],
)
@pytest.mark.parametrize("change", [_YEARS, _UGANDA], ids=["years", "places"])
@pytest.mark.parametrize("continued", [False, True], ids=["one-method", "continued"])
async def test_a_change_keeps_the_facets_the_card_was_drawn_with(
    pg_engine: AsyncEngine,
    change: dict[str, Any],
    redrawn: dict[str, str],
    lands: bool,
    continued: bool,
) -> None:
    """The fetch binds the source, method and unit the card was drawn with, so
    the data API elects none again. A redraw counting another one is refused,
    and the card stays. A retrieval whose series the data API continued with
    its source's estimates (`trace.continuations`) counts each such series
    under the method it continues, and a bound method stops the continuation,
    so its fetch leaves the method to the data API, and the redraw is still
    held to the method the card drew."""
    first_results = _first_results()
    first_results[-1] = _faceted(first_results[-1], _ELECTED)
    if continued:
        (member,) = first_results[-1][CARD_LINEAGE_KEY].lineage
        member.manifest["trace"] = {
            "continuations": [
                {"axis": "measurementMethod", "method": "Estimated", "periods": ["2024"]}
            ]
        }
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(first_results),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
    )
    app.state.studio.place_options = _Places()
    changer = _FacetedChanger(redrawn)
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, change)
        )

    (fetched,) = changer.fetched()
    bound = {axis: {"ref": ref} for axis, ref in _ELECTED.items()}
    if continued:
        bound.pop("measurementMethod")
    assert fetched["scope"]["facets"] == bound
    created = [f for f in frames if f["event"] == "component.created"]
    assert bool(created) is lands
    if not lands:
        assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")
        blocks = app.state.studio.store.current(PRINCIPAL, CONVERSATION).composition.blocks
        assert [b.component_id for b in blocks] == ["cmp_01unit0001"]


async def test_a_unit_change_binds_the_source_and_method_the_card_was_drawn_with(
    pg_engine: AsyncEngine,
) -> None:
    """Search reads the card's spec in the new unit with its source and
    method bound, and the unit left to the request."""
    search = _Search()
    first_results = _first_results()
    first_results[-1] = _faceted(first_results[-1], _ELECTED)
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(first_results),
        tool_surface=search,  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
    )
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = _FacetedChanger(_ELECTED)
        await _stream(
            client, "/api/studio/turn", _apply_body("cmp_01unit0001", _SPEC, _CONSTANT_2023)
        )

    (posted,) = search.posted
    facets = posted["identity"]["facets"]
    assert {axis: facets[axis] for axis in ("provenance", "measurementMethod")} == {
        "provenance": {"state": "value", "ref": "provenance/OECDDAC2A"},
        "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
    }
    assert facets["unit"]["request"] == _CONSTANT_2023


def _places_query(
    component_id: str = "cmp_01unit0001", role: str = "destination"
) -> dict[str, str]:
    return {"conversation_id": CONVERSATION, "component_id": component_id, "role": role}


async def test_the_places_a_role_can_change_to_are_read_when_asked(pg_engine: AsyncEngine) -> None:
    app = _changes_app(pg_engine)
    places = _Places()
    app.state.studio.place_options = places
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        read = await client.get("/api/studio/places", params=_places_query(), headers=AUTH)

    assert read.status_code == 200
    assert read.json() == {
        "component_id": "cmp_01unit0001",
        "role": "destination",
        "options": [
            {"id": "country/KEN", "label": "Kenya"},
            {"id": "country/UGA", "label": "Uganda"},
        ],
        "truncated": False,
    }
    assert places.asked == ["destination"]


@pytest.mark.parametrize(
    ("query", "door", "answer"),
    [
        (
            {"conversation_id": "conv_other"},
            _Places(),
            (404, "not_found", None, "/conversation_id"),
        ),
        (
            {"component_id": "cmp_01gone0000"},
            _Places(),
            (422, "constraint_violation", "scope_choice_stale", "/component_id"),
        ),
        (
            {"role": "observationAbout"},
            _Places(),
            (422, "constraint_violation", "change_not_offered", "/role"),
        ),
        (
            {},
            _Places(down=True),
            (503, "transient_unavailable", "place_options_unavailable", "/role"),
        ),
        ({}, None, (503, "capability_unavailable", None, None)),
    ],
    ids=["unknown-conversation", "card-gone", "role-not-offered", "door-down", "door-off"],
)
async def test_a_places_read_that_cannot_answer_is_refused(
    pg_engine: AsyncEngine,
    query: dict[str, str],
    door: _Places | None,
    answer: tuple[int, str, str | None, str | None],
) -> None:
    app = _changes_app(pg_engine)
    app.state.studio.place_options = door
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        read = await client.get(
            "/api/studio/places", params={**_places_query(), **query}, headers=AUTH
        )

    body = read.json()
    assert (read.status_code, body["error"], body.get("constraint_id"), body.get("pointer")) == (
        answer
    )


class _NoDoorApis:
    """An apis with no options route: the route answers 404 and the
    capabilities document lists no options door."""

    async def read_entity_options(self, **_: Any) -> Any:
        raise LiveApisError(status=404, code="not_found", retryable=False)

    async def read_search_doors(self) -> frozenset[str]:
        return frozenset({"stated_scope"})


async def test_an_apis_serving_no_options_door_stops_the_cards_offering_places(
    pg_engine: AsyncEngine,
) -> None:
    """A change of places meets the 404 and answers `capability_unavailable`,
    as the places read does after it, and the card offers its years and no
    places."""
    app = _changes_app(pg_engine)
    app.state.studio.place_options = ApisPlaceOptions(
        base_url="http://apis.test",
        client=_NoDoorApis(),  # ty: ignore[invalid-argument-type] -- the two reads the reader makes
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        # The change meets the 404 first, then the places read finds the
        # offers stopped.
        changed = await client.post(
            "/api/studio/turn", json=_change_body("cmp_01unit0001", _SPEC, _UGANDA), headers=AUTH
        )
        reads = [
            changed,
            await client.get("/api/studio/places", params=_places_query(), headers=AUTH),
        ]
        state = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 0}, headers=AUTH
        )

    assert "places" in _frame(first, "component.created")["card_changes"]
    assert [(r.status_code, r.json()["error"]) for r in reads] == [
        (503, "capability_unavailable"),
        (503, "capability_unavailable"),
    ]
    (card,) = state.json()["closed_since"][0]["cards"]
    assert "places" not in card["card_changes"]
    assert "window" in card["card_changes"]


async def test_a_change_of_places_whose_door_cannot_be_read_is_refused(
    pg_engine: AsyncEngine,
) -> None:
    """The places a change names are checked against Search's list before
    the stream, and a door that cannot be read refuses the change, to retry."""
    app = _changes_app(pg_engine)
    app.state.studio.place_options = _Places()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.place_options = _Places(down=True)
        response = await client.post(
            "/api/studio/turn", json=_change_body("cmp_01unit0001", _SPEC, _UGANDA), headers=AUTH
        )

    body = response.json()
    assert (response.status_code, body["error"], body["constraint_id"], body["pointer"]) == (
        503,
        "transient_unavailable",
        "place_options_unavailable",
        "/apply_change",
    )
    assert len(app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns) == 1


async def test_a_card_offers_no_places_with_the_door_off(pg_engine: AsyncEngine) -> None:
    """With the places door off by the deployment's setting, a card offers no
    places, and a change of places answers as the picker does, 503
    `capability_unavailable`, with the conversation and its card as they were."""
    app = _changes_app(pg_engine)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        response = await client.post(
            "/api/studio/turn", json=_change_body("cmp_01unit0001", _SPEC, _UGANDA), headers=AUTH
        )

    assert "places" not in _frame(first, "component.created")["card_changes"]
    assert (response.status_code, response.json()["error"]) == (503, "capability_unavailable")
    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert len(conversation.turns) == 1
    assert [b.component_id for b in conversation.composition.blocks] == ["cmp_01unit0001"]


@pytest.mark.parametrize(
    ("entities", "read"),
    [
        (["country/TZA"], ["destination"]),
        (["country/KEN"], []),
        ([f"country/C{n}" for n in range(LINE_CHART_MAX_VALUES + 1)], []),
    ],
    ids=["unlisted", "the-place-it-shows", "more-than-the-chart-draws"],
)
async def test_a_change_of_places_outside_the_offer_is_refused(
    pg_engine: AsyncEngine, entities: list[str], read: list[str]
) -> None:
    """A place Search does not list is refused once its list is read; the
    places the card shows, and more than its chart draws, before any read.
    The conversation and its card stay as they were."""
    app = _changes_app(pg_engine)
    places = _Places()
    app.state.studio.place_options = places
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        response = await client.post(
            "/api/studio/turn",
            json=_change_body("cmp_01unit0001", _SPEC, {**_UGANDA, "entities": entities}),
            headers=AUTH,
        )

    assert response.status_code == 422
    assert (response.json()["constraint_id"], response.json()["pointer"]) == (
        "change_not_offered",
        "/apply_change",
    )
    assert places.asked == read
    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert len(conversation.turns) == 1
    assert [b.component_id for b in conversation.composition.blocks] == ["cmp_01unit0001"]


@pytest.mark.parametrize(
    ("component_id", "spec_id", "change", "constraint"),
    [
        ("cmp_01gone0000", _SPEC, _YEARS, "scope_choice_stale"),
        ("cmp_01unit0001", "spec_" + "f" * 32, _YEARS, "scope_choice_stale"),
        (
            "cmp_01unit0001",
            _SPEC,
            {"kind": "window", "start_year": 1950, "end_year": 2024},
            "change_not_offered",
        ),
        (
            "cmp_01unit0001",
            _SPEC,
            {"kind": "window", "start_year": 2015, "end_year": 2023},
            "change_not_offered",
        ),
        (
            "cmp_01unit0001",
            _SPEC,
            {"kind": "window", "start_year": 1990, "end_year": 1992},
            "change_no_data",
        ),
    ],
    ids=[
        "card-gone",
        "other-spec",
        "years-outside",
        "years-it-shows",
        "no-data",
    ],
)
async def test_a_change_that_cannot_apply_is_refused_before_the_stream_and_the_card_stays(
    pg_engine: AsyncEngine,
    component_id: str,
    spec_id: str,
    change: dict[str, Any],
    constraint: str,
) -> None:
    """The spec holds no figures from 1990 to 1992 (`years_with_data`)."""
    envelope = copy.deepcopy(_ENVELOPE)
    envelope["parts"][0]["spec"]["resolved"]["years_with_data"] = {
        "start_year": 1978,
        "end_year": 2024,
        "gap_years": [1990, 1991, 1992],
    }
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(envelope),
        executor=ScriptedExecutor(_first_results(), strict=True),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- never called here
        engine=pg_engine,
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        response = await client.post(
            "/api/studio/turn", json=_change_body(component_id, spec_id, change), headers=AUTH
        )

    assert _frame(first, "component.created")["card_changes"]["window"]["missing_years"] == [
        1990,
        1991,
        1992,
    ]
    assert response.status_code == 422
    body = response.json()
    assert (body["constraint_id"], body["pointer"]) == (constraint, "/apply_change")
    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert len(conversation.turns) == 1
    assert [b.component_id for b in conversation.composition.blocks] == ["cmp_01unit0001"]


async def test_a_change_whose_data_comes_back_empty_leaves_the_card_and_its_offers(
    pg_engine: AsyncEngine,
) -> None:
    """A retrieval that holds no rows for the new years is refused: no card is
    written, and the card on the canvas keeps its offers."""
    app = _changes_app(pg_engine)
    changer = _Changer(rows=0)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 0}, headers=AUTH
        )

    assert "component.created" not in [f["event"] for f in frames]
    assert "write_component" not in [call["tool"] for call in changer.calls]
    failed = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert [f["step_id"] for f in failed] == ["u2"]
    assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")
    first_card = _frame(first, "component.created")
    (card,) = read.json()["closed_since"][0]["cards"]
    assert card["component_id"] == "cmp_01unit0001"
    assert card["card_changes"] == first_card["card_changes"]
    assert card["unit_change"] == first_card["unit_change"]


@pytest.mark.parametrize(
    ("change", "meta", "rewritten"),
    [
        (
            _YEARS,
            {
                "subtitle": "Gross disbursements, 2015 to 2023",
                "description": "Aid has grown since 2015.",
                "x_label": "Year",
                "y_label": "USD, current prices",
            },
            {"x_label": "Year", "y_label": "USD, current prices"},
        ),
        (
            _UGANDA,
            {
                "subtitle": "What Kenya receives from France",
                "description": "Kenya's receipts set beside Kenya's needs.",
                "x_label": "Kenya",
                "y_label": "USD, current prices",
            },
            {"y_label": "USD, current prices"},
        ),
    ],
    ids=["years", "places"],
)
async def test_a_change_of_years_or_places_drops_the_card_meta_that_may_name_the_old_ones(
    pg_engine: AsyncEngine,
    change: dict[str, Any],
    meta: dict[str, str],
    rewritten: dict[str, str],
) -> None:
    """The model's subtitle and description, written about the old years or
    places, are dropped, and an axis label is kept only when it names no year
    and no place the card showed, so the chart shows what it derives from its
    data there. A `comparison` hint, which aligns its members' units, keeps a
    card whose unit did not change. The state read's card states the new
    card's facts."""
    app = _two_cards_app(pg_engine)
    app.state.studio.place_options = _Places()
    changer = _Changer()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        arranged = await client.post(
            "/api/studio/composition",
            json=_mutation_body(
                _frame(first, "done")["composition"],
                [
                    {"op": "set_card_meta", "component_id": "cmp_01unit0001", "meta": meta},
                    {
                        "op": "set_layout_hints",
                        "layout_hints": [
                            {"type": "comparison", "members": ["cmp_01unit0001", "cmp_01unit0009"]}
                        ],
                    },
                ],
            ),
            headers=AUTH,
        )
        assert arranged.status_code == 200
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, change)
        )
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 1}, headers=AUTH
        )

    after = _frame(frames, "done")["composition"]
    new, _ = after["blocks"]
    assert (new["component_id"], new["meta"]) == ("cmp_01unit0002", rewritten)
    assert after["layout_hints"] == [
        {"type": "comparison", "members": ["cmp_01unit0002", "cmp_01unit0009"]}
    ]
    ((card,),) = [closed["cards"] for closed in read.json()["closed_since"]]
    assert card["fact_summary"] == _frame(frames, "component.created")["fact_summary"]
    if change is _YEARS:
        assert card["fact_summary"]["window"] == {"start": "2010", "end": "2024"}


def _one_year_app(pg_engine: AsyncEngine, year: int) -> Any:
    """A first turn drawing France's aid to Kenya in `year` alone."""
    return _app(
        search_call("aid from France to Kenya"),
        fetch_call(_SPEC, stat_vars="sum"),
        make_charts_call(chart(frame_id_for(_REF), type_="stat_card", title=_TITLE)),
        text_part("Here is the card."),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [_REF], "summary": "fetched"},
                _VIEW,
                _written("cmp_01unit0001", _REF, f"Aid to Kenya, {year}", (year, year)),
            ]
        ),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool a unit change reads
        engine=pg_engine,
    )


@pytest.mark.parametrize("latest", [False, True], ids=["one-year", "asked-for-the-latest-year"])
async def test_a_card_showing_one_year_moves_to_another_year(
    pg_engine: AsyncEngine, latest: bool
) -> None:
    """A card drawing 2023 alone offers its years with `start_year` equal to
    `end_year`, and a change to 2020 alone fetches that one year as a range.
    A card asked for its latest year drops that window from its intent, so
    the redrawn card shows 2020 and not the latest year again."""
    app = _one_year_app(pg_engine, 2023)
    changer = _Changer()
    changer.title = "Aid to Kenya, 2023"
    changer.type = "stat_card"
    if latest:
        latest_window = {
            "start": {"anchor": "latest", "offset": 0},
            "end": {"anchor": "latest", "offset": 0},
        }
        described = changer.__call__

        async def answer(*, tool: str, args: dict[str, Any], **kw: Any) -> dict[str, Any]:
            result = await described(tool=tool, args=args, **kw)
            if tool == "describe_component":
                result["component_body"]["stored_intent"]["window"] = latest_window
            return result

        executor: Any = answer
    else:
        executor = changer
    one_year = {"kind": "window", "start_year": 2020, "end_year": 2020}
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = executor
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, one_year)
        )

    assert _frame(first, "component.created")["card_changes"]["window"] == {
        "start_year": 2023,
        "end_year": 2023,
        "first_year": 1978,
        "last_year": 2024,
        "single": True,
    }
    (fetched,) = changer.fetched()
    assert fetched["scope"] == {"window": {"kind": "range", "start_year": 2020, "end_year": 2020}}
    write = next(call for call in changer.calls if call["tool"] == "write_component")
    assert "window" not in write["args"]["intent"]
    assert write["args"]["intent"]["title"] == "Aid to Kenya, 2020"
    window = _frame(frames, "component.created")["card_changes"]["window"]
    assert (window["start_year"], window["end_year"]) == (2020, 2020)


@pytest.mark.parametrize("rows", [12, 0], ids=["held", "no-figures"])
async def test_a_card_showing_one_year_changes_place_at_that_year(
    pg_engine: AsyncEngine, rows: int
) -> None:
    """A key number for 2023 moved to Uganda fetches Uganda at 2023 alone, so
    the value drawn is the year its title keeps, whatever Uganda's latest
    year is. Where Uganda holds no figures for 2023, the change ends as a
    retrieval with no figures, and the card keeps its slot and offers."""
    app = _one_year_app(pg_engine, 2023)
    app.state.studio.place_options = _Places()
    changer = _Changer(rows=rows)
    changer.title = "Aid to Kenya, 2023"
    changer.type = "stat_card"
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _UGANDA)
        )
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 0}, headers=AUTH
        )

    shown = _frame(first, "component.created")["card_changes"]["window"]["start_year"]
    (fetched,) = changer.fetched()
    window = {"kind": "range", "start_year": shown, "end_year": shown}
    assert fetched["scope"]["window"] == window
    if rows:
        created = _frame(frames, "component.created")
        assert created["card_changes"]["window"]["start_year"] == shown
        assert created["payload"]["component"]["title"]["text"].endswith(str(shown))
        return
    assert "write_component" not in [call["tool"] for call in changer.calls]
    assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")
    (card,) = read.json()["closed_since"][0]["cards"]
    assert card["card_changes"] == _frame(first, "component.created")["card_changes"]


async def test_a_card_showing_one_value_refuses_a_range_and_one_place_too_many(
    pg_engine: AsyncEngine,
) -> None:
    """A key number shows one value: its years change one at a time, and each
    role takes one place."""
    app = _one_year_app(pg_engine, 2023)
    app.state.studio.place_options = _Places()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        refused = [
            await client.post(
                "/api/studio/turn", json=_change_body("cmp_01unit0001", _SPEC, change), headers=AUTH
            )
            for change in (
                {"kind": "window", "start_year": 2010, "end_year": 2020},
                {**_UGANDA, "entities": ["country/UGA", "country/KEN"]},
            )
        ]

    offered = _frame(first, "component.created")["card_changes"]
    assert offered["window"]["single"] is True
    assert {p["role"]: p["multiple"] for p in offered["places"]} == {
        "destination": False,
        "origin": False,
    }
    assert [(r.status_code, r.json()["constraint_id"]) for r in refused] == [
        (422, "change_not_offered"),
        (422, "change_not_offered"),
    ]


async def test_a_chart_cut_to_one_place_can_add_places_back(pg_engine: AsyncEngine) -> None:
    """Whether a role takes several places is the chart's, fixed when it is
    drawn. A chart of one recipient cut to another keeps `multiple`, and a
    change adding a place back is drawn, its `select` naming every place."""
    app = _changes_app(pg_engine)
    app.state.studio.place_options = _Places()
    changer = _Changer()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        cut = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _UGANDA)
        )
        created = _frame(cut, "component.created")
        both = {**_UGANDA, "entities": ["country/UGA", "country/KEN"]}
        added = await _stream(
            client,
            "/api/studio/turn",
            _change_body(created["component_id"], created["card_changes"]["spec_id"], both),
        )

    (destination, _) = created["card_changes"]["places"]
    assert destination["multiple"] is True
    (_, fetched) = changer.fetched()
    assert fetched["scope"]["entities"]["destination"] == ["dcid:country/UGA", "dcid:country/KEN"]
    (destination, _) = _frame(added, "component.created")["card_changes"]["places"]
    assert [o["label"] for o in destination["chosen"]] == ["Uganda", "Kenya"]


@pytest.mark.parametrize(
    ("select", "entities", "reselected"),
    [
        (
            {"destination": ["dcid:country/KEN", "dcid:country/UGA"]},
            ["country/KEN", "country/UGA", "country/TZA"],
            {"destination": ["dcid:country/KEN", "dcid:country/UGA", "dcid:country/TZA"]},
        ),
        (
            {"destination": ["dcid:country/KEN", "dcid:country/UGA"]},
            ["country/UGA"],
            {"destination": ["dcid:country/UGA"]},
        ),
        ({"destination": "country/KEN"}, ["country/UGA"], {"destination": "country/UGA"}),
        ({"stat_var": "dcid:Aid"}, ["country/UGA"], {"stat_var": "dcid:Aid"}),
        (
            {"destination": "dcid:country/KEN", "origin": "dcid:country/KEN"},
            ["country/UGA"],
            {"destination": "dcid:country/UGA", "origin": "dcid:country/KEN"},
        ),
    ],
    ids=["add-a-place", "remove-a-place", "swap-one", "other-entry", "other-role-same-place"],
)
def test_a_change_of_places_names_the_new_places_in_the_select(
    select: dict[str, Any], entities: list[str], reselected: dict[str, Any]
) -> None:
    change = PlacesChange(kind="places", role="destination", entities=entities)
    move = _move("Aid to Kenya", change)
    move = replace(
        move,
        offer=move.offer.model_copy(
            update={
                "places": [
                    PlacesOffer(
                        role="destination",
                        multiple=True,
                        chosen=[
                            PlaceOption(id="country/KEN", label="Kenya"),
                            PlaceOption(id="country/UGA", label="Uganda"),
                        ],
                    )
                ]
            }
        ),
    )
    stored = {"type": "line_chart", "title": "Aid", "select": select}

    intent, _ = _redrawn_intent(stored, {"spec_id": "s", "content_version": "c"}, move)

    assert intent["select"] == reselected


async def test_the_next_turns_model_reads_the_scope_a_changed_card_shows(
    pg_engine: AsyncEngine,
) -> None:
    """After a change of places, the turn's record and the card's digest
    entry name the `get_data` arguments that draw the card, scope included."""
    app = _changes_app(pg_engine)
    app.state.studio.place_options = _Places()
    changer = _Changer()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _UGANDA)
        )

    (fetched,) = changer.fetched()
    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    new_id = _frame(frames, "component.created")["component_id"]
    said = json.dumps(fetched, sort_keys=True)
    assert said in conversation.turns[-1]["user"]
    model_facing = json.dumps(conversation.recall_part(part="canvas", component_id=new_id))
    assert "dcid:country/UGA" in model_facing


class _Recreate:
    """An Edit Card model that rewrites the card as a table."""

    async def __call__(self, **_: Any) -> dict[str, Any]:
        return {
            "text": "Made it a table.",
            "patch": None,
            "recreate": {"type": "table", "title": "Aid to Uganda", "select": {"s": "v"}},
        }


async def test_an_edit_card_rebuild_keeps_the_places_a_change_set(pg_engine: AsyncEngine) -> None:
    """A card changed to Uganda and then rebuilt through Edit Card reads what
    the changed card read: its scope, Uganda's label and its encoding."""
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
        edit_model=_Recreate(),
    )
    app.state.studio.place_options = _Places()
    changer = _Changer()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _UGANDA)
        )
        changed = _frame(frames, "component.created")
        write = next(c for c in changer.calls if c["tool"] == "write_component")
        frame_ref = write["args"]["intent"]["frame_ref"]

        async def edit(*, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
            if tool == "describe_component":
                return _described(changed["component_id"], frame_ref, "Aid to Uganda")
            if tool == "write_component":
                return _written("cmp_01unit0009", frame_ref, args["intent"]["title"])
            return {"outcome": "ok", "frame_refs": []}

        app.state.studio.recorder._executor = edit
        body = _turn_body("make it a table")
        body["component_ref"] = {
            "component_id": changed["component_id"],
            "revision": 1,
            "intent": "edit",
        }
        rebuilt = await _stream(client, "/api/studio/turn", body)

    created = _frame(rebuilt, "component.created")
    assert created["replaces"] == changed["component_id"]
    places = {p["role"]: p for p in created["card_changes"]["places"]}
    assert places["destination"]["chosen"] == [{"id": "country/UGA", "label": "Uganda"}]
    assert places["destination"]["multiple"] is True


class _ToUganda:
    """An Edit Card model that narrows the card's recipients to Uganda, in
    place."""

    async def __call__(self, **_: Any) -> dict[str, Any]:
        select = {"stat_var": "dcid:Aid", "financingRecipient": ["country/UGA"]}
        return {"text": "Uganda alone.", "patch": {"select": select}, "recreate": None}


async def test_an_edit_card_revise_offers_the_places_the_revised_card_draws(
    pg_engine: AsyncEngine,
) -> None:
    """A card on Kenya and Uganda narrowed to Uganda through Edit Card, in
    place. Its stored offers follow the revised card: the recipient shows
    Uganda alone, so a change back to Kenya and Uganda is applied, and a
    change to Uganda alone is refused as the card already showing it."""
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
        edit_model=_ToUganda(),
    )
    app.state.studio.place_options = _Places()
    changer = _Changer()
    both = {**_UGANDA, "entities": ["country/KEN", "country/UGA"]}
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, both)
        )
        changed = _frame(frames, "component.created")
        drawn = {p["role"]: p for p in changed["card_changes"]["places"]}["destination"]
        assert [o["id"] for o in drawn["chosen"]] == both["entities"]
        write = next(c for c in changer.calls if c["tool"] == "write_component")
        frame_ref = write["args"]["intent"]["frame_ref"]

        async def edit(*, tool: str, args: dict[str, Any], **_: Any) -> dict[str, Any]:
            if tool == "describe_component":
                return _described(changed["component_id"], frame_ref, "Aid to Kenya and Uganda")
            if tool == "write_component":
                revised = _written(changed["component_id"], frame_ref, "Aid to Uganda")
                revised["component"]["revision"] = 2
                revised["component"]["payload"]["revision"] = 2
                revised["component"]["fact_summary"]["revision"] = 2
                return revised
            return {"outcome": "ok", "frame_refs": []}

        app.state.studio.recorder._executor = edit
        body = _turn_body("only Uganda")
        body["component_ref"] = {
            "component_id": changed["component_id"],
            "revision": 1,
            "intent": "edit",
        }
        revised = await _stream(client, "/api/studio/turn", body)
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 0}, headers=AUTH
        )
        app.state.studio.recorder._executor = changer
        same = await client.post(
            "/api/studio/turn",
            json=_change_body(changed["component_id"], _SPEC, _UGANDA),
            headers=AUTH,
        )
        back = await _stream(
            client, "/api/studio/turn", _change_body(changed["component_id"], _SPEC, both)
        )

    assert _frame(revised, "component.revised")["revision"] == 2
    cards = {c["component_id"]: c for t in read.json()["closed_since"] for c in t["cards"]}
    offered = {p["role"]: p for p in cards[changed["component_id"]]["card_changes"]["places"]}
    assert [o["id"] for o in offered["destination"]["chosen"]] == ["country/UGA"]
    assert (
        _frame(revised, "component.revised")["card_changes"]
        == (cards[changed["component_id"]]["card_changes"])
    )
    assert (same.status_code, same.json()["constraint_id"]) == (422, "change_not_offered")
    assert _frame(back, "component.created")["replaces"] == changed["component_id"]


async def test_a_bar_chart_over_time_keeps_one_place(pg_engine: AsyncEngine) -> None:
    """A bar chart of one series draws its periods, and a second place would
    turn it into one bar per place at one period. It offers one place per
    role, and a change adding a place is refused."""
    app = _app(
        search_call("aid from France to Kenya"),
        fetch_call(_SPEC, stat_vars="sum"),
        make_charts_call(
            chart(
                frame_id_for(_REF),
                type_="bar_chart",
                title=_TITLE,
                select={"stat_var": "dcid:Aid", "destination": "dcid:country/KEN"},
            )
        ),
        text_part("Here is the card."),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [_REF], "summary": "fetched"},
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "view": {
                        "dimensions": [
                            {"name": "stat_var", "values": ["dcid:Aid"], "plural": False},
                            {"name": "destination", "values": ["country/KEN"], "plural": False},
                        ]
                    },
                },
                _written("cmp_01unit0001", _REF, _TITLE),
            ]
        ),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
    )
    app.state.studio.place_options = _Places()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        both = {**_UGANDA, "entities": ["country/UGA", "country/KEN"]}
        refused = await client.post(
            "/api/studio/turn", json=_change_body("cmp_01unit0001", _SPEC, both), headers=AUTH
        )

    places = _frame(first, "component.created")["card_changes"]["places"]
    assert {p["role"]: p["multiple"] for p in places} == {"destination": False, "origin": False}
    assert (refused.status_code, refused.json()["constraint_id"]) == (422, "change_not_offered")


# A retrieval scoped to Togo. The model named the place as spoken under its
# own role name, and Search served the spec's identity with Togo's ref under
# the block's own role.
_TOGO_REF = {"spec_id": "spec_" + "7" * 32, "content_version": "cv_" + "7" * 32}
_BASE_IDENTITY = _ENVELOPE["parts"][0]["spec"]["definition"]["identity"]
_TOGO_IDENTITY = {
    **_BASE_IDENTITY,
    "entities": {"destination": ["country/TGO"], "origin": ["country/FRA"]},
}


def _togo_app(
    pg_engine: AsyncEngine,
    identity: Mapping[str, Any],
    labels: Mapping[str, str] | None = None,
    title: str = "Aid to Togo",
) -> Any:
    return _app(
        search_call("aid from France to Togo"),
        fetch_call(_SPEC, stat_vars="sum", scope={"entities": {"recipient": ["Togo"]}}),
        make_charts_call(chart(frame_id_for(_TOGO_REF), type_=_TYPE, title="Aid to Togo")),
        text_part("Here is the card."),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [_TOGO_REF], "summary": "fetched"},
                _VIEW,
                _written(
                    "cmp_01unit0001",
                    _TOGO_REF,
                    title,
                    identity=identity,
                    entity_labels={"country/TGO": "Togo"} if labels is None else labels,
                ),
            ]
        ),
        tool_surface=_Search(),  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
    )


async def test_a_card_fetched_by_spoken_names_offers_the_blocks_roles_and_refs(
    pg_engine: AsyncEngine,
) -> None:
    """The card reads the places Search served under the block's roles: its
    places read posts a role the block binds, and a change of the other role
    fetches it again by Togo's ref."""
    app = _togo_app(pg_engine, _TOGO_IDENTITY)
    places = _Places()
    app.state.studio.place_options = places
    changer = _Changer()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Togo"))
        read = await client.get(
            "/api/studio/places", params=_places_query(role="destination"), headers=AUTH
        )
        app.state.studio.recorder._executor = changer
        germany = {"kind": "places", "role": "origin", "entities": ["country/DEU"]}
        await _stream(client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, germany))

    offered = _frame(first, "component.created")["card_changes"]
    assert offered["spec_id"] == _SPEC
    chosen = {p["role"]: [(o["id"], o["label"]) for o in p["chosen"]] for p in offered["places"]}
    assert chosen == {
        "destination": [("country/TGO", "Togo")],
        "origin": [("country/FRA", "France")],
    }
    assert (read.status_code, places.asked[0]) == (200, "destination")
    (fetched,) = changer.fetched()
    assert fetched["scope"]["entities"] == {
        "destination": ["dcid:country/TGO"],
        "origin": ["dcid:country/DEU"],
    }


async def test_a_card_fetched_with_facets_offers_no_change(pg_engine: AsyncEngine) -> None:
    """A served identity that differs from the block's in its facets is a
    source the scope cannot name, so the card offers no change of years or
    places."""
    faceted = {**_TOGO_IDENTITY, "facets": {"provenance": {"ref": "OECD_DAC2A"}}}
    app = _togo_app(pg_engine, faceted)
    app.state.studio.place_options = _Places()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Togo"))

    assert "card_changes" not in _frame(first, "component.created")


def test_a_changed_cards_unit_words_keep_their_base_year() -> None:
    """On a card served in constant 2023 US dollars, the year in the unit's
    words is no period. A change of years keeps them in the title, and keeps
    an axis label that names only the unit."""
    move = _move("x", WindowChange(kind="window", start_year=2010, end_year=2024))
    block = copy.deepcopy(move.block)
    block["definition"]["identity"]["facets"] = {
        "unit": {"request": {"currency": "USD", "prices": "constant", "base_year": 2023}}
    }
    move = replace(move, block=block)

    title = _changed_title("Aid to Kenya in constant 2023 US dollars, 2015 to 2023", move)
    _, rewrite = _redrawn_intent({"type": "line_chart", "title": "t"}, _REF, move)

    assert title == "Aid to Kenya in constant 2023 US dollars, 2010 to 2024"
    assert rewrite("y_label", "Constant 2023 USD") == "Constant 2023 USD"
    assert rewrite("x_label", "2015 to 2023") is None


@pytest.mark.parametrize("count", [1, 3, 10, 50])
def test_a_composed_title_keeps_its_period_whole_and_counts_the_places_it_cannot_list(
    count: int,
) -> None:
    """However many places a change names, the title lists whole labels as far
    as they fit, closes on "and N others", and ends on the whole period."""
    names = [f"Republic of Placename Number {n}" for n in range(count)]
    refs = [f"country/P{n:02d}" for n in range(count)]
    card = UnitCard(
        spec_id=_SPEC,
        args={},
        scope={"entities": {"donor": [f"dcid:{r}" for r in refs]}},
        labels=dict(zip(refs, names, strict=True)),
    )

    title = _composed_title(_ENVELOPE["parts"][0]["spec"], card, "2015 to 2023")

    assert len(title) <= 120
    assert title.endswith(", 2015 to 2023")
    listed = [name for name in names if re.search(rf"{name}(?!\d)", title)]
    rest = count - len(listed)
    if rest:
        assert title.endswith(f" and {rest} other{'s' if rest > 1 else ''}, 2015 to 2023")


async def test_a_card_cut_to_its_largest_members_offers_no_years_or_places(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its members are the largest at one period, so another year or other
    places would draw them under a ranking they do not hold. It offers
    neither, a change of either is refused as one it does not offer, and the
    stored card keeps the mark."""
    monkeypatch.setattr(
        "harness.studio_translate.cut_cards", lambda state: frozenset({"cmp_01unit0001"})
    )
    app = _changes_app(pg_engine)
    app.state.studio.place_options = _Places()
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        refused = [
            await client.post(
                "/api/studio/turn", json=_change_body("cmp_01unit0001", _SPEC, change), headers=AUTH
            )
            for change in (_YEARS, _UGANDA)
        ]

    assert "card_changes" not in _frame(first, "component.created")
    assert [(r.status_code, r.json()["constraint_id"]) for r in refused] == [
        (422, "change_not_offered")
    ] * 2
    stored = app.state.studio.store.current(PRINCIPAL, CONVERSATION).components["cmp_01unit0001"]
    assert stored["unit_card"]["cut"] is True


async def test_a_place_with_no_name_is_never_shown_by_its_ref(pg_engine: AsyncEngine) -> None:
    """A card whose retrieval named no label for Togo offers no change of
    Togo's role, and a title built after a change of its other role names
    no ref and leaves the places out."""
    app = _togo_app(pg_engine, _TOGO_IDENTITY, labels={}, title="Aid in 2023 to Togo")
    app.state.studio.place_options = _Places()
    changer = _Changer()
    changer.title = "Aid in 2023 to Togo"
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Togo"))
        app.state.studio.recorder._executor = changer
        germany = {"kind": "places", "role": "origin", "entities": ["country/DEU"]}
        await _stream(client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, germany))

    offered = _frame(first, "component.created")["card_changes"]
    assert [p["role"] for p in offered["places"]] == ["origin"]
    assert all(o["label"] != o["id"] for p in offered["places"] for o in p["chosen"])
    write = next(call for call in changer.calls if call["tool"] == "write_component")
    assert "country/" not in write["args"]["intent"]["title"]
    assert write["args"]["intent"]["title"].endswith(", 2015 to 2023")


async def test_a_places_change_while_the_offers_are_stopped_is_capability_unavailable(
    pg_engine: AsyncEngine,
) -> None:
    """Once a places read learnt apis serves no door, a change of places
    answers what the places read does, before reading the card's offer."""
    app = _changes_app(pg_engine)
    app.state.studio.place_options = ApisPlaceOptions(
        base_url="http://apis.test",
        client=_NoDoorApis(),  # ty: ignore[invalid-argument-type] -- the two reads the reader makes
    )
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        read = await client.get("/api/studio/places", params=_places_query(), headers=AUTH)
        changed = await client.post(
            "/api/studio/turn", json=_change_body("cmp_01unit0001", _SPEC, _UGANDA), headers=AUTH
        )

    assert [(r.status_code, r.json()["error"]) for r in (read, changed)] == [
        (503, "capability_unavailable"),
        (503, "capability_unavailable"),
    ]


@pytest.mark.parametrize(
    ("origin", "destination"),
    [
        (
            "International Bank for Reconstruction and Development",
            "Democratic Republic of the Congo",
        ),
        ("I" * 70, "D" * 70),
    ],
    ids=["two-long-labels", "longer-still"],
)
def test_long_places_give_way_before_the_subject(origin: str, destination: str) -> None:
    """The subject stays whole and the title opens on it, and the flow falls
    back to a count of its places."""
    block = copy.deepcopy(_ENVELOPE["parts"][0]["spec"])
    block["resolved"]["concept_label"] = "Official development assistance"
    card = UnitCard(
        spec_id=_SPEC,
        args={},
        scope={"entities": {"origin": ["dcid:org/IBRD"], "destination": ["dcid:country/COD"]}},
        labels={"org/IBRD": origin, "country/COD": destination},
    )

    title = _composed_title(block, card, "2010 to 2024")

    assert title == "Official development assistance, 2 places, 2010 to 2024"


def test_unit_words_keep_their_base_year_on_a_card_whose_unit_cannot_change() -> None:
    move = _move("x", WindowChange(kind="window", start_year=2010, end_year=2024))
    block = copy.deepcopy(move.block)
    block["resolved"].pop("unit_conversion", None)
    block["resolved"].pop("price_bases", None)
    block["definition"]["identity"]["facets"] = {
        "unit": {"request": {"currency": "USD", "prices": "constant", "base_year": 2023}}
    }
    move = replace(move, block=block)
    _, rewrite = _redrawn_intent({"type": "line_chart", "title": "t"}, _REF, move)

    assert _changed_title("Aid in constant 2023 prices, 2015 to 2023", move) == (
        "Aid in constant 2023 prices, 2010 to 2024"
    )
    assert rewrite("y_label", "USD, constant 2023 prices") == "USD, constant 2023 prices"


class _NarrowingSearch(_Search):
    """Search with no `years_with_data`, whose answer serves the window the
    posted identity requests, as a data API that has not shipped the member
    answers a narrowed card."""

    async def call_tool(
        self, *, tool: ToolId, arguments: Any, on_progress: ProgressCallback | None = None
    ) -> ToolCallResult:
        answered = await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        window = (arguments.get("identity") or {}).get("requested_window") or {}
        if window.get("kind") == "range":
            (part,) = answered.meta[SEARCH_ENVELOPE_META_KEY]["parts"]
            part["spec"]["resolved"]["download"]["window"] = {
                "start_year": window["start_year"],
                "end_year": window["end_year"],
            }
        return answered


async def test_a_change_never_shrinks_the_years_a_card_offers(pg_engine: AsyncEngine) -> None:
    """Narrowed to 2018 to 2020 on an answer that states no years with data,
    the card still offers 1978 to 2024, and a change back to them is drawn."""
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        tool_surface=_NarrowingSearch(),  # ty: ignore[invalid-argument-type] -- the one tool a change reads
        engine=pg_engine,
    )
    changer = _Changer()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        narrow = {"kind": "window", "start_year": 2018, "end_year": 2020}
        narrowed = _frame(
            await _stream(
                client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, narrow)
            ),
            "component.created",
        )
        wide = {"kind": "window", "start_year": 1978, "end_year": 2024}
        widened = await _stream(
            client,
            "/api/studio/turn",
            _change_body(narrowed["component_id"], narrowed["card_changes"]["spec_id"], wide),
        )

    window = narrowed["card_changes"]["window"]
    assert (window["first_year"], window["last_year"]) == (1978, 2024)
    assert _frame(widened, "component.created")["card_changes"]["window"]["start_year"] == 1978


# --- a card on a sum of its one retrieval ------------------------------------

_SUM_OPS = {"spec_id": "ops_" + "c" * 32, "content_version": "ocv_" + "3" * 32}
_SUM_PARAMETERS = {"aggregation": "sum", "group_by": ["time_period", "financingRecipient"]}
# The pair the store holds the first retrieval under, minted with extra
# columns, beside the pair the data API serves it under (`_REF`).
_STORED = {"spec_id": _SPEC, "content_version": "cv_" + "8" * 32}
# The pair the analysis runner records the retrieval it summed under, which
# is neither of the other two.
_RUNNER = {"spec_id": _SPEC, "content_version": "cv_" + "7" * 32}


def _sum_recipe(
    stored: Mapping[str, str],
    served: Mapping[str, str],
    parameters: Mapping[str, Any] = _SUM_PARAMETERS,
) -> dict[str, Any]:
    """The recipe dfo-apis' runner stores for the sum by year and recipient
    of the retrieval the store holds as `stored` and the data API serves as
    `served`, each node naming its frame (`common.recipe`)."""
    return {
        "schema_version": 1,
        "tree": {
            "frame": dict(_SUM_OPS),
            "op_spec": {
                "identity_version": "ops1",
                "verb": "aggregate",
                "frame_ref": dict(stored),
                "parameters": dict(parameters),
            },
            "inputs": {"frame_ref": {"frame": dict(stored), "root": 0}},
        },
        "roots": [
            {"schema_version": 1, "frame": dict(stored), "dataset": dict(served), "options": {}}
        ],
        "anchor": None,
        "left_out": None,
    }


def _stored_apart(written: dict[str, Any]) -> dict[str, Any]:
    """`written`, whose one retrieval the store holds as `_STORED`."""
    lineage = written[CARD_LINEAGE_KEY]
    (member,) = lineage.lineage
    stored = ProvenanceInput(_STORED["spec_id"], _STORED["content_version"], member.manifest)
    written[CARD_LINEAGE_KEY] = replace(lineage, lineage=(stored,))
    return written


def _on_sum(written: dict[str, Any], ops: Mapping[str, str]) -> dict[str, Any]:
    """`written`, a card on its retrieval's frame, drawn on the sum `ops` of it."""
    lineage = written[CARD_LINEAGE_KEY]
    written[CARD_LINEAGE_KEY] = replace(lineage, frame=(ops["spec_id"], ops["content_version"]))
    written["component"]["provenance"]["inputs"] = [dict(ops)]
    return written


class _SummingSearch(_Search):
    """Search's resolve, and analysis-mcp's `aggregate`, which answers a sum
    under a pair of its own, or refuses when `refuse`."""

    def __init__(self, *, refuse: bool = False) -> None:
        super().__init__()
        self.summed: list[dict[str, Any]] = []
        self.refuse = refuse
        # The family ref each sum's series carry, by the sum's spec_id: the
        # `dcid:<spec_id>` apis writes on a summed retrieval's series, for
        # the spec the retrieval is served under.
        self.families: dict[str, str] = {}

    async def call_tool(
        self, *, tool: ToolId, arguments: Any, on_progress: ProgressCallback | None = None
    ) -> ToolCallResult:
        if tool.name != "aggregate":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        self.summed.append(dict(arguments))
        if self.refuse:
            return ToolCallResult(
                is_error=True,
                structured_content={"code": "unit_heterogeneous", "message": "two units"},
            )
        digest = hashlib.sha1(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        self.families["ops_" + digest[:32]] = f"dcid:{arguments['frame_ref']['spec_id']}"
        return ToolCallResult(
            is_error=False,
            structured_content={
                "spec_id": "ops_" + digest[:32],
                "content_version": "ocv_" + "6" * 32,
                "rows": 9,
            },
        )


class _SummedChanger(_Changer):
    """`_Changer` for a card on a sum: each retrieval is kept in `provenance`
    as the executor keeps one, stored under a pair of its own beside the one
    it is served under, as a frame minted with extra columns is, and only a
    served pair can be described. Each redraw is on the sum the write names."""

    def __init__(
        self,
        provenance: ProvenanceStore,
        *,
        view: list[dict[str, Any]] | None = None,
        facets: Mapping[str, str] | None = None,
        views: Mapping[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()
        self.provenance = provenance
        self.conversation: str | None = None
        self.view = view or []
        self.facets = facets
        # The view of a frame by its spec_id, else `view`.
        self.views = dict(views or {})
        self.stored_only = {_STORED["content_version"]}

    async def __call__(self, *, tool: str, args: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if tool == "describe_frame":
            self.calls.append({"tool": tool, "args": args})
            if args["frame_ref"]["content_version"] in self.stored_only:
                return {"outcome": "error", "error": {"code": "not_found"}}
            dimensions = self.views.get(args["frame_ref"]["spec_id"], self.view)
            return {"outcome": "ok", "frame_refs": [], "view": {"dimensions": list(dimensions)}}
        answer = await super().__call__(tool=tool, args=args, **kw)
        if tool == "get_data":
            (ref,) = answer["frame_refs"]
            stored = {"spec_id": ref["spec_id"], "content_version": "cv_" + "9" * 32}
            self.stored_only.add(stored["content_version"])
            self.conversation = current_conversation_id.get()
            self.provenance.put_retrieval(
                conversation=self.conversation,
                manifest={
                    "definition": {"spec_id": ref["spec_id"]},
                    "content_version": ref["content_version"],
                    "frame_ref": stored,
                    "artifacts": [{"rendering": _SUMMED}],
                },
            )
            answer["store_refs"] = [stored]
        if tool == "write_component":
            answer = _on_sum(answer, args["intent"]["frame_ref"])
            if self.facets is not None:
                answer = _faceted(answer, self.facets)
        return answer


def _summed_app(
    pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    surface: _SummingSearch,
    facets: Mapping[str, str] | None = None,
    parameters: Mapping[str, Any] = _SUM_PARAMETERS,
) -> tuple[Any, ProvenanceStore]:
    """The first turn draws France's aid to Kenya on the sum by year and
    recipient of its retrieval, called with `parameters`, whose recipe the
    runner serves."""

    async def recipe_of(ref: Any) -> dict[str, Any] | None:
        if ref.spec_id != _SUM_OPS["spec_id"]:
            return None
        return _sum_recipe(_RUNNER, _REF, parameters)

    monkeypatch.setattr(card_unit, "derived_recipe", recipe_of)
    provenance = ProvenanceStore()
    results = _first_results()
    results[-1] = _on_sum(_stored_apart(results[-1]), _SUM_OPS)
    if facets is not None:
        results[-1] = _faceted(results[-1], facets)
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(results),
        tool_surface=surface,  # ty: ignore[invalid-argument-type] -- resolve and aggregate
        engine=pg_engine,
        provenance=provenance,
    )
    # The runner the server builds holds the executor's provenance store.
    app.state.studio.provenance = provenance
    return app, provenance


@pytest.mark.parametrize("change", [_YEARS, _UGANDA], ids=["years", "places"])
async def test_a_card_on_a_sum_changes_by_summing_its_new_retrieval_as_recorded(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    """The card draws the sum by year and recipient of its one retrieval. It
    offers years and places, and no unit. A change fetches the retrieval over
    the new scope, sums it with the parameters the recipe recorded, writes
    the sum's lineage from the new retrieval, and redraws the card on the sum
    in its slot, offering its changes again."""
    surface = _SummingSearch()
    app, provenance = _summed_app(pg_engine, monkeypatch, surface)
    app.state.studio.place_options = _Places()
    changer = _SummedChanger(provenance)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, change)
        )

    drawn = _frame(first, "component.created")
    assert "unit_change" not in drawn
    assert drawn["card_changes"]["window"]["first_year"] == 1978
    assert [p["role"] for p in drawn["card_changes"]["places"]] == ["destination", "origin"]
    (fetched,) = changer.fetched()
    assert fetched["stat_vars"] == "sum"
    (summed,) = surface.summed
    root = {k: v for k, v in summed.items() if k == "frame_ref"}
    assert {k: v for k, v in summed.items() if k != "frame_ref"} == _SUM_PARAMETERS
    write = next(call for call in changer.calls if call["tool"] == "write_component")
    sum_ref = write["args"]["intent"]["frame_ref"]
    assert sum_ref["spec_id"].startswith("ops_") and sum_ref != _SUM_OPS
    assert changer.conversation is not None
    lineage = provenance.get(conversation=changer.conversation, **sum_ref)
    assert lineage is not None
    assert [(m.spec_id, m.content_version) for m in lineage] == [
        (root["frame_ref"]["spec_id"], root["frame_ref"]["content_version"])
    ]
    created = _frame(frames, "component.created")
    assert created["replaces"] == "cmp_01unit0001"
    assert "window" in created["card_changes"]
    assert "unit_change" not in created


async def test_a_sum_the_analysis_service_refuses_leaves_the_card(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sum refused over the new data ends the change on the retrieval's
    step: nothing is written, and the card keeps its slot and offers."""
    surface = _SummingSearch(refuse=True)
    app, provenance = _summed_app(pg_engine, monkeypatch, surface)
    changer = _SummedChanger(provenance)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )
        read = await client.get(
            "/api/studio/state", params={"conversation_id": CONVERSATION, "since": 0}, headers=AUTH
        )

    assert len(surface.summed) == 1
    assert "write_component" not in [call["tool"] for call in changer.calls]
    failed = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert [f["step_id"] for f in failed] == ["u2"]
    assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")
    (card,) = read.json()["closed_since"][0]["cards"]
    assert card["component_id"] == "cmp_01unit0001"
    assert card["card_changes"] == _frame(first, "component.created")["card_changes"]


_SUMMED_CARD = card_unit.SummedFrame(
    frame=(_SUM_OPS["spec_id"], _SUM_OPS["content_version"]),
    root=(_STORED["spec_id"], _STORED["content_version"]),
    served=(_REF["spec_id"], _REF["content_version"]),
    parameters=dict(_SUM_PARAMETERS),
)


_RECORDED_SUM_TITLE = "Total gross ODA disbursements to Kenya for health, 2010 to 2020"
# A title in the unit Search recorded for UK ODA (`prices_decision_uk_oda.json`:
# "constant 2024 US dollars"), whose base year falls after the years it holds.
_CONSTANT_TITLE = "ODA from the United Kingdom in constant 2024 US dollars, 2010 to 2020"


def _in_constant_2024_dollars_through_2020(move: ChangeMove) -> ChangeMove:
    """`move` on a spec served in constant 2024 US dollars, holding data up to
    2020, so its unit's base year lies outside the years it offers."""
    block = copy.deepcopy(move.block)
    facets = block["definition"]["identity"].setdefault("facets", {})
    facets["unit"] = {
        "state": "request",
        "request": {"currency": "USD", "prices": "constant", "base_year": 2024},
    }
    window = move.offer.window
    assert window is not None
    offer = move.offer.model_copy(
        update={"window": window.model_copy(update={"last_year": 2020, "end_year": 2020})}
    )
    return replace(move, block=block, offer=offer)


@pytest.mark.parametrize(
    ("title", "change", "redrawn"),
    [
        (
            _RECORDED_SUM_TITLE,
            WindowChange(kind="window", start_year=2012, end_year=2018),
            "Total gross ODA disbursements to Kenya for health, 2012 to 2018",
        ),
        (
            "Total gross ODA disbursements to Kenya for health, 2012 to 2018",
            WindowChange(kind="window", start_year=2014, end_year=2020),
            "Total gross ODA disbursements to Kenya for health, 2014 to 2020",
        ),
        (
            "Total gross ODA disbursements to Kenya for health, 2014 to 2020",
            PlacesChange(kind="places", role="destination", entities=["country/UGA"]),
            "Total gross ODA disbursements to Uganda for health, 2014 to 2020",
        ),
        (
            _CONSTANT_TITLE,
            WindowChange(kind="window", start_year=2012, end_year=2018),
            "ODA from the United Kingdom in constant 2024 US dollars, 2012 to 2018",
        ),
    ],
    ids=["years", "years-again", "places", "constant-prices"],
)
def test_a_changed_sums_title_keeps_its_measure_as_a_plain_cards_does(
    title: str, change: Any, redrawn: str
) -> None:
    """A sum's title that counts no members keeps its subject and its measure
    as a plain card's does, with the new period after a change of years and
    the new place where the title named the old one. The input is the
    recorded sum of health aid to Kenya, changed to 2012 to 2018, then to
    2014 to 2020, then to Uganda."""
    assert _REDRAWS["titles"][0] == _RECORDED_SUM_TITLE
    move = _move(title, change)
    move = replace(move, card=replace(move.card, total=_SUMMED_CARD))
    if title == _CONSTANT_TITLE:
        move = _in_constant_2024_dollars_through_2020(move)
    stored = {"title": title, "type": "line_chart", "frame_ref": dict(_SUM_OPS)}

    intent, _ = _redrawn_intent(stored, dict(_SUM_OPS), move)

    assert intent["title"] == redrawn


@pytest.mark.parametrize(
    "change",
    [
        WindowChange(kind="window", start_year=2010, end_year=2024),
        PlacesChange(kind="places", role="destination", entities=["country/UGA"]),
    ],
    ids=["years", "places"],
)
@pytest.mark.parametrize(
    "title",
    [
        "Total ODA from 32 donors to Kenya, 2015 to 2023",
        "Aid to Kenya, 8 largest of 36, 2015 to 2023",
    ],
    ids=["donors", "largest"],
)
def test_a_changed_sums_title_and_labels_count_no_members(change: Any, title: str) -> None:
    """A sum's title may count the members it adds up, which a change can
    alter. Such a title is built from the measure the card's Search block
    heads its sentence with (here the recorded sum of health aid to Kenya's
    "Gross ODA disbursements"), places and period, and an axis label stating
    a count is dropped, so neither states a count the new sum may not have,
    and the measure is the block's whole heading."""
    block = {**_ENVELOPE["parts"][0]["spec"], "resolved": _REDRAWS["block_labels"]["resolved"]}
    (what,) = [
        part["text"] for part in block["resolved"]["sentence_plan"] if part.get("axis") == "what"
    ]
    move = replace(_move(title, change), block=block)
    move = replace(move, card=replace(move.card, total=_SUMMED_CARD))
    stored = {"title": title, "type": "line_chart", "frame_ref": dict(_SUM_OPS)}

    intent, rewrite = _redrawn_intent(stored, dict(_SUM_OPS), move)

    assert not {"32", "36", "8"} & set(re.findall(r"\d+", intent["title"]))
    assert intent["title"].startswith(what)
    assert rewrite("y_label", "Sum over 32 donors") is None
    assert rewrite("y_label", "USD, current prices") == "USD, current prices"


def _facet(axis: str, *values: str) -> dict[str, Any]:
    """A frame view's dimension of facet `axis`, holding `values`."""
    return {
        "name": axis,
        "values": list(values),
        "plural": len(values) > 1,
        "total_count": len(values),
    }


@pytest.mark.parametrize(
    "view",
    [
        [_facet("unit", "CurrencyUSD", "CurrencyEUR")],
        [_facet("unit", "CurrencyEUR")],
        [_facet("measurementMethod", "GrantEquivalent", "Flows")],
        [_facet("provenance", "provenance/OECDCRS")],
    ],
    ids=["two-units", "another-unit", "two-methods", "another-source"],
)
async def test_a_sum_whose_new_retrieval_is_counted_otherwise_is_refused_before_it_is_summed(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, view: list[dict[str, Any]]
) -> None:
    """The card sums series that share one source, method and unit, which
    its retrieval holds alone. A new retrieval holding two units or methods,
    or one other than the card's, would sum unlike figures or relabel them,
    so the change is refused once the new retrieval is read, before
    `aggregate` is called, and nothing is written."""
    surface = _SummingSearch()
    app, provenance = _summed_app(pg_engine, monkeypatch, surface, facets=_ELECTED)
    held = [_facet(axis, ref) for axis, ref in _ELECTED.items()]
    changer = _SummedChanger(provenance, view=view, views={_SPEC: held})
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )

    assert len(changer.fetched()) == 1
    described = [c["args"]["frame_ref"] for c in changer.calls if c["tool"] == "describe_frame"]
    assert described[-1]["spec_id"] != _SPEC
    assert surface.summed == []
    assert "write_component" not in [call["tool"] for call in changer.calls]
    failed = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert [f["step_id"] for f in failed] == ["u2"]
    assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")


async def test_a_sum_binds_the_facets_its_series_share_and_checks_them_again(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every series under the sum shares one source, method and unit, which
    the sum's own series name. The refetch binds all three, a new retrieval
    holding them alone is summed, and the redrawn sum is checked for them."""
    surface = _SummingSearch()
    app, provenance = _summed_app(pg_engine, monkeypatch, surface, facets=_ELECTED)
    view = [_facet(axis, ref) for axis, ref in _ELECTED.items()]
    changer = _SummedChanger(provenance, view=view, facets=_ELECTED)
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )

    (fetched,) = changer.fetched()
    assert fetched["scope"]["facets"] == {axis: {"ref": ref} for axis, ref in _ELECTED.items()}
    assert len(surface.summed) == 1
    assert _frame(frames, "component.created")["replaces"] == "cmp_01unit0001"


class _UnitBindingChanger(_SummedChanger):
    """`_SummedChanger` over a retrieval holding `units`, of which a new
    retrieval holds only the one its fetch binds, else all of them."""

    def __init__(self, provenance: ProvenanceStore, units: list[str]) -> None:
        super().__init__(provenance, views={_SPEC: [_facet("unit", *units)]})
        self.units = units

    async def __call__(self, *, tool: str, args: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if tool == "describe_frame" and self.fetched():
            bound = ((self.fetched()[-1].get("scope") or {}).get("facets") or {}).get("unit")
            self.view = [_facet("unit", *([bound["ref"]] if bound else self.units))]
        return await super().__call__(tool=tool, args=args, **kw)


async def test_a_sum_pinned_to_one_unit_binds_it_on_the_new_retrieval(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrieval holding two units is summed in the one the recipe pins,
    since `aggregate` sums the pinned unit's rows alone. A change binds the
    pinned unit on the refetch, so the new retrieval holds it alone and is
    summed with the pin, as recorded."""
    units = ["CurrencyEUR", "CurrencyUSD"]
    pinned = {**_SUM_PARAMETERS, "unit": f"dcid:{units[1]}"}
    surface = _SummingSearch()
    app, provenance = _summed_app(pg_engine, monkeypatch, surface, parameters=pinned)
    changer = _UnitBindingChanger(provenance, units)
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )

    (fetched,) = changer.fetched()
    assert fetched["scope"]["facets"]["unit"] == {"ref": units[1]}
    (summed,) = surface.summed
    assert summed["unit"] == pinned["unit"]
    assert _frame(frames, "component.created")["replaces"] == "cmp_01unit0001"


class _BindingChanger(_SummedChanger):
    """`_SummedChanger` whose platform stores and binds a sum's select as the
    data API does: the card's stored intent selects the family ref its sum's
    series carry (`dcid:<spec_id>` of the spec its retrieval is served
    under), and a write selecting a `stat_var` the drawn frame does not hold
    is refused `binding_invalid`."""

    def __init__(self, provenance: ProvenanceStore, surface: _SummingSearch) -> None:
        super().__init__(provenance)
        self.surface = surface
        self.card_frame = dict(_SUM_OPS)
        self.written: list[dict[str, Any]] = []

    def family(self, spec_id: str) -> str | None:
        return {_SUM_OPS["spec_id"]: f"dcid:{_SPEC}", **self.surface.families}.get(spec_id)

    async def __call__(self, *, tool: str, args: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if tool == "describe_component":
            self.calls.append({"tool": tool, "args": args})
            stored = _described(args["component_id"], self.card_frame, self.title)
            intent = stored["component_body"]["stored_intent"]
            intent["select"] = {"stat_var": self.family(self.card_frame["spec_id"])}
            return stored
        if tool == "write_component":
            intent = args["intent"]
            held = self.family(intent["frame_ref"]["spec_id"])
            named = intent["select"].get("stat_var")
            missing = [v for v in (named if isinstance(named, list) else [named]) if v != held]
            if missing:
                self.calls.append({"tool": tool, "args": args})
                return {
                    "outcome": "refusal",
                    "refusal": {
                        "code": "binding_invalid",
                        "retry_mode": "correct_and_retry",
                        "fix": None,
                        "message": f"'stat_var' does not hold value(s) {missing}",
                    },
                }
            self.written.append(intent)
            self.card_frame = dict(intent["frame_ref"])
        return await super().__call__(tool=tool, args=args, **kw)


@pytest.mark.parametrize(
    ("change", "then"), [(_YEARS, _UGANDA), (_UGANDA, _YEARS)], ids=["years", "places"]
)
async def test_a_changed_sum_selects_the_family_its_new_retrieval_is_summed_under(
    pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, Any],
    then: dict[str, Any],
) -> None:
    """A sum's series carry its family ref, `dcid:` and the spec its
    retrieval is served under, and the card's select names it. A change
    fetches a retrieval served under another spec, so the redraw selects
    that spec's ref, and the platform binds it. The redrawn card offers its
    changes again, keeps the `aggregate` step on its canvas entry, and a
    second change on it sums and redraws as the first did."""
    surface = _SummingSearch()
    app, provenance = _summed_app(pg_engine, monkeypatch, surface)
    app.state.studio.place_options = _Places()
    changer = _BindingChanger(provenance, surface)
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, change)
        )
        created = _frame(frames, "component.created")
        again = await _stream(
            client, "/api/studio/turn", _change_body(created["component_id"], _SPEC, then)
        )

    assert len(changer.fetched()) == 2
    assert [intent["select"] for intent in changer.written] == [
        {"stat_var": f"dcid:{summed['frame_ref']['spec_id']}"} for summed in surface.summed
    ]
    assert changer.written[0]["select"] != {"stat_var": f"dcid:{_SPEC}"}
    assert created["replaces"] == "cmp_01unit0001"
    entry = app.state.studio.store.current(PRINCIPAL, CONVERSATION).components[
        created["component_id"]
    ]
    assert entry["summed_with"] == {"tool": "aggregate", "arguments": _SUM_PARAMETERS}
    offer = created["card_changes"]
    assert "window" in offer
    if change is _UGANDA:
        (places,) = [p for p in offer["places"] if p["role"] == "destination"]
        assert [o["id"] for o in places["chosen"]] == change["entities"]
    redrawn = _frame(again, "component.created")
    assert redrawn["replaces"] == created["component_id"]


_REDRAWS = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded_changed_sum_redraws.json").read_text()
)


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("years", WindowChange(kind="window", start_year=2012, end_year=2018)),
        ("places", PlacesChange(kind="places", role="destination", entities=["country/UGA"])),
    ],
    ids=["years", "places"],
)
def test_a_recorded_sums_redraw_selects_only_what_its_new_frame_holds(
    name: str, change: Any
) -> None:
    """The recorded sum of health aid to Kenya, changed to 2012 to 2018 and to
    Uganda. Its stored select names the family ref of the spec its retrieval
    was served under, which each new sum frame lacks, and the platform
    refuses a select naming a value its frame lacks. Redrawn with the new
    retrieval's spec, every value the select names is one the new frame's
    view holds."""
    recorded = _REDRAWS["changes"][name]
    stored = _REDRAWS["stored_intent"]
    old = _REDRAWS["old_served"]
    held = {dim["name"]: dim["values"] for dim in recorded["view"]["dimensions"]}
    assert stored["select"]["stat_var"] == f"dcid:{old['spec_id']}"
    assert stored["select"]["stat_var"] not in held["stat_var"]
    manifest = recorded["derived_manifest"]
    move = _move(stored["title"], change)
    total = card_unit.SummedFrame(
        frame=(stored["frame_ref"]["spec_id"], stored["frame_ref"]["content_version"]),
        root=(old["spec_id"], old["content_version"]),
        served=(old["spec_id"], old["content_version"]),
        parameters=manifest["recipe"]["tree"]["op_spec"]["parameters"],
    )
    move = replace(move, card=replace(move.card, total=total))
    frame_ref = {"spec_id": manifest["spec_id"], "content_version": manifest["content_version"]}

    intent, _ = _redrawn_intent(
        stored, frame_ref, move, family=(old["spec_id"], recorded["served"]["spec_id"])
    )

    for column, value in intent["select"].items():
        assert set(value if isinstance(value, list) else [value]) <= set(held[column])
    assert intent["frame_ref"] == frame_ref


def _ending(frames: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    """What the page reads off a change's stream: each frame's event, with a
    step's id and status."""
    return [
        (f["event"], f["step_id"], f["status"]) if f["event"] == "activity.step" else (f["event"],)
        for f in frames
    ]


async def test_a_refused_sum_ends_as_a_plain_cards_refused_change(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The site gives a change's prompt back on what the harness answers: a
    refusal before the stream, or a stream ending with no card. A sum refused
    in the stream ends on the frames a plain card's empty retrieval ends on,
    and a change refused before the stream answers the plain card's 422, so
    the reader's allowance is given back alike."""
    plain_app = _changes_app(pg_engine)
    async with _client(plain_app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        plain_app.state.studio.recorder._executor = _Changer(rows=0)
        plain = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )
        plain_early = await client.post(
            "/api/studio/turn",
            json=_change_body("cmp_01unit0001", _SPEC, {**_YEARS, "start_year": 1900}),
            headers=AUTH,
        )

    surface = _SummingSearch(refuse=True)
    app, provenance = _summed_app(pg_engine, monkeypatch, surface)
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = _SummedChanger(provenance)
        summed = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )
        summed_early = await client.post(
            "/api/studio/turn",
            json=_change_body("cmp_01unit0001", _SPEC, {**_YEARS, "start_year": 1900}),
            headers=AUTH,
        )

    assert len(surface.summed) == 1
    assert _ending(summed) == _ending(plain)
    assert (summed_early.status_code, summed_early.json()) == (
        plain_early.status_code,
        plain_early.json(),
    )


# --- recorded cards: a change of places rewrites the role's own column -------

_RECORDED_CARDS = json.loads(
    (Path(__file__).parent / "fixtures" / "recorded_card_selects.json").read_text()
)


def _recorded_move(name: str, role: str, chosen: list[str], entities: list[str]) -> Any:
    """A change of the places `role` shows on the recorded card `name`, from
    `chosen` to `entities`, with the card's recorded type, title and select
    as its stored intent."""
    card = _RECORDED_CARDS[name]
    change = PlacesChange(kind="places", role=role, entities=entities)
    offer = CardChanges(
        spec_id=_SPEC,
        places=[
            PlacesOffer(
                role=role, multiple=True, chosen=[PlaceOption(id=ref, label=ref) for ref in chosen]
            )
        ],
    )
    move = ChangeMove(
        "cmp_01unit0001",
        card["title"],
        UnitCard(spec_id=_SPEC, args={}),
        change,
        {},
        "",
        offer,
        _ENVELOPE["parts"][0]["spec"],
        old=tuple(chosen),
        new=tuple(entities),
    )
    stored = {"type": card["type"], "title": card["title"], "select": card["written_select"]}
    return stored, move


def test_a_change_of_a_recorded_recipient_writes_the_select_the_live_redraw_wrote() -> None:
    """The recorded debt card for Kenya writes no recipient in its select,
    its places being the retrieval's. Changed to Uganda, it writes the select
    the recorded redraw for Uganda wrote."""
    stored, move = _recorded_move("debt_kenya", "destination", ["country/KEN"], ["country/UGA"])
    assert "financingRecipient" not in stored["select"]

    intent, _ = _redrawn_intent(stored, dict(_REF), move)

    assert intent["select"] == _RECORDED_CARDS["debt_uganda_after_change"]["written_select"]


@pytest.mark.parametrize(
    ("role", "chosen", "entities", "column", "written"),
    [
        ("destination", ["country/KEN"], ["country/UGA"], "financingRecipient", "country/UGA"),
        (
            "origin",
            None,
            ["country/CAN", "country/GBR"],
            "financingProvider",
            ["country/CAN", "country/GBR"],
        ),
    ],
    ids=["destination", "origin"],
)
def test_a_change_of_one_role_beside_another_listed_dimension_rewrites_its_column_alone(
    role: str, chosen: list[str] | None, entities: list[str], column: str, written: Any
) -> None:
    """Health aid to Kenya by provider, recorded, writes eight providers as a
    list beside its one recipient as one value. A change of either role
    rewrites that role's column in its own form, and the other entry stays
    as recorded."""
    recorded = _RECORDED_CARDS["health_aid_kenya_by_provider"]["written_select"]
    shown = chosen if chosen is not None else recorded["financingProvider"]
    stored, move = _recorded_move("health_aid_kenya_by_provider", role, shown, entities)

    intent, _ = _redrawn_intent(stored, dict(_REF), move)

    assert intent["select"][column] == written
    assert {k: v for k, v in intent["select"].items() if k != column} == {
        k: v for k, v in recorded.items() if k != column
    }


def test_a_change_of_places_leaves_another_role_naming_the_same_place() -> None:
    """Health aid to Kenya by provider, recorded, with Kenya also named as its
    provider: a change of the recipient rewrites `financingRecipient` and
    leaves `financingProvider` naming Kenya."""
    stored, move = _recorded_move(
        "health_aid_kenya_by_provider", "destination", ["country/KEN"], ["country/UGA"]
    )
    stored["select"] = {**stored["select"], "financingProvider": ["country/KEN"]}

    intent, _ = _redrawn_intent(stored, dict(_REF), move)

    assert intent["select"]["financingRecipient"] == "country/UGA"
    assert intent["select"]["financingProvider"] == ["country/KEN"]


_CRS = _RECORDED_CARDS["health_aid_kenya_summed"]["fact_summary"]["series"][0]["coordinates"][
    "provenance"
]


@pytest.mark.parametrize(("source", "summed"), [(_CRS, True), ("provenance/OECDDAC2A", False)])
async def test_a_sum_keeps_the_source_its_retrieval_was_counted_in(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, source: str, summed: bool
) -> None:
    """The sum's own series name no source, as a sum whose frame carries no
    facet column draws them. The source is read from the retrieval's frame
    (the recorded CRS source of the live sum of health aid to Kenya), bound
    on the refetch, and a new retrieval counted from another source is
    refused before it is summed. A changed sum's canvas entry names the
    `aggregate` step beside the `get_data` arguments."""
    surface = _SummingSearch()
    app, provenance = _summed_app(pg_engine, monkeypatch, surface)
    changer = _SummedChanger(
        provenance,
        views={_REF["spec_id"]: [_facet("provenance", _CRS)]},
        view=[_facet("provenance", source)],
    )
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))
        app.state.studio.recorder._executor = changer
        frames = await _stream(
            client, "/api/studio/turn", _change_body("cmp_01unit0001", _SPEC, _YEARS)
        )

    (fetched,) = changer.fetched()
    assert fetched["scope"]["facets"]["provenance"] == {"ref": _CRS}
    assert len(surface.summed) == (1 if summed else 0)
    if not summed:
        assert _frame(frames, "message.complete")["text"].startswith("The card is unchanged.")
        return
    created = _frame(frames, "component.created")
    entry = app.state.studio.store.current(PRINCIPAL, CONVERSATION).components[
        created["component_id"]
    ]
    assert entry["summed_with"] == {"tool": "aggregate", "arguments": _SUM_PARAMETERS}
