"""A card's unit change through `POST /api/studio/turn` with `apply_unit`.

The first turn searches France's aid to Kenya, a spec Search converts into US
dollars at current prices or at a base year from 2000 to 2023, fetches it and
draws one card on the retrieval's frame. A later `apply_unit` re-reads the spec
in the chosen unit through Search, fetches it with the card's frame arguments
and redraws the card in its slot, with no model call.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from common.mcp_telemetry import WRITE_TOKEN_KEY
from compiler.surface import ProgressCallback, ToolCallResult, ToolId
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.card_unit import UNIT_GONE_NOTE
from harness.graph.state import frame_id_for
from harness.provenance import CARD_LINEAGE_KEY, CardLineage, ProvenanceInput
from harness.studio_state import DatabaseUnavailable
from harness.studio_store import StudioStore
from harness.wire import CompositionManifest
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


def _component(component_id: str, ref: Mapping[str, str], title: str) -> dict[str, Any]:
    """The platform's card `component_id` on frame `ref`, titled `title`."""
    component = copy.deepcopy(COMPONENT)
    component["component_id"] = component_id
    component["payload"] = resolved_revision(
        "stat_card", component_id=component_id, revision=1, frame_ref=dict(ref), title=title
    )
    component["fact_summary"]["component_id"] = component_id
    component["provenance"]["inputs"] = [dict(ref)]
    return component


def _written(component_id: str, ref: Mapping[str, str], title: str) -> dict[str, Any]:
    """A `write_component` outcome for that card, with the lineage of the one
    `get_data` retrieval it draws, which summed the spec's StatVars."""
    manifest = {
        "definition": {"spec_id": ref["spec_id"]},
        "content_version": ref["content_version"],
        "artifacts": [{"rendering": _SUMMED}],
    }
    lineage = CardLineage(
        component_id=component_id,
        frame=(ref["spec_id"], ref["content_version"]),
        lineage=(ProvenanceInput(ref["spec_id"], ref["content_version"], manifest),),
    )
    return {
        "outcome": "ok",
        "frame_refs": [],
        "component": _component(component_id, ref, title),
        CARD_LINEAGE_KEY: lineage,
        WRITE_TOKEN_KEY: "cmw_" + "t" * 22,
    }


def _described(component_id: str, ref: Mapping[str, str], title: str) -> dict[str, Any]:
    return {
        "outcome": "ok",
        "frame_refs": [],
        "component_body": {
            "component_id": component_id,
            "revision": 1,
            "stored_intent": {
                "type": "stat_card",
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
        make_charts_call(chart(frame_id_for(_REF), type_="stat_card", title=_TITLE)),
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
    assert created["created_from"]["step_id"] == "u3"
    manifest = _frame(frames, "done")["composition"]
    assert [block["component_id"] for block in manifest["blocks"]] == ["cmp_01unit0002"]
    assert [f["event"] for f in frames[-2:]] == ["suggestions", "done"]
    assert "message.complete" not in [f["event"] for f in frames]

    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
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
            chart(frame_id_for(_REF), type_="stat_card", title=_TITLE),
            chart(frame_id_for(_REF), type_="stat_card", title=_SIBLING_TITLE),
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
