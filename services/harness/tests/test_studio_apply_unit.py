"""A card's unit change through `POST /api/studio/turn` with `apply_unit`.

The first turn searches France's aid to Kenya, a spec Search converts into US
dollars at current prices or at a base year from 2000 to 2023, fetches it and
draws one card on the retrieval's frame. A later `apply_unit` re-reads the spec
in the chosen unit through Search, fetches it with the card's frame arguments
and redraws the card in its slot, with no model call.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import pytest
from common.mcp_telemetry import WRITE_TOKEN_KEY
from compiler.surface import ProgressCallback, ToolCallResult, ToolId
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from harness.graph.state import frame_id_for
from harness.provenance import CARD_LINEAGE_KEY, CardLineage, ProvenanceInput
from harness.studio_store import StudioStore
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
    app = _app(
        *_first_turn(),
        resolver=CannedSearch(_ENVELOPE),
        executor=ScriptedExecutor(_first_results()),
        engine=pg_engine,
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("France's aid to Kenya"))

    created = _frame(frames, "component.created")
    assert created["unit_change"]["spec_id"] == created["provenance"]["inputs"][0]["spec_id"]
    assert created["unit_change"]["offer"]["start"] == {"currency": "USD", "prices": "current"}
    assert [c["code"] for c in created["unit_change"]["offer"]["currencies"]] == ["USD"]


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

    # A state read gives the turn as its stream showed it: no reply, and the
    # change in words with no spec_id.
    shown = f"Change {_TITLE} to constant 2023 US dollars."
    (closed,) = read.json()["closed_since"]
    assert (closed["user_message"], closed["reply"]) == (shown, "")
    assert read.json()["turn"]["user_message"] == shown


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
