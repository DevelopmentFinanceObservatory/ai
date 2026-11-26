"""Each model call attempt of a Studio turn writes one `studio.model_call` line
(`harness.studio_store.write_model_call_line`), from the process that made it,
naming the turn's trace id, number and history arm, so a turn's cost is the
sum of its lines by `trace_id`."""

from __future__ import annotations

import contextlib
import json
import logging
from asyncio import create_task
from contextlib import asynccontextmanager
from typing import Any

import pytest
from common.logging import JSON_FIELDS
from google.genai import types
from harness.conversation import current_model_usage, shared_work_context
from harness.generative import GeminiEdit
from orchestrator.model import orchestrator_model
from planner.model import planner_model
from planner.plan import Plan
from planner.specs import PROPOSE_PLAN
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_genai import (
    Reply,
    ScriptedGenai,
    call_part,
    malformed_call_reply,
    reply,
    text_part,
    usage_metadata,
)
from .scripted_planner import scripted_planner
from .scripted_tools import CannedSearch, definite_envelope, plan_call
from .studio_routes import CLIENT_FEATURES, orchestrator_app, stream, studio_client, turn_body
from .test_generative import _FakeClient
from .test_planner_adapter import _translated_plan
from .test_planner_model import _After, _budget, _HangingGenai, _pai
from .test_planner_node import _QUESTION, _SIGNED, _proposals
from .test_studio import (
    COMPONENT,
    PRINCIPAL,
    WRITE_TOKEN,
    ScriptedExecutor,
    _card,
    _card_results,
    _described,
    _edit_body,
    _search,
)
from .test_studio_state import _connection_reset

pytestmark = pytest.mark.usefixtures("hydrogen_studio")

_CONVERSATION = "conv_01calls0000"
_EDIT_MODEL = "gemini-edit-test"


@pytest.fixture
def plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return _translated_plan(decision, requirement, derived_requirement)


def _lines(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, Any]]:
    """The JSON fields of every `event` line, in order."""
    return [
        fields
        for record in caplog.records
        if isinstance(fields := getattr(record, JSON_FIELDS, None), dict)
        and fields.get("event") == event
    ]


def _tokens(usages: list[types.GenerateContentResponseUsageMetadata]) -> tuple[int, ...]:
    """The input, cached, output and thinking tokens `usages` sum to."""
    return (
        sum(u.prompt_token_count or 0 for u in usages),
        sum(u.cached_content_token_count or 0 for u in usages),
        sum(u.candidates_token_count or 0 for u in usages),
        sum(u.thoughts_token_count or 0 for u in usages),
    )


def _summed(lines: list[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(
        sum(line[key] for line in lines)
        for key in ("input_tokens", "cached_tokens", "output_tokens", "thinking_tokens")
    )


def _usage(n: int) -> types.GenerateContentResponseUsageMetadata:
    """Usage no other call of a test reports, so a line counted twice or
    missed moves every sum."""
    return usage_metadata(prompt=1_000 * n, cached=100 * n, candidates=10 * n, thoughts=n)


async def test_every_model_call_attempt_of_a_turn_writes_one_line(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture, plan: Plan
) -> None:
    """Three turns of one conversation: a card turn whose first orchestrator
    call Gemini ends on a function call it could not parse and the harness
    sends again, an Edit Card turn, and a planned turn whose first planner
    call ends the same way. Each attempt writes one line naming its turn,
    role and arm, and each turn's lines sum to the usage its calls reported."""
    card = [malformed_call_reply(usage=_usage(1))] + [
        Reply(chunks=[[part]], usage=_usage(n)) for n, part in enumerate(_card(), start=2)
    ]
    edit_usage = _usage(6)
    planner = ScriptedGenai(
        [
            malformed_call_reply(usage=_usage(8)),
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED), usage=_usage(9)),
        ]
    )
    revised = {
        **COMPONENT,
        "revision": 2,
        "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
    }
    patch = {"title": "Aid to Kenya, 2018-2024"}
    app, _ = orchestrator_app(
        *card,
        Reply(chunks=[[plan_call(_QUESTION)]], usage=_usage(7)),
        resolver=_search(),
        executor=ScriptedExecutor(
            [
                *_card_results(write_token=WRITE_TOKEN),
                _described(),
                {"outcome": "ok", "frame_refs": [], "component": revised},
            ]
        ),
        planner=scripted_planner(planner, tools=_proposals(records=plan)),
        edit_model=GeminiEdit(
            model_name=_EDIT_MODEL,
            timeout_s=30,
            client=_FakeClient(
                json.dumps(
                    {"text": "Renamed.", "patch_json": json.dumps(patch), "recreate_json": None}
                ),
                edit_usage,
            ),
        ),
        engine=pg_engine,
    )

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        async with studio_client(app) as client:
            drawn = await stream(client, "/api/studio/turn", _body("Aid to Kenya"))
            assert drawn[-1]["event"] == "done"
            edited = await stream(client, "/api/studio/turn", _edit(_edit_body("rename it")))
            assert edited[-1]["event"] == "done"
            planned = await stream(client, "/api/studio/turn", _body(_QUESTION))
            assert planned[-1]["event"] == "plan.proposed"

    turns = {line["trace_id"]: line for line in _lines(caplog, "studio.turn")}
    calls = _lines(caplog, "studio.model_call")
    assert {call["trace_id"] for call in calls} == set(turns)
    for call in calls:
        turn = turns[call["trace_id"]]
        assert (call["conversation_id"], call["turn"]) == (_CONVERSATION, turn["turn"])
        assert (call["history"], call["history_mode"]) == (turn["history"], turn["history_mode"])
    by_turn = {number: [call for call in calls if call["turn"] == number] for number in (1, 2, 3)}
    assert [[call["role"] for call in by_turn[n]] for n in (1, 2, 3)] == [
        ["orchestrator"] * 5,
        ["edit"],
        ["orchestrator", "planner", "planner"],
    ]
    assert by_turn[2][0]["model"] == _EDIT_MODEL
    assert _summed(by_turn[1]) == _tokens([_usage(n) for n in range(1, 6)])
    assert _summed(by_turn[2]) == _tokens([edit_usage])
    assert _summed(by_turn[3]) == _tokens([_usage(n) for n in (7, 8, 9)])


def _body(message: str) -> dict[str, Any]:
    return turn_body(message, conversation_id=_CONVERSATION)


def _edit(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "conversation_id": _CONVERSATION}


def _answer(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """The `/interrupt` answer to the stream's question, picking its first option."""
    (elicitation,) = [f for f in frames if f["event"] == "elicitation.request"]
    (question,) = elicitation["questions"]
    return {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": _CONVERSATION,
        "interrupt_id": elicitation["interrupt_id"],
        "decision": "answer",
        "answers": {question["id"]: {"kind": "option", "option_id": question["options"][0]["id"]}},
        "client_features": CLIENT_FEATURES,
    }


@pytest.mark.parametrize("lost_ack", [False, True], ids=["resumed", "save-ack-lost-then-load"])
async def test_a_turn_resumed_on_another_instance_sums_both_segments_once(
    pg_engine: AsyncEngine, caplog: pytest.LogCaptureFixture, lost_ack: bool
) -> None:
    """A turn pauses on one instance and resumes on another. The lines of
    both segments carry the turn's one trace id and sum to the usage both
    instances' calls reported. A save on the first instance whose COMMIT
    lands and then raises, and the load after it, write no line."""
    asking = call_part(
        "ask_user",
        {"questions": [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]},
    )
    first, _ = orchestrator_app(Reply(chunks=[[asking]], usage=_usage(1)), engine=pg_engine)
    second, _ = orchestrator_app(
        reply(text_part("A it is."), usage=_usage(2)),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )

    with caplog.at_level(logging.INFO, logger="harness.studio_store"):
        async with studio_client(first) as client:
            paused = await stream(client, "/api/studio/turn", _body("Compare them"))
        assert paused[-1]["event"] == "elicitation.request"
        if lost_ack:
            await _save_losing_the_ack(first.state.studio.store)
            assert await first.state.studio.store.load(PRINCIPAL, _CONVERSATION) is not None
        # A request of its own starts with no sink, as another process's does.
        current_model_usage.set(None)
        async with studio_client(second) as client:
            frames = await stream(client, "/api/studio/interrupt", _answer(paused))
        assert frames[-1]["event"] == "done"

    (closed,) = [line for line in _lines(caplog, "studio.turn") if line["status"] == "closed"]
    calls = _lines(caplog, "studio.model_call")
    assert [(call["trace_id"], call["turn"]) for call in calls] == [(closed["trace_id"], 1)] * 2
    assert _summed(calls) == _tokens([_usage(1), _usage(2)])


async def _save_losing_the_ack(store: Any) -> None:
    """Save a change to the paused turn's record once, with its COMMIT
    landing and the acknowledgement lost, as a connection dropped after the
    commit does. The save carries the turn's `turn_usage` and no turn line."""
    store.current(PRINCIPAL, _CONVERSATION).note_spec_use("fetched", "spec_noted")
    real_transaction = store._transaction
    faults: list[int] = []

    @asynccontextmanager
    async def losing_the_ack() -> Any:
        async with real_transaction() as conn:
            yield conn
        if not faults:
            faults.append(1)
            raise _connection_reset("COMMIT")

    store._transaction = losing_the_ack
    try:
        with contextlib.suppress(DBAPIError):
            await store.save(PRINCIPAL, _CONVERSATION)
    finally:
        store._transaction = real_transaction
    assert faults == [1]


async def test_a_model_call_in_shared_work_writes_no_line() -> None:
    """Work several requests join (`shared_work_context`) reports its model
    calls to no turn, even from inside a turn whose sink is set."""
    heard: list[Any] = []
    current_model_usage.set(heard.append)
    edit = GeminiEdit(
        model_name=_EDIT_MODEL,
        timeout_s=30,
        client=_FakeClient(
            json.dumps({"text": "Done.", "patch_json": None, "recreate_json": None}), _usage(1)
        ),
    )

    async def call() -> None:
        await edit(message="rename it", stored_intent={}, frame_view={})

    await create_task(call(), context=shared_work_context())
    await call()

    assert [entry.role for entry in heard] == ["edit"]


@pytest.mark.parametrize("role", ["planner", "orchestrator"])
@pytest.mark.parametrize(
    ("bound", "expected"),
    [
        # The stream stops after a chunk that reported usage, and the
        # model-time bound closes the call.
        ("model-time-cut", [1]),
        # The first send stalls before any chunk, so it reported no usage, and
        # the send that replaces it reports its own.
        ("first-chunk-resend", [2]),
    ],
)
async def test_a_call_a_bound_closes_or_resends_writes_the_usage_it_reported(
    role: str, bound: str, expected: list[int]
) -> None:
    """A call a bound closes mid-stream reports the usage its chunks carried,
    the usage the planner's `token usage` line logs, and a first-chunk resend
    reports the usage of the send that answered, once."""
    if bound == "model-time-cut":
        scripted: ScriptedGenai = _HangingGenai(
            [Reply(chunks=[[text_part("Partly")]], usage=_usage(1), finish_reason=None)],
            hang_after=1,
        )
        budget = _budget(model_time_s=0.05)
    else:
        scripted = ScriptedGenai(
            [
                Reply(chunks=[[text_part("stalled")]], usage=_usage(1), stall_s=5.0),
                reply(text_part("Accra"), usage=_usage(2)),
            ]
        )
        budget = _budget(first_chunk_s=0.05)
    after = _After()
    model = (
        planner_model("gemini-test", client=scripted.client, budget=budget, after_response=after)
        if role == "planner"
        else orchestrator_model("gemini-test", client=scripted.client, budget=budget)
    )
    heard: list[Any] = []
    current_model_usage.set(heard.append)

    await _pai(model)

    assert [(call.role, call.usage) for call in heard] == [(role, _usage(n)) for n in expected]
    if role == "planner":
        assert [usage for _, usage in after.calls if usage is not None] == [
            call.usage for call in heard
        ]
