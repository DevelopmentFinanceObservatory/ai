"""A Pydantic AI planner call reads its steps and its response's calls as an ADK planner call does.

Each case is one conversation, written once and rendered twice: as the ADK
planner's session events, read by `AdkCall`, and as the session events
before the live run plus the live run's Pydantic AI messages, read by
`PydanticCall`.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.adk.events import Event
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from harness.turn_events import PLANNER_ASK
from planner.agent import AdkCall, session_steps
from planner.call import RecordingState
from planner.pydantic_call import PydanticCall
from planner.tools import ASK_ACCEPTED
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

_PLANNER = "dfo_planner"
_REFUSED = {"outcome": "refused", "code": "plan_invalid", "message": "no"}
_ASK = {"message": "Which source?", "candidates": [{"spec_id": "s1", "label": "One"}]}


def _response(*calls: tuple[str, dict[str, Any], str | None]) -> tuple[str, Any]:
    return ("response", calls)


def _returns(*answers: tuple[str, str, dict[str, Any]]) -> tuple[str, Any]:
    return ("returns", answers)


# Where the live run starts: what precedes it is the session's earlier events.
_LIVE = ("live", None)

# The question the turn opens with; the session's first event.
_QUESTION = ("question", "aid to Ghana")


def _adk_events(conversation: Sequence[tuple[str, Any]]) -> list[Event]:
    events: list[Event] = []
    for kind, value in conversation:
        if kind == "question":
            content = types.Content(role="user", parts=[types.Part(text=value)])
            events.append(Event(author="user", content=content))
        elif kind == "response":
            parts = [
                types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))
                for name, args, call_id in value
            ]
            content = types.Content(role="model", parts=parts)
            events.append(Event(author=_PLANNER, branch=_PLANNER, content=content))
        elif kind == "returns":
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(id=call_id, name=name, response=body)
                )
                for name, call_id, body in value
            ]
            content = types.Content(role="user", parts=parts)
            events.append(Event(author=_PLANNER, branch=_PLANNER, content=content))
    return events


def _live_messages(conversation: Sequence[tuple[str, Any]]) -> list[ModelMessage]:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("aid to Ghana")], instructions="note"),
        ModelRequest(parts=[UserPromptPart("the brief")], instructions="planner"),
    ]
    for kind, value in conversation:
        if kind == "response":
            messages.append(
                ModelResponse(
                    parts=[
                        ToolCallPart(name, args, tool_call_id=str(call_id))
                        for name, args, call_id in value
                    ]
                )
            )
        elif kind == "returns":
            messages.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(name, body, tool_call_id=call_id)
                        for name, call_id, body in value
                    ]
                )
            )
    return messages


def _calls(
    conversation: Sequence[tuple[str, Any]], serving: str | None
) -> tuple[AdkCall, PydanticCall]:
    live = conversation.index(_LIVE)
    state: dict[str, Any] = {}
    tool_context = SimpleNamespace(
        state=state,
        function_call_id=serving,
        session=SimpleNamespace(events=_adk_events(conversation)),
    )
    adk = AdkCall(cast(ToolContext, tool_context))
    pydantic = PydanticCall(
        state=RecordingState(state),
        call_id=serving,
        earlier=tuple(session_steps(_adk_events(conversation[:live]))),
        messages=_live_messages(conversation[live:]),
    )
    return adk, pydantic


_CASES = {
    "a first call": (
        [_QUESTION, _LIVE, _response(("propose_plan", {"question": "q"}, "p1"))],
        "p1",
    ),
    "a call after a refused proposal": (
        [
            _QUESTION,
            _LIVE,
            _response(("propose_plan", {"question": "q"}, "p1")),
            _returns(("propose_plan", "p1", _REFUSED)),
            _response(("propose_plan", {"question": "q2"}, "p2")),
        ],
        "p2",
    ),
    "a call after a refused ask": (
        [
            _QUESTION,
            _LIVE,
            _response((PLANNER_ASK, _ASK, "a1")),
            _returns((PLANNER_ASK, "a1", _REFUSED)),
            _response(("propose_plan", {}, "p1")),
        ],
        "p1",
    ),
    "the second ask in one response": (
        [_QUESTION, _LIVE, _response((PLANNER_ASK, _ASK, "a1"), (PLANNER_ASK, _ASK, "a2"))],
        "a2",
    ),
    "a proposal beside an ask in its own response": (
        [_QUESTION, _LIVE, _response((PLANNER_ASK, _ASK, "a1"), ("propose_plan", {}, "p1"))],
        "p1",
    ),
    "an ask after an accepted ask nobody answered": (
        [
            _QUESTION,
            _response((PLANNER_ASK, _ASK, "a1")),
            _returns((PLANNER_ASK, "a1", dict(ASK_ACCEPTED))),
            _LIVE,
            _response((PLANNER_ASK, _ASK, "a2")),
        ],
        "a2",
    ),
    "a read made while no call is served": (
        [_QUESTION, _LIVE, _response((PLANNER_ASK, _ASK, "a1"), ("propose_plan", {}, "p1"))],
        None,
    ),
}


@pytest.mark.parametrize(("conversation", "serving"), _CASES.values(), ids=_CASES.keys())
def test_a_pydantic_ai_call_reads_its_steps_and_response_calls_as_an_adk_call_does(
    conversation: list[tuple[str, Any]], serving: str | None
) -> None:
    adk, pydantic = _calls(conversation, serving)

    assert tuple(pydantic.steps()) == tuple(adk.steps())
    assert tuple(pydantic.response_calls()) == tuple(adk.response_calls())
