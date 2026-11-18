"""Transcript records, as the planner's tools read them (`planner.session.SessionCall`).

The planner's own records hold a model response's function calls
(`model_response`) or the answers of its calls (`tool_answers`), as the
planner host writes them. The turn loop's hold the user's message, a pause
and the user's answer to it (`harness.turn.transcript`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from google.genai import types
from harness.turn import transcript
from harness.turn.store import AgentRecord
from planner.session import answers_record, response_record


def tool_call(call_id: str | None, name: str, args: Mapping[str, Any]) -> types.FunctionCall:
    """One function call of a model response."""
    return types.FunctionCall(id=call_id, name=name, args=dict(args))


def model_response(*calls: types.FunctionCall) -> AgentRecord:
    """The planner record of one model response holding `calls`."""
    return response_record([types.Part(function_call=call) for call in calls])


def tool_answers(*answers: tuple[str, str, Mapping[str, Any]]) -> AgentRecord:
    """The planner record of the answers to one response's calls, each an
    `(id, name, response)`."""
    return answers_record(
        [
            types.FunctionResponse(id=call_id, name=name, response=dict(response))
            for call_id, name, response in answers
        ]
    )


def response_event(name: str, response: dict[str, Any]) -> AgentRecord:
    """The record of tool `name`'s answer `response`, the shape
    `planner.call.answered_since_referral` reads: a `propose_plan` answer
    whose `translation` carries a referral, or any other tool's."""
    return tool_answers((f"{name}-1", name, response))


def user_answer(interrupt_id: str, answer: Any) -> AgentRecord:
    """The user's `answer` to the pause `interrupt_id`, as the turn loop records it."""
    return transcript.answer(interrupt_id=interrupt_id, round=1, value=cast(Any, answer))


def turn_pause(kind: str, interrupt_id: str, payload: Mapping[str, Any]) -> AgentRecord:
    """A pause of `kind` the turn put to the user under `interrupt_id`."""
    return transcript.pause(
        kind=kind, interrupt_id=interrupt_id, round=1, payload=dict(payload), message=""
    )


def merged(*records: AgentRecord) -> AgentRecord:
    """One record holding the parts of `records`, all of one agent and kind,
    as the planner host writes the calls of one model response, or their
    answers, together."""
    (shape,) = {(record.agent, record.kind) for record in records}
    parts = [part for record in records for part in record.body.get("parts") or ()]
    return AgentRecord(agent=shape[0], kind=shape[1], body={**records[0].body, "parts": parts})
