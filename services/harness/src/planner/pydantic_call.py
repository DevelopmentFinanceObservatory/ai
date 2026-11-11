"""The `planner.call.PlannerCall` a tool call of the Pydantic AI planner is.

A `PydanticCall` reads the steps of its session as `planner.agent.AdkCall`
reads them for the ADK planner: the steps of the session's earlier runs,
handed in as `planner.agent.session_steps` reads the session's events, then
the steps of the live run, read off the run's messages by `run_steps` as
`session_steps` reads the same exchange from ADK events.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from harness.turn_events import PLANNER_ASK
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from planner.call import RecordingState, ResponseCall, Step
from planner.tool_results import Outcome


@dataclass(frozen=True)
class PydanticCall:
    """The `planner.call.PlannerCall` of one tool call in a Pydantic AI planner run.

    `state` is the view over the session's state the call writes through.
    `call_id` is the id of the tool call being served, `None` outside one.
    `earlier` is the steps of the session before the live run, and `messages`
    the live run's messages as they stood when the call began: the calls of
    the response being served are in them, and none of their returns is,
    since Pydantic AI adds a response's returns once all its calls have
    finished, as ADK appends their responses.
    """

    state: RecordingState
    call_id: str | None
    earlier: Sequence[Step]
    messages: Sequence[ModelMessage]

    @functools.cached_property
    def _steps(self) -> tuple[Step, ...]:
        return (*self.earlier, *run_steps(self.messages, serving=self.call_id))

    def steps(self) -> Sequence[Step]:
        """The earlier steps, then the live run's (`run_steps`), read once per call."""
        return self._steps

    def response_calls(self) -> Sequence[ResponseCall]:
        """The calls of the latest response in the run that holds the call's own, in order."""
        calls = _calls_beside(self.messages, self.call_id)
        return tuple(ResponseCall(call.tool_name, call.tool_call_id) for call in calls)


def run_steps(messages: Sequence[ModelMessage], *, serving: str | None = None) -> Iterator[Step]:
    """The steps of a live Pydantic AI planner run's `messages`, in order.

    They are what `planner.agent.session_steps` reads from the ADK planner's
    events for the same exchange. Each `PLANNER_ASK` call is a `call` step,
    unless its return is the tool's refusal (`_never_reached`), and each
    return of any other tool is a `response` step. An ask's own return is
    `ASK_ACCEPTED` or its refusal, neither of which is the user's answer; the
    answer arrives in a later run's brief, so a live run holds no `answer`
    step. The user's message and the brief are the seam's input, not the
    user typing into the session, and give no step: the user's message is a
    step of the session the earlier steps are read from.

    `serving` is the id of the tool call being served. When it is no ask, the
    calls of its own response that no return answers yet give no step, as
    `session_steps` holds them back.
    """
    returns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    refused = {
        part.tool_call_id
        for part in returns
        if part.tool_name == PLANNER_ASK and _never_reached(_body(part))
    }
    in_flight = _in_flight(messages, serving, answered={part.tool_call_id for part in returns})
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if (
                    isinstance(part, ToolCallPart)
                    and part.tool_name == PLANNER_ASK
                    and part.tool_call_id not in refused
                    and part.tool_call_id not in in_flight
                ):
                    yield Step("call", part.tool_call_id, part.args_as_dict())
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name != PLANNER_ASK:
                yield Step("response", body=_body(part), name=part.tool_name)


def _body(part: ToolReturnPart) -> dict[str, Any]:
    return as_function_response(part.content)


def as_function_response(value: object) -> dict[str, Any]:
    """`value` as a function response carries it: a dict as it is, anything else
    under `result`, as ADK wraps a tool's answer that is no dict."""
    if isinstance(value, dict):
        # A function response's keys are the JSON object's, so strings.
        return cast("dict[str, Any]", value)
    return {"result": value}


def _never_reached(body: Mapping[str, Any]) -> bool:
    """Whether an ask's return is its tool's refusal: neither an answer (no
    `result`) nor the accepted ask (`ASK_ACCEPTED`, an `ok` answer)."""
    return "result" not in body and body.get("outcome") != Outcome.OK.value


def _calls_beside(messages: Sequence[ModelMessage], call_id: str | None) -> list[ToolCallPart]:
    """The calls of the latest response in `messages` holding the call `call_id`, or none."""
    if call_id is None:
        return []
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
        if any(call.tool_call_id == call_id for call in calls):
            return calls
    return []


def _in_flight(
    messages: Sequence[ModelMessage], serving: str | None, *, answered: set[str]
) -> frozenset[str]:
    """The ids of the calls of the response holding `serving` that no return answers,
    when `serving` is no ask (`planner.agent`'s `_in_flight_calls`)."""
    calls = _calls_beside(messages, serving)
    if any(call.tool_call_id == serving and call.tool_name == PLANNER_ASK for call in calls):
        return frozenset()
    return frozenset(call.tool_call_id for call in calls if call.tool_call_id not in answered)
