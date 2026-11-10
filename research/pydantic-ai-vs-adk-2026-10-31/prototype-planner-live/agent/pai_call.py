"""The `planner.call.PlannerCall` over a Pydantic AI planner's message record.

The tool bodies in `planner.tools` read their call through `PlannerCall`: the session
state, the id of the call being served, the `Step`s of what the user was asked and
answered and the calls of the response being served (`response_calls`). `PaiCall` produces those steps from messages, as `planner.agent._session_steps`
produces them from ADK events:

- the user's own message (`user_message`) is a `message` step;
- an accepted ask is a `call` step under its tool call id. An ask whose return is the
  tool's refusal never reached the user and yields no step;
- the user's answer (`answer_message`) is a return of the ask's tool under the ask's id
  carrying a `result`, and is the `answer` step of that ask, with the ask's `args`;
- any other tool's return is a `response` step with its name and body.

The ask is a plain tool, so its accepted return (`ASK_ACCEPTED`) and its refusal are
tool returns like any other; neither is an answer, because neither carries a `result`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from harness.turn_events import PLANNER_ASK
from planner.call import ResponseCall, Step
from planner.tool_results import Outcome
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Pydantic AI merges consecutive requests into one before sending them
# (`pydantic_ai.messages._merge_consecutive_messages`), which would send the user's
# message and the seam's brief as one content where ADK sends two. It leaves apart two
# requests whose `instructions` both are set and differ, and the agent's own instruction
# is the one the brief's request carries, so the user's message carries this note.
QUESTION_INSTRUCTIONS = "The user's message, sent to the orchestrator before the planner ran."

# The key `user_message` marks its request under, which is how `PaiCall` tells the
# user's own message from the brief the seam sends.
USER_MESSAGE = "dfo.user_message"


def user_message(text: str) -> ModelRequest:
    """The message the user typed into the session."""
    return ModelRequest(
        parts=[UserPromptPart(content=text)],
        instructions=QUESTION_INSTRUCTIONS,
        metadata={USER_MESSAGE: True},
    )


def answer_message(ask_id: str, answer: Mapping[str, Any]) -> ModelRequest:
    """The user's answer to the ask `ask_id`, as ADK's interrupt route records it: a
    response under the ask's id whose body is the answer, `{"result": ...}`."""
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=PLANNER_ASK, tool_call_id=ask_id, content=dict(answer))]
    )


@dataclass(frozen=True)
class PaiCall:
    """One tool call's view of its session: every message of the session in order, the
    state, and the id of the call being served (`None` outside one).

    `messages` is every earlier run's messages and the current run's own, which holds
    the response whose calls are in flight. The calls of that response run together, so
    while one of them is served its siblings have no return in `messages`; an ask among
    them is not yet put to the user, and a tool other than an ask reads the session as
    it stood before the response (`planner.agent._in_flight_calls`).
    """

    messages: Sequence[ModelMessage]
    state: dict[str, Any]
    call_id: str | None = None

    def steps(self) -> Sequence[Step]:
        return tuple(self._steps())

    def _steps(self) -> Iterator[Step]:
        refused = {
            part.tool_call_id
            for part in self._parts(ToolReturnPart)
            if part.tool_name == PLANNER_ASK and _refusal(part.content)
        }
        held = self._in_flight()
        asks: dict[str, dict[str, Any]] = {}
        for message in self.messages:
            if isinstance(message, ModelResponse):
                for call in (p for p in message.parts if isinstance(p, ToolCallPart)):
                    if (
                        call.tool_name == PLANNER_ASK
                        and call.tool_call_id not in refused
                        and call.tool_call_id not in held
                    ):
                        asks[call.tool_call_id] = call.args_as_dict()
                        yield Step("call", call.tool_call_id, asks[call.tool_call_id])
                continue
            if (message.metadata or {}).get(USER_MESSAGE):
                yield Step("message")
                continue
            for part in message.parts:
                if not isinstance(part, ToolReturnPart):
                    continue
                body = part.content if isinstance(part.content, dict) else {"result": part.content}
                if part.tool_name != PLANNER_ASK:
                    yield Step("response", body=body, name=part.tool_name)
                elif "result" in body:
                    yield Step("answer", part.tool_call_id, asks.get(part.tool_call_id), body)

    def response_calls(self) -> Sequence[ResponseCall]:
        """The calls of the model response holding the call being served, in order."""
        if self.call_id is None:
            return ()
        for message in reversed(self.messages):
            if not isinstance(message, ModelResponse):
                continue
            calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
            if any(call.tool_call_id == self.call_id for call in calls):
                return tuple(ResponseCall(call.tool_name, call.tool_call_id) for call in calls)
        return ()

    def _parts[P](self, kind: type[P]) -> Iterator[P]:
        for message in self.messages:
            if isinstance(message, ModelRequest):
                yield from (part for part in message.parts if isinstance(part, kind))

    def _in_flight(self) -> frozenset[str]:
        """The ids of the calls in the response holding the call being served, when that
        call is not an ask."""
        if self.call_id is None:
            return frozenset()
        for message in reversed(self.messages):
            if not isinstance(message, ModelResponse):
                continue
            calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
            serving = next((call for call in calls if call.tool_call_id == self.call_id), None)
            if serving is None:
                continue
            if serving.tool_name == PLANNER_ASK:
                return frozenset()
            return frozenset(call.tool_call_id for call in calls)
        return frozenset()


def _refusal(content: Any) -> bool:
    """Whether an ask's return is its tool's refusal: neither the user's answer (it
    carries a `result`) nor the accepted ask (`ASK_ACCEPTED`, an `ok` answer)."""
    return (
        isinstance(content, dict)
        and "result" not in content
        and content.get("outcome") != Outcome.OK.value
    )
