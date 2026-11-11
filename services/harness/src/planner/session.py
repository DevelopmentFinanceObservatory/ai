"""The planner's reader of an ADK session: `session_steps` and `SessionCall`.

The planner stores its run in the session as `harness.planner_events` builds
it, the events an ADK `LlmAgent` stores for the same exchange. `SessionCall`
is the `planner.call.PlannerCall` of a call over its session's events.

The events a call reads are those of the session as its tool runs. A model
response's event is appended before the response's calls run, and their
answers in one event once every call has answered: the planner's node
(`harness.planner_node`) yields each response's event before Pydantic AI runs
its calls, and ADK appends each event a node yields before the node goes on.
So a call sees the earlier events, its own response's call event, and no
answer of its own response's calls.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from google.adk import Context
from google.adk.events import Event
from harness.turn_events import PLANNER_ASK, REQUEST_INPUT_FUNCTION_CALL_NAME

from planner.call import ResponseCall, Step
from planner.plan import SessionState
from planner.tool_results import Outcome

# A session's events, as `session_steps` and `SessionCall` read them.
type SessionEvents = Sequence[Event]


@dataclass(frozen=True)
class SessionCall:
    """The `planner.call.PlannerCall` of one tool call over its session.

    `events` is the session's events, the live list the session appends to,
    `state` the state the call reads and writes, and `call_id` the id of the
    function call the tool serves, `None` outside one. The steps are the
    events read as `session_steps` reads them.
    """

    events: SessionEvents
    state: SessionState
    call_id: str | None

    @classmethod
    def of(cls, context: Context) -> SessionCall:
        """The call `context` serves: a tool's `ToolContext` or a graph node's
        `Context`, over its session's events and state."""
        return cls(context.session.events, context.state, context.function_call_id)

    @functools.cached_property
    def _steps(self) -> tuple[Step, ...]:
        return tuple(session_steps(self.events, serving=self.call_id))

    def steps(self) -> Sequence[Step]:
        """`session_steps` over the call's session events, read once per call.

        A `SessionCall` serves one tool call, and the session appends the
        answers of a model response's calls only after they all finish, so
        the events a call sees do not change while it runs and the cached
        view cannot outlive the response it was read in.
        """
        return self._steps

    def response_calls(self) -> Sequence[ResponseCall]:
        """The calls of the planner model's response that holds the call's own
        (`_response_holding`)."""
        if self.call_id is None:
            return ()
        calls = _response_holding(self.events, self.call_id)
        return tuple(ResponseCall(call.name, call.id) for call in calls)


def _function_calls(event: Any) -> list[Any]:
    """The function calls of `event`'s parts, in order."""
    if not (event.content and event.content.parts):
        return []
    return [part.function_call for part in event.content.parts if part.function_call]


def _response_holding(events: SessionEvents, call_id: str) -> list[Any]:
    """The function calls of the latest model response in `events` that holds a
    call with `call_id`, in order, or none.

    The graph's planner node pauses on an accepted ask with an interrupt call
    (`REQUEST_INPUT_FUNCTION_CALL_NAME`) under the ask's own id, so a later
    event can hold the id without being the response that made the call. A
    call of that name under `call_id` is skipped.
    """
    for event in reversed(events):
        calls = _function_calls(event)
        if any(
            call.id == call_id and call.name != REQUEST_INPUT_FUNCTION_CALL_NAME for call in calls
        ):
            return calls
    return []


def _in_flight_calls(events: SessionEvents, serving: str | None) -> frozenset[str]:
    """The ids of the calls in the model response that holds the call `serving`
    which no event answers yet.

    `serving` is a tool that reads steps, never an ask, which decides from its
    response's call list instead (`_ask_beside_ask`).

    The session appends a response's merged answers after all its calls
    finish, so while one call of the response runs, its siblings have no
    answer in the session, whether they have finished or not.
    """
    if serving is None:
        return frozenset()
    answered = {
        part.function_response.id
        for event in events
        for part in (event.content.parts if event.content and event.content.parts else ())
        if part.function_response is not None
    }
    calls = _response_holding(events, serving)
    if not any(call.id == serving and call.name != PLANNER_ASK for call in calls):
        return frozenset()
    return frozenset(call.id for call in calls if call.id and call.id not in answered)


def session_steps(events: SessionEvents, *, serving: str | None = None) -> Iterator[Step]:
    """The steps of a session's `events`, in event order (`planner.call.Step`).

    The only reader of ADK events in the planner. A message the user typed
    (`_user_message`) is one `message` step. The event's other parts are not
    read, and no producer in the harness sends a user-typed event that also
    carries a function part. Every other event yields a `call` step for each
    ask carrying an id, an `answer` step for each response to one, paired
    with the `args` of the nearest preceding call sharing its id, and a
    `response` step for each other tool's response.

    An ask is the planner's own (`PLANNER_ASK`) or one of the workflow's
    interrupts. The planner's call holds its question flat in `args`. A
    planner ask whose response is its tool's refusal, or ADK's rejection of
    the call, never reached the user: it yields no step at all, so every
    reader sees the session as if the ask had not been made. An accepted
    one's tool answers `ASK_ACCEPTED`, which says nothing the user did and is
    not read, and the graph's planner node pauses on it with a `RequestInput`
    under the call's own id, so the user's answer to that interrupt is the
    answer to the call, and the interrupt's own call repeats the question.
    A workflow interrupt is a `RequestInput` its node yields.
    `create_request_input_event` builds the call with `id=interrupt_id` and
    dumps the whole `RequestInput` into its `args`, so the node's own
    `payload` nests under `args["payload"]`, and
    `create_request_input_response`, which `harness.studio`'s `/interrupt`
    route sends, builds the answer with `FunctionResponse.id` set to that
    same `interrupt_id`.

    `serving` is the id of the tool call being served. The calls of its own
    model response that no event answers yet (`_in_flight_calls`) yield no
    step, so a call running beside an ask reads the session as it stood
    before that response, and the ask, which may still be refused or may be
    accepted, is not counted as put to the user.
    """
    refused = _refused_asks(events)
    in_flight = _in_flight_calls(events, serving)
    calls: dict[str, dict[str, Any]] = {}
    planner_asks: set[str] = set()
    for event in events:
        if not (event.content and event.content.parts):
            continue
        if _user_message(event):
            yield Step("message")
            continue
        for part in event.content.parts:
            call = part.function_call
            if call is not None and call.id is not None:
                if call.id in in_flight:
                    continue
                if call.name == PLANNER_ASK:
                    if call.id in refused:
                        continue
                    planner_asks.add(call.id)
                elif call.name != REQUEST_INPUT_FUNCTION_CALL_NAME or call.id in planner_asks:
                    continue
                calls[call.id] = call.args or {}
                yield Step("call", call.id, call.args or {})
                continue
            response = part.function_response
            if response is None:
                continue
            body = response.response or {}
            if response.name == PLANNER_ASK and "result" not in body:
                continue
            if response.name in (PLANNER_ASK, REQUEST_INPUT_FUNCTION_CALL_NAME):
                yield Step("answer", response.id, calls.get(response.id or ""), body)
            else:
                yield Step("response", body=body, name=response.name)


def _refused_asks(events: Iterable[Event]) -> frozenset[str]:
    """The ids of the planner asks that never reached the user: their response
    is neither `ASK_ACCEPTED` nor an answer (no `result`), so it is the tool's
    refusal or ADK's rejection of the call."""
    return frozenset(
        response.id
        for event in events
        for part in (event.content.parts if event.content and event.content.parts else ())
        if (response := part.function_response) is not None
        and response.id is not None
        and response.name == PLANNER_ASK
        and "result" not in (body := response.response or {})
        and body.get("outcome") != Outcome.OK.value
    )


def _user_message(event: Any) -> bool:
    """Whether `event` is a message the user typed into the session.

    A user-authored event with text and no `branch`. The planner seam sends
    its own messages (the brief, an amendment's or a refusal's ask) as
    user-role content under `harness.planner_adapter._PLANNER_BRANCH`, so they
    are not the user's.
    """
    return (
        getattr(event, "author", None) == "user"
        and not getattr(event, "branch", None)
        and any((getattr(part, "text", None) or "").strip() for part in event.content.parts)
    )
