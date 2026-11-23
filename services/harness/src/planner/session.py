"""The planner's reader of a turn's transcript: `session_steps` and `SessionCall`.

A turn's transcript (`harness.turn.transcript`) holds, in order, the user's
message, each pause the turn put to the user and each answer, which the
turn loop writes, and the planner's own run records, which the planner host
(`harness.planner_adapter`) writes as a run goes:

- `response_record` for each model response with parts, holding its parts
  as the model role's content, each function call with its id, written as
  soon as the response arrives and before its calls run;
- `answers_record` for the answers of one response's calls, written once
  every call of the response has answered, holding every answer in the
  order the calls were made.

`SessionCall` is the `planner.call.PlannerCall` of a call over the
transcript as it stands when the call runs, so a call sees the earlier
records, its own response's record, and no answer of its own response's
calls.

A record's body is JSON, as the transcript stores it. A planner record's is a
google-genai `Content` dumped with `model_dump(mode="json",
exclude_none=True)`, read here as the dict it is.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from google.genai import types
from harness.turn import transcript
from harness.turn.store import AgentRecord
from harness.turn_events import PLANNER_ASK

from planner.call import ResponseCall, Step
from planner.plan import SessionState
from planner.tool_results import Outcome

# The kinds of the planner's own run records.
RESPONSE = "response"
ANSWERS = "answers"

# The kind of the pause on the planner's own ask (`harness.graph.planning`).
PLANNER_ASK_PAUSE = "planner_ask"
# The key of a planner-side pause's payload that names the id of the planner
# ask it put to the user (`harness.graph.planning.planner_ask_pause`).
ASKED_CALL_ID = "call_id"

# A turn's transcript, as `session_steps` and `SessionCall` read it.
type Records = Sequence[AgentRecord]


def response_record(parts: Sequence[types.Part]) -> AgentRecord:
    """The record of a model response holding `parts`: the model role's
    content, each function call with its id."""
    content = types.Content(role="model", parts=list(parts))
    return AgentRecord(agent="planner", kind=RESPONSE, body=_dumped(content))


def answers_record(answers: Sequence[types.FunctionResponse]) -> AgentRecord:
    """The one record of the answers of a response's calls: the user role's
    content holding `answers`' function responses, in call order."""
    content = types.Content(
        role="user", parts=[types.Part(function_response=answer) for answer in answers]
    )
    return AgentRecord(agent="planner", kind=ANSWERS, body=_dumped(content))


def _dumped(content: types.Content) -> dict[str, Any]:
    return content.model_dump(mode="json", exclude_none=True)


@dataclass(frozen=True)
class SessionCall:
    """The `planner.call.PlannerCall` of one tool call over its turn's transcript.

    `records` is the transcript as it stands when the call runs, `state` the
    state the call reads and writes, and `call_id` the id of the function
    call the tool serves, `None` outside one. The steps are the records read
    as `session_steps` reads them.
    """

    records: Records
    state: SessionState
    call_id: str | None

    @functools.cached_property
    def _steps(self) -> tuple[Step, ...]:
        return tuple(session_steps(self.records, serving=self.call_id))

    def steps(self) -> Sequence[Step]:
        """`session_steps` over the call's records, read once per call.

        A `SessionCall` serves one tool call over the records as they stood
        when it began, and the transcript gains the answers of a model
        response's calls only after they all finish, so the cached view
        cannot outlive the response it was read in.
        """
        return self._steps

    def records_read(self) -> int:
        """How many records the call reads: every record of `records`."""
        return len(self.records)

    def response_calls(self) -> Sequence[ResponseCall]:
        """The calls of the planner model's response that holds the call's own
        (`_response_holding`)."""
        if self.call_id is None:
            return ()
        calls = _response_holding(self.records, self.call_id)
        return tuple(ResponseCall(str(call["name"]), call.get("id")) for call in calls)


def _parts(record: AgentRecord, kind: str) -> list[Mapping[str, Any]]:
    """The parts of a planner record of `kind`, in order, or none for any other record."""
    if record.agent != "planner" or record.kind != kind:
        return []
    return list(record.body.get("parts") or ())


def function_calls(record: AgentRecord) -> list[Mapping[str, Any]]:
    """The function calls of a planner response record, in order."""
    return [part["function_call"] for part in _parts(record, RESPONSE) if "function_call" in part]


def function_responses(record: AgentRecord) -> list[Mapping[str, Any]]:
    """The function responses of a planner answers record, in order."""
    return [
        part["function_response"] for part in _parts(record, ANSWERS) if "function_response" in part
    ]


def _response_holding(records: Records, call_id: str) -> list[Mapping[str, Any]]:
    """The function calls of the latest model response in `records` that holds
    a call with `call_id`, in order, or none."""
    for record in reversed(records):
        calls = function_calls(record)
        if any(call.get("id") == call_id for call in calls):
            return calls
    return []


def _in_flight_calls(records: Records, serving: str | None) -> frozenset[str]:
    """The ids of the calls in the model response that holds the call `serving`
    which no record answers yet.

    `serving` is a tool that reads steps, never an ask, which decides from its
    response's call list instead (`SessionCall.response_calls`).

    The transcript gains a response's answers after all its calls finish, so
    while one call of the response runs, its siblings have no answer in it,
    whether they have finished or not.
    """
    if serving is None:
        return frozenset()
    answered = {response.get("id") for record in records for response in function_responses(record)}
    calls = _response_holding(records, serving)
    if not any(call.get("id") == serving and call.get("name") != PLANNER_ASK for call in calls):
        return frozenset()
    return frozenset(call["id"] for call in calls if call.get("id") and call["id"] not in answered)


def session_steps(records: Records, *, serving: str | None = None) -> Iterator[Step]:
    """The steps of a turn's transcript `records`, in order (`planner.call.Step`).

    The user's message (`transcript.USER`) is one `message` step. A pause the
    turn put to the user (`transcript.PAUSE`) is a `call` step under its wire
    interrupt id, its `args` holding the pause's `message` and its `payload`.
    The user's answer (`transcript.ANSWER`) is an `answer` step under the id
    of the call it answers, paired with the `args` of the nearest preceding
    call sharing that id, its body `{"result": <the answer>}`.

    The planner's own ask (`PLANNER_ASK`) is a `call` step under the call's id,
    its question flat in `args`. A planner ask its tool refused never reached
    the user. It yields no step at all, so every reader sees the transcript as
    if the ask had not been made. An accepted one's tool answers
    `ASK_ACCEPTED`, which says nothing the user did and is not read. The
    planner node pauses on it (`PLANNER_ASK_PAUSE`), and that pause yields no
    step of its own. It puts the ask its payload names (`ASKED_CALL_ID`) to
    the user, so the answer to the pause is the answer to that call. Every
    other tool's response is a `response` step.

    Each step carries the position of the record it was read from
    (`Step.record`).

    `serving` is the id of the tool call being served. The calls of its own
    model response that no record answers yet (`_in_flight_calls`) yield no
    step, so a call running beside an ask reads the transcript as it stood
    before that response, and the ask, which may still be refused or may be
    accepted, is not counted as put to the user.
    """
    refused = _refused_asks(records)
    in_flight = _in_flight_calls(records, serving)
    calls: dict[str, dict[str, Any]] = {}
    # The call each planner-side pause's wire id put to the user.
    asks: dict[str, str] = {}
    for index, record in enumerate(records):
        if record.agent == "loop":
            if record.kind == transcript.PAUSE and record.body["kind"] == PLANNER_ASK_PAUSE:
                asks[str(record.body["interrupt_id"])] = str(record.body["payload"][ASKED_CALL_ID])
                continue
            step = _loop_step(record, calls, asks)
            if step is not None:
                yield step._replace(record=index)
            continue
        for call in function_calls(record):
            call_id = call.get("id")
            if (
                call_id is None
                or call.get("name") != PLANNER_ASK
                or call_id in in_flight
                or call_id in refused
            ):
                continue
            calls[call_id] = dict(call.get("args") or {})
            yield Step("call", call_id, calls[call_id], record=index)
        for response in function_responses(record):
            if response.get("name") != PLANNER_ASK:
                yield Step(
                    "response",
                    body=response.get("response") or {},
                    name=response["name"],
                    record=index,
                )


def _loop_step(
    record: AgentRecord, calls: dict[str, dict[str, Any]], asks: Mapping[str, str]
) -> Step | None:
    """The step of a record the turn loop wrote, other than a planner-side
    pause, recording a pause's `args` in `calls`, or `None` for a record no
    reader reads."""
    body = record.body
    if record.kind == transcript.USER:
        return Step("message")
    if record.kind == transcript.PAUSE:
        interrupt_id = str(body["interrupt_id"])
        calls[interrupt_id] = {"message": body.get("message", ""), "payload": body["payload"]}
        return Step("call", interrupt_id, calls[interrupt_id])
    if record.kind == transcript.ANSWER:
        interrupt_id = str(body["interrupt_id"])
        call_id = asks.get(interrupt_id, interrupt_id)
        return Step("answer", call_id, calls.get(call_id), {"result": body["answer"]})
    return None


def _refused_asks(records: Iterable[AgentRecord]) -> frozenset[str]:
    """The ids of the planner asks that never reached the user: their tool
    answered anything but `ASK_ACCEPTED`, so it refused them."""
    return frozenset(
        response["id"]
        for record in records
        for response in function_responses(record)
        if response.get("id") is not None
        and response.get("name") == PLANNER_ASK
        and (response.get("response") or {}).get("outcome") != Outcome.OK.value
    )
