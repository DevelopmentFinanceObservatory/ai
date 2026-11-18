"""One orchestrator run inside a turn: what the turn loop calls and what it gets back.

`run_orchestrator` runs the agent from the turn's opening or from the
hand-back that answers its deferred call, and yields `OrchestratorEvent`s as
they happen:

- `TextDelta`: answer text as the model streams it (Studio's `message.delta`).
- `ModelResponded`: a whole model response, once streamed, before its tools
  run (Activity lines for its calls; the thoughts are for traces).
- `ToolsAnswered`: the answers of one response's calls once every call that
  ran has answered, and a copy of the state those calls wrote.
- `HandedOff`: the run ended on a `plan` or `ask_user` call, which waits on a
  hand-back. It carries the hand-off's record and the history to store with
  the turn until the hand-back arrives.
- `Answered`: the run ended on the reply. Nothing is stored: the next turn
  opens from the digests.

The run happens in the caller's task, so cancelling the caller cancels the
model stream and any tool in flight. A call a bound of the turn's budget
closes ends the run on the bound's sentence (`orchestrator.model`), as an
`Answered` like any other; `TurnBudget.closed` says which bound it was.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncGenerator, MutableMapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Literal

from common.telemetry import current_trace_id
from google.genai import types
from harness.gemini_agent import (
    NO_REQUEST_LIMIT,
    QUESTION_INSTRUCTIONS,
    EmptyResponseError,
    RecordingState,
    as_function_response,
    holds_no_answer,
    response_parts,
    tool_not_found,
    without_unsigned_thoughts,
)
from harness.gemini_wire import StreamResumedError, resumed_over_output
from pydantic import BaseModel, SerializerFunctionWrapHandler, field_serializer, field_validator
from pydantic_ai import (
    Agent,
    Conversation,
    DeferredToolRequests,
    DeferredToolResults,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model

from orchestrator.agent import INSTRUCTION, Orchestrator, OrchestratorDeps

logger = logging.getLogger(__name__)

# Carried by the request holding the date and digests. Pydantic AI merges
# consecutive requests into one content unless their `instructions` differ, so
# it is a content of its own, as is the user's message when a seed hand-back
# follows it (`harness.gemini_agent.QUESTION_INSTRUCTIONS`). Only the latest
# request's instructions reach the model, so neither text is sent.
_DIGESTS_INSTRUCTIONS = "The turn's date and digests."

# `INSTRUCTION` as each request records it, which Pydantic AI strips of
# surrounding whitespace, and what a stored request carries in its place
# (`StoredHistory`).
_RECORDED_INSTRUCTION = INSTRUCTION.strip()
_STORED_INSTRUCTION = "orchestrator.agent.INSTRUCTION"


def _with_instruction_as(
    messages: list[ModelMessage], instruction: str, stored: str
) -> list[ModelMessage]:
    """`messages`, each request whose instructions are `instruction` carrying
    `stored` in their place."""
    return [
        replace(message, instructions=stored)
        if isinstance(message, ModelRequest) and message.instructions == instruction
        else message
        for message in messages
    ]


class StoredHistory(BaseModel):
    """The orchestrator's conversation while its turn waits on a hand-off.

    `schema_version` names this envelope's form, so a release that changes it
    can tell a stored turn of an older form. `conversation` is Pydantic AI's:
    the messages, with thought signatures in each part's `provider_details`,
    and the deferred call with its record.

    Each request Pydantic AI recorded carries the agent's whole instruction,
    about 15 KB, though only the latest request's reaches the model. A dump
    puts a short marker in its place and a load puts the instruction back, so a
    resumed run reads the conversation it would have read unstored. The
    marker differs from every other request's instructions, so requests merge
    into contents as they did before the dump.
    """

    schema_version: Literal[1] = 1
    conversation: Conversation

    @field_serializer("conversation", mode="wrap")
    def _dumped(self, conversation: Conversation, handler: SerializerFunctionWrapHandler) -> Any:
        messages = _with_instruction_as(
            conversation.messages, _RECORDED_INSTRUCTION, _STORED_INSTRUCTION
        )
        return handler(replace(conversation, messages=messages))

    @field_validator("conversation", mode="after")
    @classmethod
    def _loaded(cls, conversation: Conversation) -> Conversation:
        messages = _with_instruction_as(
            conversation.messages, _STORED_INSTRUCTION, _RECORDED_INSTRUCTION
        )
        return replace(conversation, messages=messages)


@dataclass(frozen=True)
class Opening:
    """What a turn's first run starts from: the user's `question`, the UTC
    date `today`, the canvas and conversation digests as JSON, and the seed
    hand-back (`seeded`), the datasets the user brought from Search, when the
    turn opened with one."""

    question: str
    today: date
    canvas_digest: str | None = None
    conversation_digest: str | None = None
    seeded: str | None = None


@dataclass(frozen=True)
class HandBack:
    """The result of the run's deferred `answers` call: the planner's result
    (`plan`) or the user's answer (`ask_user`)."""

    answers: Literal["plan", "ask_user"]
    result: dict[str, Any]


@dataclass(frozen=True)
class TextDelta:
    """Answer text, as the model streamed it."""

    text: str


@dataclass(frozen=True)
class ModelResponded:
    """A model response, once streamed: its parts as the google-genai parts
    Gemini sent, each with its thought signature, less unsigned thought
    summaries."""

    parts: tuple[types.Part, ...]


@dataclass(frozen=True)
class ToolsAnswered:
    """The answers of one response's calls, with each call's id, in call
    order, and a copy of the turn state's value of every key those calls
    wrote, so a later call's change does not reach an event still queued. A
    call that handed off has no answer here; a run resumed after it opens on
    one whose answer is the hand-back."""

    answers: tuple[types.FunctionResponse, ...]
    state: dict[str, Any]


@dataclass(frozen=True)
class HandedOff:
    """The run ended on `tool`'s call `call_id`, which waits on a hand-back.
    `record` is the tool's answer that recorded the hand-off, and `history`
    the conversation to store until the hand-back arrives."""

    tool: str
    call_id: str
    record: dict[str, Any]
    history: StoredHistory


@dataclass(frozen=True)
class Answered:
    """The run ended on the reply `text`, a bound's closing sentence included."""

    text: str


type OrchestratorEvent = TextDelta | ModelResponded | ToolsAnswered | HandedOff | Answered


class NoCallWaitsError(RuntimeError):
    """A hand-back answers no call the stored history waits on."""


async def run_orchestrator(
    orchestrator: Orchestrator,
    *,
    state: MutableMapping[str, Any],
    turn_id: str,
    model: Model,
    opening: Opening | None = None,
    history: StoredHistory | None = None,
    hand_back: HandBack | None = None,
    conversation_id: str | None = None,
) -> AsyncGenerator[OrchestratorEvent]:
    """Run the orchestrator once, from `opening` or from `hand_back`.

    A turn's first run takes `opening`. A run after a hand-off takes the
    `history` `HandedOff` gave and the `hand_back` that answers its deferred
    call. `state` is the turn's state and `turn_id` its id, the same on every
    run of the turn: each tool call reads them as its
    `orchestrator.call.OrchestratorCall`. `model` is the run's model
    (`orchestrator.model.orchestrator_model`, on the turn's budget).
    `conversation_id` is the conversation the run's spans name
    (`gen_ai.conversation.id`); on a resume it takes the place of the id the
    stored conversation opened with. None keeps the stored id, or leaves
    Pydantic AI to make one up for a first run.

    The first request opens with the date and the digests as one user content,
    then the user's message as its own, then the seed hand-back when there is
    one. A response with neither answer text nor a function call ends the run
    (`harness.gemini_agent.holds_no_answer`) and is not sent back for another
    try. Any other exception the model or Pydantic AI raises propagates.

    Raises:
        EmptyResponseError: A model response held neither answer text nor a
            function call.
        StreamResumedError: google-genai resumed a stream over output no
            continuation token covers (`harness.gemini_wire.resumed_over_output`),
            a dropped connection a later attempt may not meet.
        NoCallWaitsError: `hand_back` answers no call `history` waits on.
        ValueError: Neither or both of an opening and a resume were given.
    """
    deps = OrchestratorDeps(state=state, turn_id=turn_id)
    if opening is not None and history is None and hand_back is None:
        prompt, messages = _opening(opening)
        conversation, results = None, None
    elif opening is None and history is not None and hand_back is not None:
        prompt, messages = None, None
        conversation = history.conversation
        if conversation_id is not None:
            # Pydantic AI takes the id from the conversation, and refuses one
            # passed beside it.
            conversation = replace(conversation, conversation_id=conversation_id)
            conversation_id = None
        results = DeferredToolResults(calls={_waiting(history, hand_back): hand_back.result})
    else:
        raise ValueError("give an opening for a turn's first run, or a history and a hand-back")

    try:
        async with orchestrator.agent.iter(
            prompt,
            model=model,
            deps=deps,
            message_history=messages,
            conversation=conversation,
            deferred_tool_results=results,
            usage_limits=NO_REQUEST_LIMIT,
            conversation_id=conversation_id,
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    deps.view = RecordingState(deps.state)
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if (text := _text(event)) is not None:
                                yield TextDelta(text)
                        response = stream.response
                    if holds_no_answer(response):
                        raise EmptyResponseError(
                            "the model's response held neither answer text nor a function call"
                        )
                    yield ModelResponded(parts=response_parts(without_unsigned_thoughts(response)))
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as stream:
                        answered = [
                            _answer(event, orchestrator.tools)
                            async for event in stream
                            if isinstance(event, FunctionToolResultEvent)
                        ]
                    if answered:
                        yield ToolsAnswered(
                            answers=tuple(answered),
                            state=_written_state(deps),
                        )
            result = run.result
    except Exception as exc:
        if not resumed_over_output(exc):
            raise
        logger.warning(
            "orchestrator stream resumed over output turn_id=%s trace_id=%s: %s",
            turn_id,
            current_trace_id(),
            exc,
        )
        raise StreamResumedError(str(exc)) from exc
    assert result is not None
    output = result.output
    if isinstance(output, DeferredToolRequests):
        (deferred,) = output.calls
        yield HandedOff(
            tool=deferred.tool_name,
            call_id=deferred.tool_call_id,
            record=output.metadata.get(deferred.tool_call_id, {}).get("record", {}),
            history=StoredHistory(conversation=result.conversation),
        )
        return
    yield Answered(text=output)


def _opening(opening: Opening) -> tuple[str, list[ModelMessage]]:
    """The prompt and the history before it that open a turn: the date and the
    digests as one request (`_DIGESTS_INSTRUCTIONS`), then the user's message
    as the prompt, or as a request of its own (`QUESTION_INSTRUCTIONS`)
    ahead of the seed hand-back as the prompt."""
    lines = [f"Today's date (UTC): {opening.today.isoformat()}"]
    lines += [
        f"{label}: {digest}"
        for label, digest in (
            ("The user's canvas", opening.canvas_digest),
            ("Earlier turns in this conversation", opening.conversation_digest),
        )
        if digest
    ]
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("\n\n".join(lines))], instructions=_DIGESTS_INSTRUCTIONS)
    ]
    if not opening.seeded:
        return opening.question, messages
    messages.append(
        ModelRequest(parts=[UserPromptPart(opening.question)], instructions=QUESTION_INSTRUCTIONS)
    )
    return opening.seeded, messages


def _waiting(history: StoredHistory, hand_back: HandBack) -> str:
    """The id of the one deferred call of `history` that `hand_back` answers."""
    requests = history.conversation.deferred_tool_requests
    calls = [
        call for call in (requests.calls if requests else ()) if call.tool_name == hand_back.answers
    ]
    if len(calls) != 1:
        raise NoCallWaitsError(f"no one {hand_back.answers} call waits on this hand-back")
    return calls[0].tool_call_id


def _text(event: object) -> str | None:
    """The answer text a stream event adds, if it adds any."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        return event.part.content or None
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return event.delta.content_delta or None
    return None


def _written_state(deps: OrchestratorDeps) -> dict[str, Any]:
    """A deep copy of each state key the response's calls wrote, for a
    `ToolsAnswered` event that may wait in the translator's queue while
    later calls change the state, in place or not."""
    return {key: copy.deepcopy(deps.state[key]) for key in deps.view.written if key in deps.state}


def _answer(event: FunctionToolResultEvent, tools: Sequence[str]) -> types.FunctionResponse:
    """A call's answer as the function response the model reads: the tool's
    own answer, or `tool_not_found`'s for a tool the agent lacks, as the
    history replays it (`harness.gemini_agent.replayed_history`)."""
    part = event.part
    if part.tool_name not in tools:
        body = tool_not_found(part.tool_name, tools)
    elif isinstance(part, ToolReturnPart):
        body = as_function_response(part.content)
    else:
        body = {"error": part.model_response()}
    return types.FunctionResponse(id=part.tool_call_id, name=part.tool_name, response=body)


__all__ = [
    "Answered",
    "HandBack",
    "HandedOff",
    "ModelResponded",
    "NoCallWaitsError",
    "Opening",
    "OrchestratorEvent",
    "StoredHistory",
    "TextDelta",
    "ToolsAnswered",
    "run_orchestrator",
]
