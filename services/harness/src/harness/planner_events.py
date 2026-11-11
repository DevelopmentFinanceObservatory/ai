"""The session events a Pydantic AI planner run stores, shaped as the ADK planner's.

The session's planner events are the one stored record of a planner run on
either planner: `planner.agent.session_steps`, `harness.studio_translate` and
the planner seam read them. So the node that drives `planner.pydantic_agent.
run_planner` yields, for each item the run makes, the event the ADK planner's
`LlmAgent` appends for the same exchange:

- `model_response_event` for a model response (`PlannerResponse`), as soon as
  it arrives;
- `tool_answers_event` for the answers of one response's calls
  (`PlannerToolAnswer`), once every call of the response has answered
  (`answered_every_call`). The ADK planner runs a response's calls together
  and appends one event holding every answer, in the order the calls were
  made, with one state delta, after the last call has answered.

The builders leave out what ADK's node runner sets on every event a node
yields: the event's id and timestamp, `invocation_id`, `node_info.path` and
`node_info.output_for`, `author`, which it takes from the node's
`Context.event_author`, so the node sets that to `planner.agent.AGENT_NAME`,
and `branch`, which it takes from the run's `override_branch`.

A run stopped while a response's calls are running leaves no event for that
response, as the ADK planner leaves none: ADK cancels the calls and appends
nothing for them, so the state the response's calls wrote, answered or not,
is not stored. A run whose calls end on an error that is not a cancellation
stores the answers of the calls that answered, as ADK keeps them: the node
yields `tool_answers_event` over those answers alone.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from google.adk.events import Event, EventActions
from google.adk.events.event import NodeInfo
from google.genai import types
from planner.pydantic_agent import PlannerResponse, PlannerToolAnswer

type Usage = types.GenerateContentResponseUsageMetadata

# What ADK records on a finished response that holds nothing to act on (`error_code`,
# `error_message` of `google.adk.flows.llm_flows.core._model_call`).
NO_CONTENT_ERROR_CODE = "MODEL_RETURNED_NO_CONTENT"
NO_CONTENT_ERROR_MESSAGE = (
    "The model returned no actionable content (finish_reason=STOP with"
    " thought-only or whitespace-only parts)."
)


def model_response_event(
    response: PlannerResponse,
    *,
    usage: Usage | None,
    finish_reason: types.FinishReason | None,
    model_version: str | None = None,
    custom_metadata: Mapping[str, Any] | None = None,
) -> Event:
    """The event the ADK planner stores for the model response `response`.

    Its content is the response's parts as the model role's, each function
    call with its id. A response with a call names no long-running call,
    since no planner tool is one. A response with no call is the run's
    output, its text less any thought, as ADK takes a finished agent turn's.
    A finished response whose parts hold nothing to act on (only thoughts or
    blank text) carries ADK's no-content error, as a streamed run records it.

    Args:
        response: The model response, its unsigned thoughts already dropped
            (`planner.settings.without_unsigned_thoughts`).
        usage: The usage the API reported for the call, as google-genai reads it.
        finish_reason: Why the model stopped, as the API reported it.
        model_version: The model version the API reported, if it reported one.
        custom_metadata: What a closing turn carries
            (`harness.turn_budget.CLOSED_BY_KEY`), or None.
    """
    parts = [part.model_copy(deep=True) for part in response.parts]
    calls = [part.function_call for part in parts if part.function_call is not None]
    event = Event(
        content=types.Content(role="model", parts=parts),
        partial=False,
        finish_reason=finish_reason,
        usage_metadata=usage.model_copy(deep=True) if usage is not None else None,
        model_version=model_version,
        custom_metadata=dict(custom_metadata) if custom_metadata is not None else None,
    )
    if calls:
        event.long_running_tool_ids = set()
        return event
    if finish_reason == types.FinishReason.STOP and parts and not _actionable(parts):
        event.error_code = NO_CONTENT_ERROR_CODE
        event.error_message = NO_CONTENT_ERROR_MESSAGE
    event.output = "".join(part.text for part in parts if part.text and not part.thought)
    event.node_info = NodeInfo(message_as_output=True)
    return event


def answered_every_call(response: PlannerResponse, answers: Sequence[PlannerToolAnswer]) -> bool:
    """Whether `answers` answer every function call of `response`."""
    answered = {answer.call_id for answer in answers}
    return all(call_id in answered for call_id in _call_ids(response))


def tool_answers_event(
    response: PlannerResponse, answers: Sequence[PlannerToolAnswer], *, ended: bool
) -> Event:
    """The one event the ADK planner stores for the answers of `response`'s calls.

    `answers` are given in the order the calls answered, as `run_planner`
    yields them. The event holds each answer as a function response, in the
    order of the calls in `response`. Its state delta merges each call's
    writes in that order, a later call's value replacing an earlier one's and
    two dicts merging key by key, as ADK merges them. A list or dict that more
    than one call wrote then takes the value of the call that answered last,
    as ADK takes the value the session holds once every call has answered.
    `ended` says whether a call ended the run (`planner.tools.EndRun`), which
    ADK records as `skip_summarization`.

    Raises:
        ValueError: An answer is to no call of `response`, or `answers` is empty.
    """
    if not answers:
        raise ValueError("a function response event needs at least one answer")
    order = {call_id: position for position, call_id in enumerate(_call_ids(response))}
    stray = [answer.call_id for answer in answers if answer.call_id not in order]
    if stray:
        raise ValueError(f"answers to calls the response did not make: {stray}")
    in_call_order = sorted(answers, key=lambda answer: order[answer.call_id])
    parts = [
        types.Part(
            function_response=types.FunctionResponse(
                id=answer.call_id, name=answer.name, response=copy.deepcopy(answer.answer)
            )
        )
        for answer in in_call_order
    ]
    return Event(
        content=types.Content(role="user", parts=parts),
        actions=EventActions(
            state_delta=_merged_writes(in_call_order, answered_last_first=answers[::-1]),
            skip_summarization=True if ended else None,
        ),
    )


def _call_ids(response: PlannerResponse) -> list[str]:
    return [
        part.function_call.id
        for part in response.parts
        if part.function_call is not None and part.function_call.id is not None
    ]


def _merged_writes(
    in_call_order: Sequence[PlannerToolAnswer], *, answered_last_first: Sequence[PlannerToolAnswer]
) -> dict[str, Any]:
    """The calls' writes merged as ADK merges the state deltas of parallel calls."""
    merged: dict[str, Any] = {}
    for answer in in_call_order:
        _deep_merge(merged, copy.deepcopy(answer.writes))
    writers = Counter(key for answer in in_call_order for key in answer.writes)
    latest: dict[str, Any] = {}
    for key, count in writers.items():
        if count < 2:
            continue
        value = next(answer.writes[key] for answer in answered_last_first if key in answer.writes)
        if isinstance(value, dict | list):
            latest[key] = copy.deepcopy(value)
    return _deep_merge(merged, latest)


def _deep_merge(into: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """`other` merged into `into`: dicts key by key, anything else replaced."""
    for key, value in other.items():
        if isinstance(into.get(key), dict) and isinstance(value, dict):
            _deep_merge(into[key], value)
        else:
            into[key] = value
    return into


def _actionable(parts: Sequence[types.Part]) -> bool:
    """Whether any of `parts` is something to act on: a call, a response, code,
    data, or text that is no thought and not blank."""
    return any(
        part.function_call is not None
        or part.function_response is not None
        or part.executable_code is not None
        or part.code_execution_result is not None
        or part.inline_data is not None
        or part.file_data is not None
        or (part.text is not None and not part.thought and part.text.strip() != "")
        for part in parts
    )
