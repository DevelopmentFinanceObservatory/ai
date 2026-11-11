"""The session events a Pydantic AI planner run stores, shaped as the ADK planner's.

The session's planner events are the one stored record of a planner run on
either planner: `planner.session.session_steps`, `harness.studio_translate` and
the planner seam read them. So the node that drives `planner.pydantic_agent.
run_planner` yields, for each item the run makes, the event the ADK planner's
`LlmAgent` appends for the same exchange:

- `model_response_event` for a model response (`PlannerResponse`), as soon as
  it arrives;
- `tool_answers_event` for the answers of one response's calls
  (`PlannerAnswers`), once every call of the response has answered. The ADK
  planner runs a response's calls together and appends one event holding
  every answer, in the order the calls were made, with one state delta, after
  the last call has answered.

The builders leave out what ADK's node runner sets on every event a node
yields: the event's id and timestamp, `invocation_id`, `node_info.path` and
`node_info.output_for`, `author`, which it takes from the node's
`Context.event_author`, so the node sets that to `planner.agent.AGENT_NAME`,
and `branch`, which it takes from the `override_branch` the node is run with
(`harness.planner_seam`'s planner branch). The model response's finish reason,
model version and usage are the ones the API reported, which the node reads
off the raw google-genai response.

A run stopped or failed while a response's calls are running leaves no event
for that response's answers, so the state its calls wrote, answered or not,
is not stored. ADK leaves none on a stop, and on an error other than a
cancellation stores the answers of the calls that answered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from google.adk.events import Event, EventActions
from google.adk.events.event import NodeInfo
from google.genai import types
from planner.pydantic_agent import PlannerAnswers, PlannerResponse

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
    partial: bool | None = False,
    error_code: str | None = None,
    error_message: str | None = None,
) -> Event:
    """The event the ADK planner stores for the model response `response`.

    Its content is the response's parts as the model role's, each function
    call with its id, or none for a response with no parts. Those are the parts
    Pydantic AI kept, so an empty text chunk Gemini streams after a function
    call is not among them, where ADK's stream aggregator keeps it as a part. A
    response with
    a call names no long-running call, since no planner tool is one. A
    response with parts and no call is the run's output, its text less any
    thought, as ADK takes a finished agent turn's. A finished response whose
    parts hold nothing to act on (only thoughts or blank text), and that is
    no error of its own, carries ADK's no-content error, as a streamed run
    records it.

    Args:
        response: The model response, its unsigned thoughts already dropped
            (`planner.settings.without_unsigned_thoughts`).
        usage: The usage the API reported for the call, as google-genai reads it.
        finish_reason: Why the model stopped, as the API reported it.
        model_version: The model version the API reported, if it reported one.
        custom_metadata: What a closing turn carries
            (`harness.turn_budget.CLOSED_BY_KEY`), or None.
        partial: False for a response the model streamed, which ADK stores
            as the aggregate of its stream, and None for a model turn put in
            the model's place, which ADK stores as the turn was given.
        error_code: The error the API reported the response as, a finish
            reason other than `STOP` or a blocked prompt's reason.
        error_message: The message the API gave with that error.
    """
    parts = [part.model_copy(deep=True) for part in response.parts]
    calls = [part.function_call for part in parts if part.function_call is not None]
    event = Event(
        content=types.Content(role="model", parts=parts) if parts else None,
        partial=partial,
        finish_reason=finish_reason,
        usage_metadata=usage.model_copy(deep=True) if usage is not None else None,
        model_version=model_version,
        custom_metadata=dict(custom_metadata) if custom_metadata is not None else None,
        error_code=error_code,
        error_message=error_message,
    )
    if calls:
        event.long_running_tool_ids = set()
        return event
    if not parts:
        return event
    if error_code is None and finish_reason == types.FinishReason.STOP and not _actionable(parts):
        event.error_code = NO_CONTENT_ERROR_CODE
        event.error_message = NO_CONTENT_ERROR_MESSAGE
    event.output = "".join(part.text for part in parts if part.text and not part.thought)
    event.node_info = NodeInfo(message_as_output=True)
    return event


def tool_answers_event(answers: PlannerAnswers, *, ended: bool) -> Event:
    """The one event the ADK planner stores for the answers of a response's calls.

    The event holds `answers`' function responses, in call order, and their
    state as its state delta, which is the session's value once every call
    has answered. `ended` says whether a call ended the run
    (`planner.tools.EndRun`), which ADK records as `skip_summarization`.
    """
    parts = [
        types.Part(function_response=answer.model_copy(deep=True)) for answer in answers.answers
    ]
    return Event(
        content=types.Content(role="user", parts=parts),
        actions=EventActions(
            state_delta=answers.state,
            skip_summarization=True if ended else None,
        ),
    )


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
