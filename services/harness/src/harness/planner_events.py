"""The session events a planner run stores.

The session's planner events are the one stored record of a planner run:
`planner.session.session_steps`, `harness.studio_translate` and the planner
seam read them. They read each event's author, branch, invocation, node path,
content and state delta, so the node that drives
`planner.pydantic_agent.run_planner` yields, for each item the run makes, an
event holding the content and state delta an ADK `LlmAgent` stores for the
same exchange:

- `model_response_event` for a model response with parts (`PlannerResponse`),
  as soon as it arrives;
- `tool_answers_event` for the answers of one response's calls
  (`PlannerAnswers`), once every call of the response has answered: one
  event holding every answer, in the order the calls were made, with one
  state delta, as an `LlmAgent` appends it after the last call has answered.

ADK's node runner sets the rest of each event: its id and timestamp,
`invocation_id`, `node_info.path`, `author`, which it takes from the node's
`Context.event_author`, so the node sets that to `planner.settings.AGENT_NAME`,
and `branch`, which it takes from the `override_branch` the node is run with
(`harness.planner_adapter`'s planner branch).

A run stopped or failed while a response's calls are running leaves no event
for that response's answers, so the state its calls wrote, answered or not,
is not stored. ADK leaves none on a stop, and on an error other than a
cancellation stores the answers of the calls that answered.
"""

from __future__ import annotations

from google.adk.events import Event, EventActions
from google.genai import types
from planner.pydantic_agent import PlannerAnswers, PlannerResponse


def model_response_event(response: PlannerResponse) -> Event:
    """The event the planner stores for the model response `response`, holding
    its parts as the model role's content, each function call with its id.

    Those are the parts Pydantic AI kept, so an empty text chunk Gemini streams
    after a function call is not among them, where ADK's stream aggregator
    keeps it as a part. `response` holds at least one part, its unsigned
    thoughts already dropped (`planner.settings.without_unsigned_thoughts`).
    """
    parts = [part.model_copy(deep=True) for part in response.parts]
    return Event(content=types.Content(role="model", parts=parts))


def tool_answers_event(answers: PlannerAnswers) -> Event:
    """The one event the planner stores for the answers of a response's calls.

    The event holds `answers`' function responses, in call order, and their
    state as its state delta, which is the session's value once every call
    has answered.
    """
    parts = [
        types.Part(function_response=answer.model_copy(deep=True)) for answer in answers.answers
    ]
    return Event(
        content=types.Content(role="user", parts=parts),
        actions=EventActions(state_delta=answers.state),
    )
