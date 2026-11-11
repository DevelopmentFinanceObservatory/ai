"""The ADK planner and the Pydantic AI planner run on one script, for tests that compare them.

`adk_run` runs the ADK planner (`create_planner_agent` on `Gemini` over the
scripted client) through `Runner`, in a workflow node that calls
`ctx.run_node` once per brief, as `harness.planner_seam.PlannerSeam` does: a
run after the first takes a `run_id` of its own, as the run the user's answer
resumes does. `pydantic_run` runs the Pydantic AI planner (`planner_agent` on
`harness.planner_model.planner_model` over the scripted client) once per
brief, each run after the first handed the steps of the runs before it.

Each returns the requests the scripted client recorded and the conversation
the run leaves (`Turn`), read off the ADK planner's events or the Pydantic AI
planner's items in one shape, so a test compares the two directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from compiler.surface import ToolSurface
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow, node
from google.genai import types
from harness.planner_model import planner_model
from orchestrator.agent import model_retry_options
from planner.agent import (
    AGENT_NAME,
    BRIEF_STATE_KEY,
    create_planner_agent,
    planner_instruction,
    session_steps,
)
from planner.pydantic_agent import (
    PlannerDeps,
    PlannerItem,
    PlannerResponse,
    planner_agent,
    planner_profile,
    run_planner,
)
from planner.specs import planner_tool_specs, skill_tool_spec
from translator.tool import Translate

from ..scripted_genai import Reply, Request, ScriptedGenai

MODEL = "gemini-3.8-flash"
QUESTION = "How much aid did Ghana receive?"
BRIEF = "Context for this question: the readings."

# The branch `PlannerSeam` runs the planner on (`harness.planner_seam`).
_BRANCH = "dfo_planner"

# One entry of a run's conversation: a model response's part (`types.Part`
# dumped), a call's answer (`("answer", name, call_id, answer)`), or a blocked
# prompt (`("blocked", reason)`).
type Turn = tuple[Any, ...]


@dataclass
class PairRun:
    """What one planner's runs sent and left: the requests in order, and each
    run's conversation (`Turn`s), in order."""

    requests: list[Request]
    conversations: list[list[Turn]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


async def adk_run(
    replies: Sequence[Reply],
    *,
    briefs: Sequence[str] = (BRIEF,),
    surface: ToolSurface | None = None,
    translate: Translate | None = None,
) -> PairRun:
    """The ADK planner's runs on `replies`, one per brief, in one session."""
    scripted = ScriptedGenai(replies)
    agent = create_planner_agent(
        model_name=MODEL,
        translate=translate,
        surface=surface,
        question_fixed=True,
        reads_user_words=False,
    )
    agent.model = Gemini(model=MODEL, client=scripted.client)
    marks: list[int] = []

    async def planner(ctx: Any, node_input: Any) -> None:
        del node_input
        for position, brief in enumerate(briefs):
            marks.append(len(ctx.session.events))
            ctx.state[BRIEF_STATE_KEY] = brief
            await ctx.run_node(
                agent,
                None,
                run_id=f"run-{position}" if position else None,
                override_branch=_BRANCH,
            )
        marks.append(len(ctx.session.events))

    workflow = Workflow(
        name="turn", edges=[(START, node(planner, name="planner", rerun_on_resume=True))]
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="app", user_id="user")
    runner = Runner(node=workflow, app_name="app", session_service=sessions)
    async for _ in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=QUESTION)]),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        pass
    stored = await sessions.get_session(app_name="app", user_id="user", session_id=session.id)
    assert stored is not None
    # The marks count the session's events as the node saw them, which are the stored ones.
    events = stored.events
    return PairRun(
        requests=scripted.requests,
        conversations=[_event_turns(events[start:end]) for start, end in pairwise(marks)],
        state={key: value for key, value in stored.state.items() if not key.startswith("temp:")},
    )


async def pydantic_run(
    replies: Sequence[Reply],
    *,
    briefs: Sequence[str] = (BRIEF,),
    surface: ToolSurface | None = None,
    translate: Translate | None = None,
) -> PairRun:
    """The Pydantic AI planner's runs on `replies`, one per brief, over one state.

    Each run after the first is handed the steps of the turn so far, read by
    `session_steps` from the events the runs before it would store.
    """
    scripted = ScriptedGenai(replies)
    specs = planner_tool_specs(
        translate=translate, question_fixed=True, reads_user_words=False, surface=surface
    )
    planner = planner_agent(
        [*([skill_tool_spec(surface)] if surface is not None else []), *specs],
        instruction=planner_instruction(
            has_translator=translate is not None,
            has_skills=surface is not None,
            question_fixed=True,
        ),
        model=planner_model(
            MODEL, client=scripted.client, budget=None, profile=planner_profile(MODEL)
        ),
    )
    state: dict[str, Any] = {}
    events = [Event(author="user", content=_user_content(QUESTION))]
    run = PairRun(requests=scripted.requests, state=state)
    for brief in briefs:
        deps = PlannerDeps(
            state=state,
            earlier=tuple(session_steps(events)),
            tool_answered=lambda name, answer: None,
            tool_failed=lambda name, error: {"error": f"{name} failed"},
        )
        items = [item async for item in run_planner(planner, deps, question=QUESTION, brief=brief)]
        run.conversations.append(_item_turns(items))
        events += _stored(items)
    return run


def _user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def _dumped(part: types.Part) -> dict[str, Any]:
    return part.model_dump(mode="json", exclude_none=True)


def _event_turns(events: Sequence[Event]) -> list[Turn]:
    """The turns of the ADK planner's events among `events`."""
    turns: list[Turn] = []
    for event in events:
        if event.author != AGENT_NAME:
            continue
        if event.error_code:
            turns.append(("blocked", event.error_code))
        for part in event.content.parts or [] if event.content else []:
            response = part.function_response
            if response is None:
                turns.append(("part", _dumped(part)))
            else:
                turns.append(("answer", response.name, response.id, response.response))
    return turns


def _item_turns(items: Sequence[PlannerItem]) -> list[Turn]:
    """The turns of the Pydantic AI planner's `items`. A response with no parts and
    no block reason is one the ADK planner stores no event for."""
    turns: list[Turn] = []
    for item in items:
        if isinstance(item, PlannerResponse):
            if item.block_reason is not None:
                turns.append(("blocked", item.block_reason))
            turns.extend(("part", _dumped(part)) for part in item.parts)
        else:
            turns.append(("answer", item.name, item.call_id, item.answer))
    return turns


def _stored(items: Sequence[PlannerItem]) -> list[Event]:
    """The events a run's `items` stand for, as far as `session_steps` reads them."""
    events: list[Event] = []
    for item in items:
        if isinstance(item, PlannerResponse):
            if item.parts:
                content = types.Content(role="model", parts=list(item.parts))
                events.append(Event(author=AGENT_NAME, branch=_BRANCH, content=content))
            continue
        response = types.FunctionResponse(id=item.call_id, name=item.name, response=item.answer)
        content = types.Content(role="user", parts=[types.Part(function_response=response)])
        events.append(Event(author=AGENT_NAME, branch=_BRANCH, content=content))
    return events


def assert_pinned_wire_differences(adk: Request, pydantic: Request) -> None:
    """Assert that the Pydantic AI planner's request equals the ADK planner's for
    the same turn apart from how each framework shapes a request on the wire.

    The contents are equal, and every config field but six: the instruction as
    one user `Content` where the ADK planner sends text; one `Tool` per
    declaration where the ADK planner sends one `Tool` of them all, a
    declaration with no parameters given an empty object schema; an explicit
    `AUTO` function calling mode where the ADK planner leaves it unset, which
    is `AUTO`; `response_modalities` `TEXT`; `automatic_continuation` unset,
    which google-genai reads as on, where the ADK planner sets it on; and each
    framework's HTTP headers, with the HTTP retries the ADK planner's model
    sets on its client carried on each request.
    """
    adk_config, adk_contents = _request_dumped(adk)
    pydantic_config, pydantic_contents = _request_dumped(pydantic)
    assert pydantic_contents == adk_contents
    differing = {
        key
        for key in adk_config.keys() | pydantic_config.keys()
        if adk_config.get(key) != pydantic_config.get(key)
    }
    assert differing == {
        "system_instruction",
        "tools",
        "tool_config",
        "response_modalities",
        "automatic_continuation",
        "http_options",
    }
    assert pydantic_config["system_instruction"] == {
        "role": "user",
        "parts": [{"text": adk_config["system_instruction"]}],
    }
    (adk_tool,) = adk_config["tools"]
    assert pydantic_config["tools"] == [
        {
            "function_declarations": [
                declaration
                if "parameters_json_schema" in declaration
                else {**declaration, "parameters_json_schema": {"type": "object", "properties": {}}}
            ]
        }
        for declaration in adk_tool["function_declarations"]
    ]
    assert pydantic_config["tool_config"] == {"function_calling_config": {"mode": "AUTO"}}
    assert "tool_config" not in adk_config
    assert pydantic_config["response_modalities"] == ["TEXT"]
    assert "response_modalities" not in adk_config
    assert adk_config["automatic_continuation"] is True
    assert "automatic_continuation" not in pydantic_config
    assert adk_config["http_options"].keys() == {"headers"}
    assert pydantic_config["http_options"].keys() == {"headers", "retry_options"}
    assert pydantic_config["http_options"]["retry_options"] == model_retry_options().model_dump(
        mode="json", exclude_none=True
    )


def _request_dumped(request: Request) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = request.generate_config.model_dump(mode="json", exclude_none=True)
    contents = [
        content.model_dump(mode="json", exclude_none=True) for content in request.content_list
    ]
    return config, contents
