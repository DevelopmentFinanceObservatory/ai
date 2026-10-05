"""The orchestrator engine's path through the graph, run by a real ADK runner
with a scripted model: answer, ask, plan, and what the agent reads after each."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.graph import (
    APPROVE_INTERRUPT,
    CLARIFY_INTERRUPT,
    ORCHESTRATOR_BRIEF_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    GenerativeNodes,
    create_root_workflow,
)
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_seam import PlannerSeamError
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.scripted_plan import scripted_fetch_series_stats
from harness.studio_translate import _node_name
from harness.wiring import handing_back_planner_failures
from orchestrator.agent import create_orchestrator_agent
from pydantic import Field


class ScriptLlm(BaseLlm):
    """Answers each model call with the next scripted part, recording requests."""

    model: str = "script-llm"
    script: list[types.Part] = Field(default_factory=list)
    requests: list[Any] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        self.requests.append(llm_request)
        part = self.script.pop(0) if self.script else types.Part.from_text(text="done")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class FailingPlanner:
    """A planner seam that raises `PlannerSeamError`, the way `PlannerSeam` fails."""

    def __init__(self, message: str = "You have not called submit_plan.") -> None:
        self.message = message

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise PlannerSeamError(self.message)


class RefusingPlanner:
    """A planner seam that refuses to draft, as `PlannerSeam` does at its round cap."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.questions.append(kwargs["question"])
        return {"refused": {"text": "the planner ran out of rounds"}}


async def _no_resolver(**kwargs: Any) -> dict[str, Any]:
    raise AssertionError("no search expected")


async def _no_executor(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    raise AssertionError(f"no tool call expected, got {tool}")


async def _describes(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    assert tool == "describe_frame", tool
    return {"outcome": "ok", "view": {"dimensions": [{"name": "observationAbout"}]}}


def _call(name: str, **args: Any) -> types.Part:
    return types.Part.from_function_call(name=name, args=args)


def create_request_input_response(interrupt_id: str, response: dict[str, Any]) -> types.Part:
    """The user's answer to a pause, as ADK's `adk_request_input` tool reads it."""
    return types.Part(
        function_response=types.FunctionResponse(
            id=interrupt_id, name="adk_request_input", response=response
        )
    )


def _texts(request: Any) -> str:
    return "\n".join(p.text or "" for c in request.contents for p in c.parts or [])


def _runner(
    *script: types.Part, planner: Any = None, tool_surface: Any = None, executor: Any = None
) -> tuple[Runner, ScriptLlm]:
    model = ScriptLlm(script=list(script))
    agent = create_orchestrator_agent(
        model_name="unused",
        tools=build_orchestrator_tools(resolver=_no_resolver, executor=executor or _no_executor),
    ).model_copy(update={"model": model})
    workflow = create_root_workflow(
        resolver=_no_resolver,
        executor=executor or _no_executor,
        tool_surface=tool_surface,
        generative=GenerativeNodes(planner=planner or RefusingPlanner()),
        fetch_series_stats=scripted_fetch_series_stats,
        orchestrator=agent,
    )
    runner = Runner(
        app=App(name="t", root_agent=workflow), session_service=InMemorySessionService()
    )
    return runner, model


async def _session(runner: Runner, **state: Any) -> None:
    await runner.session_service.create_session(
        app_name="t", user_id="u", session_id="s", state={"surface": "studio", **state}
    )


async def _run(runner: Runner, message: types.Content) -> list[Event]:
    return [
        event async for event in runner.run_async(user_id="u", session_id="s", new_message=message)
    ]


def _pause(events: list[Event]) -> types.FunctionCall:
    calls = [c for e in events for c in e.get_function_calls() if c.name == "adk_request_input"]
    assert len(calls) == 1, [c.id for c in calls]
    return calls[0]


def _visited(events: list[Event]) -> list[str]:
    return [_node_name(e) for e in events if e.node_info]


async def test_an_answered_turn_ends_after_the_agent_without_a_pause():
    runner, model = _runner(types.Part.from_text(text="Kenya's GDP was $101 billion in 2020."))
    await _session(runner)

    events = await _run(runner, types.UserContent("Kenya's GDP in 2020?"))

    assert _texts(model.requests[0]).count("Kenya's GDP in 2020?") == 1
    assert not [e for e in events if e.error_code]
    assert not [c for e in events for c in e.get_function_calls() if c.name == "adk_request_input"]
    assert "route" in _visited(events) and "ask" not in _visited(events)


async def test_ask_pauses_with_generated_option_ids_and_hands_the_choice_back_in_words():
    runner, model = _runner(
        _call("ask_user", question="Which Congo?", options=["Republic of the Congo", "DR Congo"]),
        types.Part.from_text(text="Asked."),
        types.Part.from_text(text="The Republic of the Congo's GDP was $15.7 billion in 2024."),
    )
    await _session(runner)

    pause = _pause(await _run(runner, types.UserContent("GDP of Congo?")))
    assert pause.id == CLARIFY_INTERRUPT
    assert pause.args["payload"]["candidates"] == [
        {"spec_id": "o1", "label": "Republic of the Congo"},
        {"spec_id": "o2", "label": "DR Congo"},
    ]
    answer = create_request_input_response(pause.id, {"result": {"spec_id": "o1"}})
    events = await _run(runner, types.Content(role="user", parts=[answer]))

    resumed = _texts(model.requests[-1])
    assert "GDP of Congo?" in resumed
    assert "You asked the user: Which Congo?\nThe user chose: Republic of the Congo" in resumed
    assert not [e for e in events if e.error_code]


async def test_a_free_text_answer_and_a_skip_are_handed_back_as_such():
    for answer, expected in [
        ({"text": "the big one"}, "The user answered in their own words: the big one"),
        ({"skip": True}, "The user skipped the question. Do not answer it"),
    ]:
        runner, model = _runner(
            _call("ask_user", question="Which Congo?", options=["Republic of the Congo"]),
            types.Part.from_text(text="Asked."),
            types.Part.from_text(text="Done."),
        )
        await _session(runner)
        pause = _pause(await _run(runner, types.UserContent("GDP of Congo?")))

        reply = create_request_input_response(pause.id, {"result": answer})
        await _run(runner, types.Content(role="user", parts=[reply]))

        assert expected in _texts(model.requests[-1])


async def test_a_planner_that_refuses_hands_back_and_the_agent_keeps_the_question():
    planner = RefusingPlanner()
    runner, model = _runner(
        _call("plan", brief="Niger's Global Fund receipts as a share of health spending"),
        types.Part.from_text(text="I have handed this to the planner."),
        types.Part.from_text(text="The share could not be calculated."),
        planner=planner,
    )
    await _session(runner)

    events = await _run(runner, types.UserContent("How much does Niger get from the Global Fund?"))

    assert planner.questions == ["Niger's Global Fund receipts as a share of health spending"]
    after_plan = _texts(model.requests[-1])
    assert "How much does Niger get from the Global Fund?" in after_plan
    assert "The planner could not make a plan: the planner ran out of rounds" in after_plan
    assert not [e for e in events if e.error_code]
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session.state.get("question") in (None, "")
    assert session.state[ORCHESTRATOR_BRIEF_KEY].startswith("Niger's")


async def test_a_failed_plan_gives_the_plan_call_back_once():
    planner = RefusingPlanner()
    runner, _ = _runner(
        _call("plan", brief="Kenya's aid as a share of GDP"),
        types.Part.from_text(text="Handed."),
        _call("plan", brief="Kenya's CRS aid as a share of GDP"),
        types.Part.from_text(text="Handed again."),
        _call("plan", brief="Kenya's aid as a share of GDP, any source"),
        types.Part.from_text(text="Could not be calculated."),
        planner=planner,
    )
    await _session(runner)

    events = await _run(runner, types.UserContent("Kenya's aid as a share of GDP?"))

    assert len(planner.questions) == 2
    third = [
        r.response
        for e in events
        for r in e.get_function_responses()
        if r.name == "plan" and r.response and "error" in r.response
    ]
    assert len(third) == 1 and "limit" in third[0]["error"]


async def test_a_planner_seam_failure_is_handed_back_when_wrapped_and_fatal_when_not():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt"),
        types.Part.from_text(text="Handed."),
        types.Part.from_text(text="Could not be calculated."),
        planner=handing_back_planner_failures(FailingPlanner()),
    )
    await _session(runner)

    events = await _run(runner, types.UserContent("Kenya's debt?"))

    assert "The planner could not make a plan: You have not called submit_plan." in _texts(
        model.requests[-1]
    )
    assert not [e for e in events if e.error_code]

    bare, _ = _runner(_call("plan", brief="Kenya's debt"), planner=FailingPlanner())
    await _session(bare)
    with pytest.raises(PlannerSeamError):
        await _run(bare, types.UserContent("Kenya's debt?"))


async def test_an_approved_plan_runs_and_its_final_frames_come_back_to_the_agent():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await _session(runner)

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    assert approval.id == APPROVE_INTERRUPT
    approve = create_request_input_response(
        approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
    )
    events = await _run(runner, types.Content(role="user", parts=[approve]))

    after_plan = _texts(model.requests[-1])
    assert "The approved plan ran. Chart its frames and answer." in after_plan
    body = json.loads(after_plan.split("Chart its frames and answer.\n", 1)[1].split("\n", 1)[0])
    assert body["frames"] and all(f["frame"].startswith("f") for f in body["frames"])
    assert body["step_summaries"]
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert set(session.state[ORCHESTRATOR_FRAMES_KEY]) == {f["frame"] for f in body["frames"]}
    assert not [e for e in events if e.error_code]


async def _refuses_to_describe(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    assert tool == "describe_frame", tool
    return {
        "outcome": "refusal",
        "refusal": {"code": "publish_not_configured", "retry_mode": "do_not_retry"},
    }


async def test_a_plan_whose_frames_cannot_be_described_hands_back_the_reason_not_a_frame():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Here is what the plan computed."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_refuses_to_describe,
    )
    await _session(runner)

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    after_plan = _texts(model.requests[-1])
    assert "its result cannot be drawn" in after_plan
    body = json.loads(
        after_plan.split("Only a frame listed under frames can be charted.\n", 1)[1].split("\n", 1)[
            0
        ]
    )
    assert body["frames"] == []
    assert body["frames_not_chartable"] == [
        {"step": "aggregate", "reason": "this step's data view could not be prepared"}
    ]
    assert "publish_not_configured" not in after_plan


async def test_an_amendment_redrafts_and_the_amended_plan_runs():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await _session(runner)

    first = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    amend = create_request_input_response(
        first.id, {"result": {"approved": None, "amendment": "only 2020"}}
    )
    second = _pause(await _run(runner, types.Content(role="user", parts=[amend])))
    assert second.args["payload"]["version"] == first.args["payload"]["version"] + 1
    approve = create_request_input_response(
        second.id, {"result": {"approved": second.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    assert "The approved plan ran." in _texts(model.requests[-1])


async def test_a_hand_back_is_still_in_view_after_a_later_pause():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service"),
        types.Part.from_text(text="Handed to the planner."),
        _call("ask_user", question="Which years?", options=["Since 2015", "Since 2020"]),
        types.Part.from_text(text="Asked."),
        types.Part.from_text(text="Kenya's debt service since 2015 could not be calculated."),
    )
    await _session(runner)

    pause = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    answer = create_request_input_response(pause.id, {"result": {"spec_id": "o1"}})
    await _run(runner, types.Content(role="user", parts=[answer]))

    resumed = _texts(model.requests[-1])
    assert "Kenya's debt service?" in resumed
    assert "The planner could not make a plan" in resumed
    assert "The user chose: Since 2015" in resumed
