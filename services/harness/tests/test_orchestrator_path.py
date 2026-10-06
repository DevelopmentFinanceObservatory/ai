"""The orchestrator engine's path through the graph, run by a real ADK runner
with a scripted model: answer, ask, plan, and what the agent reads after each."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import pytest
from analysis_mcp import digest as analysis_digest
from analysis_mcp.execution_identity import current_execution_identity
from common.materialisation import FrameRef
from compiler.surface import FixtureSurface, ToolCallResult
from data_mcp import get_data_digest
from google.adk.apps import App
from google.adk.events import Event
from google.adk.flows.llm_flows._fencing import OTHER_AGENT_CONTEXT_PREAMBLE, QUOTED_CONTENT_BEGIN
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness import graph, plan_display
from harness.graph import (
    APPROVE_INTERRUPT,
    CLARIFY_INTERRUPT,
    ORCHESTRATOR_BRIEF_KEY,
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    UNREADABLE_PLAN_NOTE,
    GenerativeNodes,
    _ask_status,
    _partial_result,
    _stopped_steps,
    create_root_workflow,
)
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_seam import PlannerSeam, PlannerSeamError
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.scripted_plan import scripted_fetch_series_stats
from harness.settings import Settings
from harness.studio_translate import _node_name
from harness.turn_records import PlanAsk, PlanRecord
from harness.wiring import handing_back_planner_failures
from orchestrator.agent import create_orchestrator_agent
from planner.plan import (
    ASKS_STATE_KEY,
    AskAnswer,
    Decline,
    Plan,
    UnsupportedOperation,
    load_asks,
)
from planner.settings import AgentSettings
from pydantic import Field

from .context_probe import RecordingLlm


class AskingPlanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` whose plan answers the brief's first ask with
    its final step and declines every other ask as `unsupported_ask`."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        plan = Plan.model_validate((await super().__call__(**kwargs))["plan"])
        asks = load_asks(kwargs["ctx"].state)
        final = plan.evidence_requirements[-1].id
        plan = plan.model_copy(
            update={
                "asks": asks,
                "answered": (AskAnswer(ask=0, outputs=(final,)),),
                "declined": tuple(
                    Decline(ask=ask, code=UnsupportedOperation.UNSUPPORTED_ASK, ask_ref=number)
                    for number, ask in enumerate(asks)
                    if number
                ),
            }
        )
        return {"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}


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

    def __init__(self, *, final: bool = False) -> None:
        self.questions: list[str] = []
        self.readings: list[Any] = []
        self.asks: list[tuple[str, ...]] = []
        self.final = final

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.questions.append(kwargs["question"])
        self.readings.append(kwargs.get("readings"))
        self.asks.append(load_asks(kwargs["ctx"].state))
        return {"refused": {"text": "the planner ran out of rounds", "final": self.final}}


async def _no_resolver(**kwargs: Any) -> dict[str, Any]:
    raise AssertionError("no search expected")


async def _no_executor(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    raise AssertionError(f"no tool call expected, got {tool}")


async def _describes(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    assert tool == "describe_frame", tool
    return {"outcome": "ok", "view": {"dimensions": [{"name": "observationAbout"}]}}


# The spec_id `harness.scripted_plan.scripted_plan` retrieves when no search pinned one.
_SCRIPTED_SPEC = "spec_" + "f" * 32


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


def _texts_of(content: types.Content) -> str:
    return "".join(p.text or "" for p in content.parts or [])


def _quotes_another_agent(request: Any) -> bool:
    """Whether ADK fenced any content of `request` as another agent's."""
    text = _texts(request)
    return QUOTED_CONTENT_BEGIN in text or OTHER_AGENT_CONTEXT_PREAMBLE in text


def _carries_an_interrupt(request: Any) -> bool:
    """Whether `request` holds a pause's `adk_request_input` call, as a call or quoted."""
    return "adk_request_input" in "\n".join(str(p) for c in request.contents for p in c.parts or [])


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
    assert not _carries_an_interrupt(model.requests[-1])
    assert not _quotes_another_agent(model.requests[-1])
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
        _call("plan", brief="Niger's Global Fund receipts as a share of health spending", asks=[]),
        types.Part.from_text(text="I have handed this to the planner."),
        types.Part.from_text(text="The share could not be calculated."),
        planner=planner,
    )
    searched = {"text": "Global Fund receipts", "status": "no_data", "refusal_code": "no_match"}
    earlier = {"text": "Niger health spending", "status": "definite", "spec_id": "spec_h"}
    await _session(
        runner,
        **{
            ORCHESTRATOR_SEARCHES_KEY: [searched],
            ORCHESTRATOR_PRIOR_LEDGER_KEY: {"readings": [earlier], "spec_ids": [], "warnings": {}},
        },
    )

    events = await _run(runner, types.UserContent("How much does Niger get from the Global Fund?"))

    assert planner.questions == ["Niger's Global Fund receipts as a share of health spending"]
    # An earlier turn's readings come first, then this turn's.
    assert planner.readings == [[earlier, searched]]
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
        _call("plan", brief="Kenya's aid as a share of GDP", asks=[]),
        types.Part.from_text(text="Handed."),
        _call("plan", brief="Kenya's CRS aid as a share of GDP", asks=[]),
        types.Part.from_text(text="Handed again."),
        _call("plan", brief="Kenya's aid as a share of GDP, any source", asks=[]),
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


@pytest.mark.parametrize("final", [False, True])
async def test_only_a_refusal_another_brief_could_avoid_offers_another_plan(final: bool):
    """A final refusal (an internal or repeated failure, or a question declined
    whole) fails the same way for any brief, so the plan call is not given back
    and the hand-back does not offer one."""
    runner, model = _runner(
        _call("plan", brief="Kenya's aid as a share of GDP", asks=[]),
        types.Part.from_text(text="Handed."),
        types.Part.from_text(text="Could not be calculated."),
        planner=RefusingPlanner(final=final),
    )
    await _session(runner)

    await _run(runner, types.UserContent("Kenya's aid as a share of GDP?"))

    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    calls = session.state[ORCHESTRATOR_CALLS_KEY]
    assert bool(calls.get("plan_given_back")) is not final
    assert ("planned once more" in _texts(model.requests[-1])) is not final


async def test_each_plan_call_holds_the_planner_to_its_own_asks_alone():
    """A plan call lists the asks for its own plan only: a later call with
    none leaves no earlier call's asks for the planner to account for."""
    planner = RefusingPlanner()
    asks = ["a chart of France's aid to Guatemala", "the total since 2010"]
    runner, _ = _runner(
        _call("plan", brief="France's aid to Guatemala since 2010", asks=asks),
        types.Part.from_text(text="Handed."),
        _call("plan", brief="France's bilateral aid to Guatemala", asks=[]),
        types.Part.from_text(text="Handed again."),
        types.Part.from_text(text="Could not be calculated."),
        planner=planner,
    )
    await _session(runner)

    await _run(runner, types.UserContent("France's aid to Guatemala since 2010?"))

    assert planner.asks == [tuple(asks), ()]
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session.state.get(ASKS_STATE_KEY) is None


async def test_a_planner_seam_failure_is_handed_back_when_wrapped_and_fatal_when_not():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt", asks=[]),
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

    bare, _ = _runner(_call("plan", brief="Kenya's debt", asks=[]), planner=FailingPlanner())
    await _session(bare)
    with pytest.raises(PlannerSeamError):
        await _run(bare, types.UserContent("Kenya's debt?"))


async def test_an_approved_plan_runs_and_its_final_frames_come_back_to_the_agent():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Charted."),
        types.Part.from_text(text="Uganda next."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await _session(runner)

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    assert approval.id == APPROVE_INTERRUPT
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": {"approved": revision}})
    events = await _run(runner, types.Content(role="user", parts=[approve]))

    request = model.requests[-1]
    after_plan = _texts(request)
    assert "The approved plan ran. Chart its frames and answer." in after_plan
    body = json.loads(after_plan.split("Chart its frames and answer.\n", 1)[1].split("\n", 1)[0])
    assert body["frames"] and all(f["frame"].startswith("f") for f in body["frames"])
    assert body["step_summaries"]
    assert f"Plan {revision[:12]} (approved)" in after_plan
    assert not _carries_an_interrupt(request)
    assert not _quotes_another_agent(request)
    (hand_back,) = [c for c in request.contents if "The approved plan ran" in _texts_of(c)]
    assert hand_back.role == "user"
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert set(session.state[ORCHESTRATOR_FRAMES_KEY]) == {f["frame"] for f in body["frames"]}
    assert not [e for e in events if e.error_code]

    follow_up = await _run(runner, types.UserContent("And Uganda?"))

    assert not [e for e in follow_up if e.error_code]
    assert "The approved plan ran" in _texts(model.requests[-1])
    assert not _carries_an_interrupt(model.requests[-1])


@pytest.mark.parametrize("describes", [True, False])
async def test_the_hand_back_reports_each_ask_with_its_output_status_or_decline(describes: bool):
    runner, model = _runner(
        _call(
            "plan",
            brief="Kenya's debt service since 2015, ranked against East Africa",
            asks=["Kenya's debt service each year", "a ranking of East African countries"],
        ),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Answered."),
        planner=AskingPlanner(),
        tool_surface=ScriptedSurface(),
        executor=_describes if describes else _refuses_to_describe,
    )
    await _session(runner)

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    after_plan = _texts(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    workflow = Plan.model_validate_json(session.state["plan"]).workflow
    assert workflow is not None
    (final_step,) = workflow.terminal_steps
    output = (
        {"step": final_step, "status": "data", "frame": body["frames"][0]["frame"]}
        if describes
        else {
            "step": final_step,
            "status": "not_chartable",
            "reason": body["frames_not_chartable"][0]["reason"],
        }
    )
    assert body["asks"] == [
        {"ask": "Kenya's debt service each year", "answered_by": [output]},
        {
            "ask": "a ranking of East African countries",
            "declined": plan_display.DECLINE_REASONS[UnsupportedOperation.UNSUPPORTED_ASK],
        },
    ]


class _BreakdownRefusingSurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` refuses the breakdown it was asked for,
    as dfo-apis refuses a code system it cannot read."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        if kwargs["tool"].name != "get_data":
            return await super().call_tool(**kwargs)
        return ToolCallResult(
            is_error=True,
            structured_content={
                "code": "capability_unavailable",
                "retry_mode": "do_not_retry",
                "message": "this source cannot be broken down by that code system",
                "fix": {"pointer": "/breakdown/code_system"},
            },
            text="refused",
        )


async def test_a_breakdown_refused_at_run_time_hands_back_a_failed_step_and_ask():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=["Kenya's debt service"]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It could not be computed."),
        planner=AskingPlanner(),
        tool_surface=_BreakdownRefusingSurface(),
        executor=_describes,
    )
    await _session(runner)

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    after_plan = _texts(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    assert body["failed_step"]["step_id"] == "retrieve"
    assert body["failed_step"]["code"] == "capability_unavailable"
    # The ask's output reads the refused retrieval, so it failed with it.
    failed = {"status": "failed", "failed_at": "retrieve", "code": "capability_unavailable"}
    assert body["asks"] == [
        {"ask": "Kenya's debt service", "answered_by": [{"step": "totals", **failed}]}
    ]
    # An output the failed step does not feed, and that never ran, was not run.
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    workflow = Plan.model_validate_json(session.state["plan"]).workflow
    stopped = _stopped_steps(workflow, "totals")
    assert stopped == {"totals"}
    upstream = PlanAsk(ask="the debt service series", outputs=("retrieve",))
    assert _ask_status(upstream, {}, {"step_id": "totals"}, stopped)["answered_by"] == [
        {"step": "retrieve", "status": "not_run"}
    ]


def test_an_ask_answered_in_part_hands_back_its_steps_and_the_part_not_computed() -> None:
    code = UnsupportedOperation.CALCULATION_OVER_COMBINED
    ask = PlanAsk(
        ask="bilateral, imputed and their total",
        outputs=("joined",),
        declined=code,
        not_computed="their total",
    )

    assert _ask_status(ask, {}, None, set()) == {
        "ask": "bilateral, imputed and their total",
        "answered_by": [{"step": "joined", "status": "not_run"}],
        "not_computed": {"part": "their total", "reason": plan_display.DECLINE_REASONS[code]},
    }


async def _refuses_to_describe(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    assert tool == "describe_frame", tool
    return {
        "outcome": "refusal",
        "refusal": {"code": "publish_not_configured", "retry_mode": "do_not_retry"},
    }


class RefusingGetDataSurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` answers the refusal `code` names."""

    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code

    async def call_tool(self, *, tool: Any, arguments: Any, on_progress: Any = None) -> Any:
        if tool.name != "get_data":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        refusal = {
            "code": self.code,
            "retry_mode": "do_not_retry",
            "fix": {},
            "message": "this deployment cannot serve the request",
        }
        return ToolCallResult(is_error=True, structured_content=refusal, text=refusal["message"])


@pytest.mark.parametrize("code", ["capability_unavailable", "member_not_resolvable"])
async def test_a_step_refused_for_a_missing_capability_closes_the_part_it_served(code: str):
    """A capability the platform lacks fails any brief and any fetch of the
    step's data, so the plan call is not given back and the step's spec_id
    is refused to `fetch`; another refusal leaves both open."""
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="That part is unavailable."),
        planner=ScriptedPlannerModel(),
        tool_surface=RefusingGetDataSurface(code),
        executor=_describes,
    )
    await _session(runner)

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    after_plan = _texts(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    assert body["failed_step"]["code"] == code
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    missing = code == "capability_unavailable"
    assert bool(session.state[ORCHESTRATOR_CALLS_KEY].get("plan_given_back")) is not missing
    assert (_SCRIPTED_SPEC in (session.state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {})) is missing


async def test_a_plan_whose_frames_cannot_be_described_hands_back_the_reason_not_a_frame():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
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
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
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


class _CountingSurface(ScriptedSurface):
    """`ScriptedSurface` that records the name of every tool a plan step calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        self.calls.append(kwargs["tool"].name)
        return await super().call_tool(**kwargs)


# Three plans awaiting approval as the release that listed members stored
# them, each beside the revision its card named. Their workflow's sources
# hold `stat_vars`, `stat_var_truncated` and `pinned`, which
# `planner.workflow.SelectedSource` no longer has.
_PLANS_WITH_MEMBER_LISTS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "plans_with_member_lists.json").read_text()
)
_UNREADABLE_PLAN = json.dumps(_PLANS_WITH_MEMBER_LISTS["aggregate"]["plan"])


def _stored_state(runner: Runner) -> dict[str, Any]:
    """The live state of the session `_session` opened."""
    service = cast(InMemorySessionService, runner.session_service)
    return service.sessions["t"]["u"]["s"].state


def _plan_warnings(caplog: pytest.LogCaptureFixture) -> list[tuple[object, ...]]:
    """The arguments of each warning `planner.plan` logged."""
    return [
        cast("tuple[object, ...]", record.args)
        for record in caplog.records
        if record.name == "planner.plan" and record.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(
    ("stored", "answer"),
    [
        ("aggregate", "approves"),
        ("rank", "approves"),
        ("declined", "approves"),
        ("aggregate", "amends"),
    ],
)
async def test_an_answer_to_a_plan_stored_with_member_lists_gets_a_new_plan(
    stored: str, answer: str, caplog: pytest.LogCaptureFixture
):
    """A plan awaiting approval across the release cannot be read. Approving
    it or amending it retrieves nothing: the question is planned from the
    start, the new card says why, and approving that one runs it."""
    surface = _CountingSurface()
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=_describes,
    )
    await _session(runner)
    first = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    held = _PLANS_WITH_MEMBER_LISTS[stored]
    _stored_state(runner)["plan"] = json.dumps(held["plan"])
    result = (
        {"approved": held["revision"]}
        if answer == "approves"
        else {"approved": None, "amendment": "only 2020"}
    )

    with caplog.at_level(logging.WARNING, logger="planner.plan"):
        second = _pause(
            await _run(
                runner,
                types.Content(
                    role="user",
                    parts=[create_request_input_response(first.id, {"result": result})],
                ),
            )
        )

    assert second.id == APPROVE_INTERRUPT
    assert second.args["payload"]["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert second.args["payload"]["version"] == 1
    assert surface.calls == []
    ((state_key, entries, fields),) = _plan_warnings(caplog)
    assert (state_key, entries) == ("plan", 1)
    assert "workflow.sources.stat_vars" in str(fields)
    assert held["plan"]["question"] not in caplog.text

    approve = create_request_input_response(
        second.id, {"result": {"approved": second.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    assert "get_data" in surface.calls
    assert "The approved plan ran." in _texts(model.requests[-1])


async def test_a_plan_call_drafts_over_an_earlier_plan_stored_with_member_lists():
    """A session's earlier plan call left a plan the release cannot read.
    The next plan call drafts its own plan as it would with none on record,
    and its card says nothing of the earlier one, which no card of this
    call showed."""
    runner, _ = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await _session(runner, plan=_UNREADABLE_PLAN)

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))

    assert approval.id == APPROVE_INTERRUPT
    assert "refusal_note" not in approval.args["payload"]
    assert Plan.model_validate_json(_stored_state(runner)["plan"]).version == 1


class _StoredWithMemberLists(Plan):
    """A plan that is stored as `_UNREADABLE_PLAN`, whatever it holds.

    Each node writes the plan it just built before the next reads it, so no
    answer and no tool call lands a stored shape between `approve` and
    `execute`, or between `execute` and `plan_done`. This puts one there.
    """

    def model_dump_json(self, **kwargs: Any) -> str:
        del kwargs
        return _UNREADABLE_PLAN


async def test_execute_plans_again_when_the_approved_plan_was_stored_with_member_lists(
    monkeypatch: pytest.MonkeyPatch,
):
    """`execute` runs nothing of a plan it cannot read and spends no replan
    on it: the question is planned from the start and the card says why."""
    approve_plan = Plan.approve
    monkeypatch.setattr(
        Plan,
        "approve",
        lambda self: _StoredWithMemberLists.model_validate(approve_plan(self).model_dump()),
    )
    surface = _CountingSurface()
    runner, _ = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=_describes,
    )
    await _session(runner)
    first = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        first.id, {"result": {"approved": first.args["payload"]["revision"]}}
    )

    second = _pause(await _run(runner, types.Content(role="user", parts=[approve])))

    assert second.id == APPROVE_INTERRUPT
    assert second.args["payload"]["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert surface.calls == []
    assert not _stored_state(runner).get("replans")


async def test_the_hand_back_carries_no_record_of_a_plan_stored_with_member_lists(
    monkeypatch: pytest.MonkeyPatch,
):
    """`plan_done` hands back the frames the run produced as it does when no
    plan is on record: without the plan record, and with the slot cleared."""
    run_plan = graph.execute_approved_plan

    async def stored_unreadable(plan: Plan, **kwargs: Any) -> tuple[Plan, Any]:
        finished, outcome = await run_plan(plan, **kwargs)
        return _StoredWithMemberLists.model_validate(finished.model_dump()), outcome

    monkeypatch.setattr(graph, "execute_approved_plan", stored_unreadable)
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await _session(runner)
    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": {"approved": revision}})

    events = await _run(runner, types.Content(role="user", parts=[approve]))

    handed_back = _texts(model.requests[-1])
    assert "The approved plan ran. Chart its frames and answer." in handed_back
    assert f"Plan {revision[:12]}" not in handed_back
    assert not [e for e in events if e.error_code]
    state = _stored_state(runner)
    assert state["plan"] is None
    assert not state.get(ORCHESTRATOR_PLAN_RECORDS_KEY)


async def test_approving_a_superseded_revision_retrieves_nothing_and_proposes_again():
    surface = _CountingSurface()
    runner, _ = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=_describes,
    )
    await _session(runner)

    first = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    amend = create_request_input_response(
        first.id, {"result": {"approved": None, "amendment": "only 2020"}}
    )
    second = _pause(await _run(runner, types.Content(role="user", parts=[amend])))
    stale = create_request_input_response(
        second.id, {"result": {"approved": first.args["payload"]["revision"]}}
    )
    third = _pause(await _run(runner, types.Content(role="user", parts=[stale])))

    assert third.id == APPROVE_INTERRUPT
    assert third.args["payload"]["version"] == second.args["payload"]["version"] + 1
    assert "get_data" not in surface.calls


async def test_a_hand_back_is_still_in_view_after_a_later_pause():
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service", asks=[]),
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


async def test_a_second_plan_call_runs_a_planner_that_never_sees_the_first_hand_back():
    seam = PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
        agent_settings=AgentSettings(thinking_sample_rate=0.0),
    )
    planner_llm = RecordingLlm(script=[], agent_name="planner", scenario="replan")
    seam._agent.model = planner_llm
    runner, model = _runner(
        _call("plan", brief="Kenya's debt, first brief", asks=[]),
        types.Part.from_text(text="Handed."),
        _call("plan", brief="Kenya's debt, second brief", asks=[]),
        types.Part.from_text(text="Handed again."),
        types.Part.from_text(text="Could not be calculated."),
        planner=handing_back_planner_failures(seam),
    )
    await _session(runner)

    events = await _run(runner, types.UserContent("Kenya's debt?"))

    hand_back = "The planner could not make a plan"
    second_run = [r for r in planner_llm.captured if "second brief" in _texts(r)]
    assert second_run
    assert not [r for r in second_run if hand_back in _texts(r)]
    assert hand_back in _texts(model.requests[-1])
    assert not _quotes_another_agent(model.requests[-1])
    # The hand-back's branch stays on the hand-back: the graph's own events
    # reach the same agents after each hand-back as before the first.
    graph_events = [e for e in events if e.author == "dfo_analyst_graph"]
    assert graph_events and {e.branch for e in graph_events} == {None}
    assert not [e for e in events if e.error_code]


def _rows_led_by_a_wide_step(step_rows: Any) -> Any:
    """`plan_display.step_rows` with a first row whose arguments run past
    4,000 characters, and a reading on every retrieval after it."""

    def rows(plan: Plan) -> list[dict[str, object]]:
        wide = {
            "id": "wide",
            "label": "Fetch every member",
            "args": {
                "spec_id": "spec_wide",
                "resolved_stat_var_dcids": [f"Wide_StatVar_{n:04d}" for n in range(300)],
            },
        }
        read = [
            {**row, "args": {**cast("dict[str, Any]", row["args"]), "stat_vars": "sum"}}
            if "spec_id" in cast("dict[str, Any]", row["args"])
            else row
            for row in step_rows(plan)
        ]
        return [wide, *read]

    return rows


@pytest.mark.parametrize(
    ("prior_spec_ids", "origin"),
    [([], "translator"), ([_SCRIPTED_SPEC], "search")],
    ids=["found_by_the_translator", "searched_in_an_earlier_turn"],
)
async def test_the_plan_record_and_every_reading_reach_the_agent_past_wide_step_arguments(
    monkeypatch: pytest.MonkeyPatch, prior_spec_ids: list[str], origin: str
):
    """A source an earlier turn's search returned is recorded as searched, so
    its block in that turn's record stays the one a later turn pins from."""
    monkeypatch.setattr(plan_display, "step_rows", _rows_led_by_a_wide_step(plan_display.step_rows))
    runner, model = _runner(
        _call("plan", brief="Kenya's debt service since 2015", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    prior = {"readings": [], "spec_ids": prior_spec_ids, "warnings": {}}
    await _session(runner, **{ORCHESTRATOR_PRIOR_LEDGER_KEY: prior})

    approval = _pause(await _run(runner, types.UserContent("Kenya's debt service?")))
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": {"approved": revision}})
    events = await _run(runner, types.Content(role="user", parts=[approve]))

    after_plan = _texts(model.requests[-1])
    assert f"Plan {revision[:12]} (approved)" in after_plan
    assert f"({_SCRIPTED_SPEC}, sum)" in after_plan
    assert "Wide_StatVar_" not in after_plan
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session.state[ORCHESTRATOR_READINGS_KEY] == {_SCRIPTED_SPEC: "sum"}
    (stored,) = session.state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    record = PlanRecord.model_validate(stored)
    assert record.revision == revision
    assert [source.origin for source in record.sources] == [origin]
    assert not [e for e in events if e.error_code]


class _NoisySurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` digest holds the plumbing lines data-mcp
    and analysis-mcp render, each from its producer's own renderer."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        result = await super().call_tool(**kwargs)
        if kwargs["tool"].name != "get_data":
            return result
        text = "\n\n".join(
            [
                "Retrieved life expectancy for Kenya.",
                get_data_digest._frame_reference_line(
                    FrameRef(spec_id="spec_" + "1" * 32, content_version="cv_" + "1" * 32)
                ),
                "Source: World Bank.",
                get_data_digest._operation_poll_line(
                    get_data_digest.OperationPollNote(
                        operation_id="op_123", poll_attempts=2, elapsed_s=31.0
                    )
                ),
                analysis_digest._identity_block(
                    op_spec_hash="ops_" + "1" * 32,
                    content_version="ocv_" + "1" * 32,
                    execution_identity=current_execution_identity(),
                ),
                "Coverage 1960-2023.",
            ]
        )
        return result.model_copy(update={"text": text})


async def test_step_summaries_reach_the_agent_without_frame_references_or_operation_notes():
    runner, model = _runner(
        _call("plan", brief="Kenya's life expectancy", asks=[]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=_NoisySurface(),
        executor=_describes,
    )
    await _session(runner)

    approval = _pause(await _run(runner, types.UserContent("Kenya's life expectancy?")))
    approve = create_request_input_response(
        approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    after_plan = _texts(model.requests[-1])
    body = json.loads(after_plan.split("Chart its frames and answer.\n", 1)[1].split("\n", 1)[0])
    assert (
        "Retrieved life expectancy for Kenya.\n\nSource: World Bank.\n\nCoverage 1960-2023."
        in body["step_summaries"]
    )


class TwoAsksPlanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` whose plan answers the brief's first ask with the
    retrieval and its second with the aggregate that reads it, so the
    retrieval is a terminal step another step reads, declining `declined`.
    Records each refusal it is handed."""

    def __init__(self, declined: tuple[Decline, ...] = ()) -> None:
        self.declined = declined
        self.refusals: list[dict[str, Any] | None] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.refusals.append(kwargs.get("refusal"))
        plan = Plan.model_validate((await super().__call__(**kwargs))["plan"])
        assert plan.translation is not None and plan.workflow is not None
        series, mean = (requirement.id for requirement in plan.evidence_requirements)
        terminal = tuple(step.id for step in plan.workflow.steps)
        plan = plan.model_copy(
            update={
                "asks": load_asks(kwargs["ctx"].state),
                "answered": (
                    AskAnswer(ask=0, outputs=(series,)),
                    AskAnswer(ask=1, outputs=(mean,)),
                ),
                "declined": self.declined,
                "translation": plan.translation.model_copy(update={"terminal": terminal}),
                "workflow": plan.workflow.model_copy(update={"terminal_steps": terminal}),
            }
        )
        return {"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}


class _AggregateRefusingSurface(ScriptedSurface):
    """`ScriptedSurface` whose `aggregate` refuses with a code no replan reaches."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        if kwargs["tool"].name != "aggregate":
            return await super().call_tool(**kwargs)
        refusal = {
            "code": "growth_gap_year",
            "retry_mode": "do_not_retry",
            "message": "no entity reports both endpoint years",
            "fix": {},
        }
        return ToolCallResult(is_error=True, structured_content=refusal, text="refused")


class _EmptyAggregateSurface(ScriptedSurface):
    """`ScriptedSurface` whose `aggregate` matches no rows, a stop the graph replans."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        result = await super().call_tool(**kwargs)
        if kwargs["tool"].name != "aggregate":
            return result
        return result.model_copy(
            update={"structured_content": {**(result.structured_content or {}), "rows": 0}}
        )


_TWO_ASKS = ["Kenya's life expectancy each year", "the mean across the period"]
# The first ask answered in part: its history computed, its forecast declined.
_FORECAST = Decline(
    ask="its forecast to 2030", code=UnsupportedOperation.NO_OPERATION_COVERS, ask_ref=0
)
_IN_PART = pytest.mark.parametrize("declined", [(), (_FORECAST,)], ids=["whole", "in_part"])


def _first_ask(declined: tuple[Decline, ...]) -> tuple[dict[str, Any], str]:
    """The first ask's `not_computed` entry under `declined`, and the ask as
    the charted sentence names it."""
    if not declined:
        return {}, json.dumps(_TWO_ASKS[0])
    reason = plan_display.DECLINE_REASONS[_FORECAST.code]
    return (
        {"not_computed": {"part": _FORECAST.ask, "reason": reason}},
        f"{json.dumps(_TWO_ASKS[0])} except {json.dumps(_FORECAST.ask)}, "
        f"not computed because {reason}",
    )


async def _approved_two_asks(
    surface: ScriptedSurface, planner: TwoAsksPlanner | None = None
) -> tuple[ScriptLlm, TwoAsksPlanner]:
    """Run the two-asks brief, approving every plan the planner drafts."""
    planner = planner or TwoAsksPlanner()
    runner, model = _runner(
        _call("plan", brief="Kenya's life expectancy and its mean", asks=_TWO_ASKS),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="Answered."),
        planner=planner,
        tool_surface=surface,
        executor=_describes,
    )
    await _session(runner)
    events = await _run(runner, types.UserContent("Kenya's life expectancy?"))
    while any(c.name == "adk_request_input" for e in events for c in e.get_function_calls()):
        approval = _pause(events)
        approve = create_request_input_response(
            approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
        )
        events = await _run(runner, types.Content(role="user", parts=[approve]))
    return model, planner


def _hand_back(model: ScriptLlm) -> tuple[str, dict[str, Any]]:
    text = _texts(model.requests[-1])
    return text, json.loads(next(line for line in text.split("\n") if line.startswith('{"frames"')))


async def test_a_terminal_step_another_step_reads_comes_back_on_an_accepted_plan():
    model, _ = await _approved_two_asks(ScriptedSurface())

    _, body = _hand_back(model)
    series, mean = (frame["frame"] for frame in body["frames"])
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
        },
        {"ask": _TWO_ASKS[1], "answered_by": [{"step": "totals", "status": "data", "frame": mean}]},
    ]


@_IN_PART
async def test_a_failed_plan_hands_back_the_terminal_frames_it_finished(
    declined: tuple[Decline, ...],
):
    model, _ = await _approved_two_asks(_AggregateRefusingSurface(), TwoAsksPlanner(declined))

    text, body = _hand_back(model)
    (series,) = (frame["frame"] for frame in body["frames"])
    failed = {"status": "failed", "failed_at": "totals", "code": "growth_gap_year"}
    not_computed, charted = _first_ask(declined)
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
            **not_computed,
        },
        {"ask": _TWO_ASKS[1], "answered_by": [{"step": "totals", **failed}]},
    ]
    assert f"can be charted: {charted}." in text
    assert f"Not computed: {json.dumps(_TWO_ASKS[1])}." in text


async def test_a_failed_plan_hands_back_no_step_that_is_only_read():
    """The retrieval finished, but only the refused aggregate answers an ask."""
    runner, model = _runner(
        _call("plan", brief="Kenya's life expectancy", asks=["Kenya's mean life expectancy"]),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It could not be computed."),
        planner=AskingPlanner(),
        tool_surface=_AggregateRefusingSurface(),
        executor=_describes,
    )
    await _session(runner)
    approval = _pause(await _run(runner, types.UserContent("Kenya's life expectancy?")))
    approve = create_request_input_response(
        approval.id, {"result": {"approved": approval.args["payload"]["revision"]}}
    )
    await _run(runner, types.Content(role="user", parts=[approve]))

    _, body = _hand_back(model)
    assert body["frames"] == []
    assert body["failed_step"]["step_id"] == "totals"


async def test_a_replan_names_the_asks_the_failed_run_computed():
    _, planner = await _approved_two_asks(_EmptyAggregateSurface())

    refusal = planner.refusals[-1]
    assert refusal is not None
    assert refusal["computed_asks"] == [0]


class _RefusingReplanner(TwoAsksPlanner):
    """`TwoAsksPlanner` that refuses to draft again once a run has stopped."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("refusal") is None:
            return await super().__call__(**kwargs)
        self.refusals.append(kwargs["refusal"])
        return {"refused": {"text": "no plan avoids the empty mean", "final": False}}


class _RetrievalRefusingOnReplanSurface(_EmptyAggregateSurface):
    """`_EmptyAggregateSurface` whose second `get_data` call refuses, so the
    replanned run stops before any ask is computed."""

    def __init__(self) -> None:
        super().__init__()
        self.retrievals = 0

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        if kwargs["tool"].name == "get_data":
            self.retrievals += 1
            if self.retrievals > 1:
                refusal = {
                    "code": "growth_gap_year",
                    "retry_mode": "do_not_retry",
                    "message": "refused on the replan",
                    "fix": {},
                }
                return ToolCallResult(is_error=True, structured_content=refusal, text="refused")
        return await super().call_tool(**kwargs)


@_IN_PART
async def test_a_planner_refusal_after_a_stopped_run_hands_back_the_asks_it_computed(
    declined: tuple[Decline, ...],
):
    model, planner = await _approved_two_asks(
        _EmptyAggregateSurface(), _RefusingReplanner(declined)
    )

    text, body = _hand_back(model)
    (series,) = (frame["frame"] for frame in body["frames"])
    not_computed, charted = _first_ask(declined)
    assert planner.refusals[-1] is not None
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
            **not_computed,
            "earlier_plan": True,
        }
    ]
    assert "The planner could not make a plan: no plan avoids the empty mean" in text
    assert f"can be charted: {charted}." in text


@_IN_PART
async def test_a_replanned_run_that_stops_earlier_keeps_the_asks_the_first_run_computed(
    declined: tuple[Decline, ...],
):
    model, _ = await _approved_two_asks(
        _RetrievalRefusingOnReplanSurface(), TwoAsksPlanner(declined)
    )

    text, body = _hand_back(model)
    (series,) = (frame["frame"] for frame in body["frames"])
    failed = {"status": "failed", "failed_at": "retrieve", "code": "growth_gap_year"}
    not_computed, charted = _first_ask(declined)
    assert body["failed_step"]["step_id"] == "retrieve"
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
            **not_computed,
            "earlier_plan": True,
        },
        {"ask": _TWO_ASKS[1], "answered_by": [{"step": "totals", **failed}]},
    ]
    assert f"can be charted: {charted}." in text
    assert f"Not computed: {json.dumps(_TWO_ASKS[1])}." in text


@pytest.mark.parametrize(
    ("status", "charted"), [({"status": "data"}, True), ({"status": "empty"}, False)]
)
def test_a_partial_result_offers_to_chart_only_frames_that_hold_series(
    status: dict[str, str], charted: bool
):
    asks = [{"ask": "the mean", "answered_by": [{"step": "totals", **status, "frame": "f1"}]}]

    text = _partial_result(asks, frames=True)

    assert ('can be charted: "the mean".' in text) is charted
    assert (
        'holds no series or cannot be drawn, as each step\'s status says: "the mean".' in text
    ) is not charted
