import hashlib
import json
from datetime import date
from typing import Any

import pytest
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.graph import GenerativeNodes, create_root_workflow
from harness.orchestrator_tools import build_orchestrator_tools
from harness.prefix_cache import PrefixCache, PrefixCachedGemini
from harness.scripted_plan import scripted_fetch_series_stats
from orchestrator.agent import AGENT_NAME, INSTRUCTION, create_orchestrator_agent

from ..test_orchestrator_path import RefusingPlanner, ScriptLlm, _no_executor, _no_resolver


@pytest.mark.parametrize("model_name", ["gemini-test", "Gemini:gemini-test"])
def test_the_orchestrator_runs_on_the_model_it_is_given_with_retries(model_name):
    agent = create_orchestrator_agent(model_name=model_name)

    assert agent.name == AGENT_NAME
    assert agent.model.model == "gemini-test"
    assert agent.model.retry_options.attempts == 3
    assert agent.include_contents == "default"


@pytest.mark.parametrize("model_name", ["gemini-test", "Gemini:gemini-test"])
def test_on_a_prefix_cache_the_orchestrator_keeps_its_retries(model_name):
    agent = create_orchestrator_agent(model_name=model_name, prefix_cache=PrefixCache())

    assert isinstance(agent.model, PrefixCachedGemini)
    assert agent.model.model == "gemini-test"
    assert agent.model.retry_options.attempts == 3


async def _first_request(scripted_llm: Any, **state: Any) -> Any:
    model = scripted_llm()
    agent = create_orchestrator_agent(
        model_name="unused", today=lambda: date(2026, 10, 5)
    ).model_copy(update={"model": model})
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="t", user_id="u", session_id="s", state=state)
    runner = Runner(agent=agent, app_name="t", session_service=sessions)
    async for _ in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part(text="and Ghana?")]),
    ):
        pass
    return model.captured_requests[0]


def _prefix_hash(request: Any) -> str:
    config = request.config
    prefix = [
        config.system_instruction,
        [tool.model_dump(mode="json") for tool in config.tools or []],
    ]
    return hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest()


async def test_the_static_prefix_is_the_same_whatever_the_canvas(scripted_llm: Any):
    kenya = json.dumps({"blocks": [{"title": "Debt service, Kenya"}]})
    ghana = json.dumps({"blocks": [{"title": "Debt service, Ghana"}]})
    history = json.dumps({"turns": [{"user": "Kenya's debt service"}]})

    first = await _first_request(scripted_llm, canvas_digest=kenya)
    second = await _first_request(scripted_llm, canvas_digest=ghana, conversation_digest=history)

    assert _prefix_hash(first) == _prefix_hash(second)
    assert first.contents[0].parts[0].text == (
        f"Today's date (UTC): 2026-10-05\n\nThe user's canvas: {kenya}"
    )
    assert second.contents[0].parts[0].text == (
        f"Today's date (UTC): 2026-10-05\n\nThe user's canvas: {ghana}"
        f"\n\nEarlier turns in this conversation: {history}"
    )
    assert [content.parts[0].text for content in first.contents[1:]] == ["and Ghana?"]


async def test_a_turn_with_no_digest_opens_on_the_date_and_the_question(scripted_llm: Any):
    request = await _first_request(scripted_llm)

    assert [content.parts[0].text for content in request.contents] == [
        "Today's date (UTC): 2026-10-05",
        "and Ghana?",
    ]


def test_the_instruction_lets_a_reply_about_dfo_state_the_figures_explain_dfo_gives():
    assert "stating only the figures it gives" in INSTRUCTION
    assert "Never state a\nfigure in a reply to a greeting or small talk" in INSTRUCTION


def test_the_instruction_tells_the_agent_to_state_the_years_an_event_period_reads_as():
    text = " ".join(INSTRUCTION.split())
    assert "names a period by an event or an administration" in text
    assert "say the data ends earlier and answer with what it holds" in text


async def test_every_run_of_the_agent_in_the_workflow_reads_the_clocks_utc_date():
    model = ScriptLlm(script=[types.Part.from_text(text="Done.")])
    agent = create_orchestrator_agent(
        model_name="unused",
        tools=build_orchestrator_tools(resolver=_no_resolver, executor=_no_executor),
        today=lambda: date(2026, 10, 5),
    ).model_copy(update={"model": model})
    workflow = create_root_workflow(
        resolver=_no_resolver,
        executor=_no_executor,
        tool_surface=None,
        generative=GenerativeNodes(planner=RefusingPlanner()),
        fetch_series_stats=scripted_fetch_series_stats,
        orchestrator=agent,
    )
    runner = Runner(
        app=App(name="t", root_agent=workflow), session_service=InMemorySessionService()
    )
    await runner.session_service.create_session(
        app_name="t", user_id="u", session_id="s", state={"surface": "studio"}
    )

    events = [
        event
        async for event in runner.run_async(
            user_id="u", session_id="s", new_message=types.UserContent("Aid since the election?")
        )
    ]

    assert not [e for e in events if e.error_code]
    assert model.requests[0].contents[0].parts[0].text.startswith("Today's date (UTC): 2026-10-05")
    assert "2026-10-05" not in model.requests[0].config.system_instruction
