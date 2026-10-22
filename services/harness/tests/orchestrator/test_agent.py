import hashlib
import json
from datetime import date
from typing import Any

import pytest
from google.adk.apps import App
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.graph.workflow import create_root_workflow
from harness.orchestrator_tools import build_orchestrator_tools
from harness.prefix_cache import PrefixCache, PrefixCachedGemini
from harness.scripted_plan import scripted_fetch_series_stats
from harness.turn_budget import BoundedGemini
from orchestrator.agent import AGENT_NAME, INSTRUCTION, create_orchestrator_agent

from ..data_mcp_definitions import DataMcpDefinitions
from ..orchestrator_fakes import (
    RefusingPlanner,
    ScriptLlm,
    no_executor,
    no_resolver,
    open_session,
    orchestrator_runner,
    run_turn,
)


@pytest.mark.parametrize("model_name", ["gemini-test", "Gemini:gemini-test"])
def test_the_orchestrator_runs_on_the_model_it_is_given_with_retries(model_name):
    agent = create_orchestrator_agent(model_name=model_name)

    assert agent.name == AGENT_NAME
    # Every call keeps to the request's budget.
    assert isinstance(agent.model, BoundedGemini)
    assert agent.model.model == "gemini-test"
    assert agent.model.retry_options.attempts == 3
    assert agent.include_contents == "default"


@pytest.mark.parametrize("model_name", ["gemini-test", "Gemini:gemini-test"])
def test_on_a_prefix_cache_the_orchestrator_keeps_its_retries(model_name):
    agent = create_orchestrator_agent(model_name=model_name, prefix_cache=PrefixCache())

    assert isinstance(agent.model, PrefixCachedGemini)
    assert agent.model.model == "gemini-test"
    assert agent.model.retry_options.attempts == 3


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("LOW", types.ThinkingLevel.LOW),
        ("MEDIUM", types.ThinkingLevel.MEDIUM),
        ("HIGH", types.ThinkingLevel.HIGH),
    ],
)
def test_the_thinking_level_reaches_the_generate_content_config(level, expected):
    agent = create_orchestrator_agent(model_name="gemini-test", thinking_level=level)

    config = agent.generate_content_config
    assert config is not None
    thinking = config.thinking_config
    assert thinking is not None
    assert thinking.thinking_level == expected
    assert thinking.include_thoughts is True


def test_an_agent_built_without_a_thinking_level_runs_at_low():
    agent = create_orchestrator_agent(model_name="gemini-test")

    assert agent.generate_content_config is not None
    assert agent.generate_content_config.thinking_config is not None
    assert agent.generate_content_config.thinking_config.thinking_level == types.ThinkingLevel.LOW


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


def test_the_instruction_says_what_the_canvas_digests_focus_card_is():
    """`focus_component_id` is how a card's "Ask follow up" reaches the
    model; unexplained, the model may answer about every card."""
    assert "focus_component_id" in INSTRUCTION


def test_the_instruction_lets_a_reply_about_dfo_state_the_figures_explain_dfo_gives():
    assert "stating only the figures it gives" in INSTRUCTION
    text = " ".join(INSTRUCTION.split())
    assert "Do not state a figure in a reply to a greeting or small talk" in text


def test_the_instruction_has_a_reply_state_only_the_figures_a_drawn_card_returns():
    """The reply states a drawn card's `reply_figures` and nothing else of
    it, the chart names the series the user asked a value of, an empty list
    gets what the card holds, and the conclusion comes from `holds`."""
    text = " ".join(INSTRUCTION.split())
    assert "state only the values its reply_figures list, each with its period" in text
    assert "Name in state_values_for only a series whose value the user asked to be told" in text
    assert "say what the card holds, its measure, places and years, and state no value" in text
    assert "Close with one conclusion drawn from holds" in text
    assert "compute nothing yourself" in text
    assert "state its last value and period from the card's facts" not in text


def test_the_instruction_tells_the_agent_to_state_the_years_an_event_period_reads_as():
    text = " ".join(INSTRUCTION.split())
    assert "names a period by an event or an administration" in text
    assert "say the data ends earlier and answer with what it holds" in text
    assert "ask a count or prices still open in one more ask_user call" in text
    assert "the reading search chose" not in text


def test_the_instruction_has_search_elect_the_measure_and_asks_with_its_alternatives():
    text = " ".join(INSTRUCTION.split())
    assert (
        "Put a word for how a flow is counted into `variable` only when the user said it, "
        "and let search elect the rest" in text
    )
    assert "ask about each with ask_user, one question per choice, all in one call" in text
    assert (
        "The count search chose is the first option of the count question, and the other counts "
        "the result names follow" in text
    )
    assert (
        "Over several years, prices get a question of their own, even when the result lists no "
        "`other_prices`. Its first option is the prices search read, in words and without a "
        "spec_id. Each `other_prices` entry follows, or constant prices when the result lists "
        "none" in text
    )
    assert (
        "When they pick prices the result does not name, search again with that unit on the "
        "part, and put the count they chose, if any, in `variable`" in text
    )
    assert (
        "A request that adds or combines measures from different sources, such as aid and debt, "
        "goes to plan straight after search, without ask_user. The planner asks how to combine "
        "them" in text
    )
    assert "in the user's words, except a count or prices, whose options are set out below" in text
    assert "takes one question a turn" not in text
    assert "gross or net" not in text


async def test_every_run_of_the_agent_in_the_workflow_reads_the_clocks_utc_date():
    model = ScriptLlm(script=[types.Part.from_text(text="Done.")])
    agent = create_orchestrator_agent(
        model_name="unused",
        tools=build_orchestrator_tools(resolver=no_resolver, executor=no_executor),
        today=lambda: date(2026, 10, 5),
    ).model_copy(update={"model": model})
    workflow = create_root_workflow(
        planner=RefusingPlanner(),
        orchestrator=agent,
        executor=no_executor,
        tool_surface=None,
        definition_surface=DataMcpDefinitions(),
        publisher=None,
        fetch_series_stats=scripted_fetch_series_stats,
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


async def test_a_stored_submit_analysis_call_replays_to_a_model_that_is_not_offered_the_tool():
    """A stored conversation can hold calls and answers of `submit_analysis`, a tool
    the orchestrator does not declare. ADK replays them into the request, and the
    turn runs on."""
    runner, model = orchestrator_runner(types.Part.from_text(text="Kenya's figure stands."))
    await open_session(runner)
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    stored_call = types.Part(
        function_call=types.FunctionCall(
            id="adk-stored-call",
            name="submit_analysis",
            args={"recipe": "growth", "spec_id": "spec_" + "2" * 32, "start_year": 2015},
        )
    )
    stored_answer = types.Part(
        function_response=types.FunctionResponse(
            id="adk-stored-call",
            name="submit_analysis",
            response={"outcome": "completed", "run": {"run_id": "run_01stored"}},
        )
    )
    stored = [
        ("user", "user", types.Part.from_text(text="Growth of life expectancy, 2015 to 2022")),
        (AGENT_NAME, "model", stored_call),
        (AGENT_NAME, "user", stored_answer),
        (AGENT_NAME, "model", types.Part.from_text(text="It grew by 3 per cent.")),
    ]
    for author, role, part in stored:
        await runner.session_service.append_event(
            session,
            Event(
                author=author,
                invocation_id="e-stored",
                branch=None if author == "user" else AGENT_NAME,
                content=types.Content(role=role, parts=[part]),
            ),
        )

    events = await run_turn(runner, types.UserContent("And Ghana?"))

    assert not [e for e in events if e.error_code]
    (request,) = model.requests
    replayed = [part for content in request.contents for part in content.parts or []]
    assert [p.function_call.name for p in replayed if p.function_call] == ["submit_analysis"]
    assert [p.function_response.name for p in replayed if p.function_response] == [
        "submit_analysis"
    ]
    declared = {
        declaration.name
        for tool in request.config.tools or []
        for declaration in tool.function_declarations or []
    }
    assert "plan" in declared
    assert "submit_analysis" not in declared
    answers = [e.content.parts[0].text for e in events if e.author == AGENT_NAME and e.content]
    assert answers[-1] == "Kenya's figure stands."
