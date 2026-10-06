"""Tests for `harness.agents`: the planner built with its `translate` tool.

Construction, plus one run through a real `Runner` for the rules ADK only
enforces at run time. Whether the planner's model actually calls `translate`
at the right moment is a semantic judgement, which is what `harness-pair-v1`
scores.
"""

from __future__ import annotations

from typing import Any

import pytest
from compiler.surface import FixtureSurface, McpSurface, ToolCallResult
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from harness.agents import build_tool_surface, create_planner_with_translator
from harness.settings import Settings
from planner.settings import AgentSettings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate these tests from agent models set in the ambient environment."""
    monkeypatch.delenv("PLANNER_MODEL_NAME", raising=False)


def _planner() -> LlmAgent:
    return create_planner_with_translator(settings=Settings(), surface=FixtureSurface(schemas=[]))


def test_creates_a_planner_carrying_translate_as_its_own_tool() -> None:
    """Translation is a function tool on the planner, not a second agent."""
    agent = _planner()

    assert agent.name == "dfo_planner"
    assert agent.sub_agents == []
    assert "translate" in {tool.name for tool in agent.tools}  # type: ignore[union-attr]


async def test_the_pair_runs_as_a_runner_root(
    scripted_llm: type[Any], run_to_completion: Any
) -> None:
    """ADK rejects a root agent that is not a chat agent, and only a run finds out.

    That check lives inside `Runner.run_async`, not in the constructor, so
    building a `Runner` around a wrapped parent raises nothing and asserting on
    `agent.mode` only restates ADK's rule. Running the pair exercises it: had
    the composition wrapped the planner itself -- `mode="single_turn"` or
    `"task"` -- this would raise `ValueError`. The scripted model answers with
    text and calls nothing, so the run reaches the end with no tool surface
    behind it.
    """
    agent = _planner()
    agent.model = scripted_llm()

    events = await run_to_completion(
        agent=agent, session_service=InMemorySessionService(), app_name="pair-tests"
    )

    assert events


async def test_the_planner_reads_skills_through_the_surface_translate_gets() -> None:
    surface = FixtureSurface(
        schemas=[],
        responses={"get_skill": ToolCallResult(is_error=False, text="# DFO FAQ\n")},
    )
    agent = create_planner_with_translator(settings=Settings(), surface=surface)
    tools = {tool.name: tool for tool in agent.tools if isinstance(tool, FunctionTool)}

    answer = await tools["get_skill"].func(skill_id="dfo-faq")

    assert answer == "# DFO FAQ\n"
    assert "`get_skill`" in str(agent.instruction)


def test_the_planner_runs_the_model_its_setting_pins() -> None:
    agent = create_planner_with_translator(
        settings=Settings(),
        agent_settings=AgentSettings(planner_model_name="gemini-planner"),
        surface=FixtureSurface(schemas=[]),
    )

    assert isinstance(agent.model, Gemini)
    assert agent.model.model == "gemini-planner"


def test_defaults_to_the_pinned_model() -> None:
    """Covers the branch that reads `AgentSettings` when no settings are passed."""
    model = _planner().model
    assert isinstance(model, Gemini)
    assert model.model == "gemini-3.8-flash"


def test_build_tool_surface_covers_data_and_analysis_mcp_by_name() -> None:
    """Both, because a plan retrieves before it computes. Building it opens no connection."""
    settings = Settings(
        data_mcp_url="https://data-mcp.internal",
        analysis_mcp_url="https://analysis-mcp.internal",
        mcp_api_token="test-token",
    )

    surface = build_tool_surface(settings=settings)

    assert isinstance(surface, McpSurface)
    # Reads the private mapping because `list_tools()` is the only public
    # alternative and it connects to each server, which is the opposite of
    # what this asserts.
    assert surface._sources == {
        "data-mcp": "https://data-mcp.internal",
        "analysis-mcp": "https://analysis-mcp.internal",
    }
