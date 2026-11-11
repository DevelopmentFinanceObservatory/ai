"""Tests for the planner the seam builds with translation attached, and `harness.agents`.

Construction, plus one run through a real `Runner` for the rules ADK only
enforces at run time. Whether the planner's model acts on the `translation`
verdict `propose_plan` returns is a semantic judgement, which no test here covers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import translator.tool as translator_tool
from compiler.surface import FixtureSurface, McpSurface, ToolCallResult, ToolSurface
from data_mcp import resolution
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from harness.agents import build_tool_surface
from harness.planner_seam import PlannerSeam
from harness.settings import Settings
from planner.agent import SpecTool
from planner.settings import AgentSettings
from translator import discovery

from .planner_tools import call_planner_tool


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate these tests from agent models set in the ambient environment."""
    monkeypatch.delenv("PLANNER_MODEL_NAME", raising=False)


def _planner(
    surface: ToolSurface | None = None, agent_settings: AgentSettings | None = None
) -> LlmAgent:
    """The ADK planner the seam serves."""
    seam = PlannerSeam(
        settings=Settings(planner_runner="adk"),
        agent_settings=agent_settings,
        surface=surface if surface is not None else FixtureSurface(schemas=[]),
    )
    assert seam._agent is not None
    return seam._agent


async def test_the_served_translation_resolves_through_the_graph_and_reads_the_newest_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planner's translation takes the real resolve door and newest-year read, the
    two `evals/planner_entry` replaces through `build_translate`'s keywords."""
    built: dict[str, Any] = {}
    real_core = translator_tool.build_translator_core

    def spy(**kwargs: Any) -> Any:
        built.update(kwargs)
        return real_core(**kwargs)

    monkeypatch.setattr(translator_tool, "build_translator_core", spy)

    _planner()

    assert built["resolve_entities"] is discovery.graph_resolve
    assert built["anchor_read"] is translator_tool.read_within_entry
    # `graph_resolve` reads `data_mcp.resolution` when called, so the conftest's
    # stand-in for the graph answers here.
    outcomes = await discovery.graph_resolve({"names": ["Kenya"]})
    assert outcomes == {"names": {"Kenya": resolution.NotFound(near_matches=[])}}


def test_creates_a_planner_whose_propose_plan_translates_without_a_tool_of_its_own() -> None:
    """Translation runs inside `propose_plan`, so the model is declared no `translate` tool."""
    agent = _planner()

    assert agent.name == "dfo_planner"
    assert agent.sub_agents == []
    names = {tool.name for tool in agent.tools}  # type: ignore[union-attr]
    assert "propose_plan" in names
    assert "translate" not in names


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


async def test_the_planner_reads_skills_through_the_surface_translation_gets() -> None:
    surface = FixtureSurface(
        schemas=[],
        responses={"get_skill": ToolCallResult(is_error=False, text="# DFO FAQ\n")},
    )
    agent = _planner(surface)
    tools = {tool.name: tool for tool in agent.tools if isinstance(tool, SpecTool)}

    answer = await call_planner_tool(
        tools["get_skill"],
        skill_id="dfo-faq",
        tool_context=SimpleNamespace(
            state={}, session=SimpleNamespace(events=[]), function_call_id=None
        ),
    )

    assert answer == "# DFO FAQ\n"
    assert "`get_skill`" in str(agent.instruction)


def test_the_planner_runs_the_model_its_setting_pins() -> None:
    agent = _planner(agent_settings=AgentSettings(planner_model_name="gemini-planner"))

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
