"""Tests for the planner the seam builds with translation attached, and `harness.agents`.

Each planner test reads the planner from the model's side, through the first request
that a scripted google-genai client (`ScriptedGenai`) receives from it. Whether the
planner's model acts on the `translation` verdict `propose_plan` returns is a
semantic judgement, which no test here covers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import translator.tool as translator_tool
from compiler.surface import FixtureSurface, McpSurface, ToolCallResult, ToolSurface
from data_mcp import resolution
from harness.agents import build_tool_surface
from harness.planner_adapter import PlannerAdapter
from harness.settings import Settings
from planner.settings import AgentSettings
from translator import discovery

from .planner_tools import call_planner_tool
from .scripted_genai import Request, ScriptedGenai, reply, text_part
from .test_planner_adapter import _AdapterHarness


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate these tests from agent models set in the ambient environment."""
    monkeypatch.delenv("PLANNER_MODEL_NAME", raising=False)


def _adapter(
    scripted: ScriptedGenai,
    surface: ToolSurface | None = None,
    agent_settings: AgentSettings | None = None,
) -> PlannerAdapter:
    """The seam over `surface` (an empty fixture surface by default), its model
    calls answered by `scripted`."""
    return PlannerAdapter(
        settings=Settings(),
        agent_settings=agent_settings,
        surface=surface if surface is not None else FixtureSurface(schemas=[]),
        client=scripted.client,
    )


async def _first_request(
    adapter: PlannerAdapter, scripted: ScriptedGenai, harness: _AdapterHarness | None = None
) -> Request:
    """The first request the planner `adapter` makes of its model, answered with
    text, which ends the entry."""
    scripted.replies.append(reply(text_part("No plan.")))
    await (harness or _AdapterHarness()).run(adapter, question="kenya debt", readings=[])
    return scripted.requests[0]


def _declared(request: Request) -> list[str]:
    """The names of the tools `request` declares, in order."""
    return [
        declaration.name or ""
        for tool in request.generate_config.tools or []
        for declaration in tool.function_declarations or []
    ]


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

    _adapter(ScriptedGenai([]))

    assert built["resolve_entities"] is discovery.graph_resolve
    assert built["anchor_read"] is translator_tool.read_within_entry
    # `graph_resolve` reads `data_mcp.resolution` when called, so the conftest's
    # stand-in for the graph answers here.
    outcomes = await discovery.graph_resolve({"names": ["Kenya"]})
    assert outcomes == {"names": {"Kenya": resolution.NotFound(near_matches=[])}}


async def test_creates_a_planner_whose_propose_plan_translates_without_a_tool_of_its_own() -> None:
    """Translation runs inside `propose_plan`, so the model is declared no `translate` tool."""
    scripted = ScriptedGenai([])
    harness = _AdapterHarness()

    names = _declared(await _first_request(_adapter(scripted), scripted, harness))

    assert "propose_plan" in names
    assert "translate" not in names
    assert "dfo_planner" in {event.author for event in await harness.events()}


async def test_the_planner_reads_skills_through_the_surface_translation_gets() -> None:
    surface = FixtureSurface(
        schemas=[],
        responses={"get_skill": ToolCallResult(is_error=False, text="# DFO FAQ\n")},
    )
    scripted = ScriptedGenai([])
    adapter = _adapter(scripted, surface)
    request = await _first_request(adapter, scripted)
    tools = {spec.name: spec for spec in adapter._specs}

    answer = await call_planner_tool(
        tools["get_skill"],
        skill_id="dfo-faq",
        tool_context=SimpleNamespace(
            state={}, session=SimpleNamespace(events=[]), function_call_id=None
        ),
    )

    assert answer == "# DFO FAQ\n"
    assert "get_skill" in _declared(request)
    assert "`get_skill`" in str(request.generate_config.system_instruction)


async def test_the_planner_runs_the_model_its_setting_pins() -> None:
    scripted = ScriptedGenai([])
    adapter = _adapter(scripted, agent_settings=AgentSettings(planner_model_name="gemini-planner"))

    assert (await _first_request(adapter, scripted)).model == "gemini-planner"


async def test_defaults_to_the_pinned_model() -> None:
    """Covers the branch that reads `AgentSettings` when no settings are passed."""
    scripted = ScriptedGenai([])

    assert (await _first_request(_adapter(scripted), scripted)).model == "gemini-3.8-flash"


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
