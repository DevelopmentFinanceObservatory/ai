"""Tests for `harness.agents`: the planner built with its translator attached.

Construction, plus one run through a real `Runner` for the rules ADK only
enforces at run time. Whether the planner's model actually calls the translator
at the right moment is a semantic judgement, which is what `harness-pair-v1`
scores.

`test_the_planners_context_carries_no_relayed_translator_turn` drives one
plan-then-translate round for the model-facing guarantee this module's own
docstring states. It builds its seam over `test_planner_seam`'s scripted
models and fixtures, the same way `test_context_budget.py` does.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from compiler.surface import FixtureSurface, McpSurface, ToolCallResult
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from harness.agents import build_tool_surface, create_planner_with_translator
from harness.planner_seam import PlannerSeamError
from harness.settings import Settings
from planner.settings import AgentSettings

from .test_planner_seam import (
    _NO_DATA_SEARCH_RESPONSE,
    ScriptedPlannerLlm,
    _call_and_response_names,
    _gdp_plan,
    _prompt_text,
    _run_seam,
    _translator_seam,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate these tests from agent models set in the ambient environment."""
    for var in ("PLANNER_MODEL_NAME", "TRANSLATOR_MODEL_NAME"):
        monkeypatch.delenv(var, raising=False)


def _planner() -> LlmAgent:
    return create_planner_with_translator(settings=Settings(), surface=FixtureSurface(schemas=[]))


def test_creates_a_planner_carrying_the_translator_as_a_sub_agent() -> None:
    agent = _planner()

    assert agent.name == "dfo_planner"
    assert [sub_agent.name for sub_agent in agent.sub_agents] == ["dfo_translator"]


def test_the_translator_reaches_the_planners_model_as_a_tool() -> None:
    """`single_turn` is what turns the sub-agent into one more tool declaration."""
    agent = _planner()

    assert agent.sub_agents[0].mode == "single_turn"
    assert "dfo_translator" in {tool.name for tool in agent.tools}  # type: ignore[union-attr]


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


async def test_the_planner_reads_skills_through_the_surface_the_translator_gets() -> None:
    surface = FixtureSurface(
        schemas=[],
        responses={"get_skill": ToolCallResult(is_error=False, text="# DFO FAQ\n")},
    )
    agent = create_planner_with_translator(settings=Settings(), surface=surface)
    tools = {tool.name: tool for tool in agent.tools if isinstance(tool, FunctionTool)}

    answer = await tools["get_skill"].func(skill_id="dfo-faq")

    assert answer == "# DFO FAQ\n"
    assert 'get_skill("dfo-faq")' in str(agent.instruction)


def test_each_agent_runs_the_model_its_own_setting_pins() -> None:
    agent = create_planner_with_translator(
        settings=Settings(),
        agent_settings=AgentSettings(
            planner_model_name="gemini-planner", translator_model_name="gemini-translator"
        ),
        surface=FixtureSurface(schemas=[]),
    )

    assert agent.model == "gemini-planner"
    assert agent.sub_agents[0].model == "gemini-translator"


def test_defaults_to_the_pinned_models() -> None:
    """Covers the branch that reads `AgentSettings` when no settings are passed.

    Both fields default to the same model, so this says nothing about which
    field feeds which agent -- `test_each_agent_runs_the_model_its_own_setting_pins`
    is what holds that, with two distinguishable names.
    """
    agent = _planner()

    assert agent.model == "gemini-3.8-flash"
    assert agent.sub_agents[0].model == "gemini-3.8-flash"


def _unpaired(request: Any) -> list[str]:
    """Function calls or responses in `request.contents` with no counterpart.

    ADK strips `adk-`-prefixed ids from a request
    (`flows/llm_flows/contents._copy_content_for_request`), so the pairing is
    read the way the model reads it: in order, by name. A call opens a slot
    that the next response of the same name closes. Anything left open at the
    end is an orphan (a call with no response, or a response with no call),
    which is the malformed request this checks the branch fix does not create.
    """
    open_calls: dict[str, int] = {}
    orphans: list[str] = []
    for content in request.contents:
        for part in content.parts or []:
            if part.function_call is not None:
                name = part.function_call.name
                open_calls[name] = open_calls.get(name, 0) + 1
            elif part.function_response is not None:
                name = part.function_response.name
                if open_calls.get(name, 0) == 0:
                    orphans.append(f"functionResponse:{name} with no call")
                else:
                    open_calls[name] -= 1
    orphans.extend(
        f"functionCall:{name} with no response x{count}"
        for name, count in sorted(open_calls.items())
        if count
    )
    return orphans


async def test_the_planners_context_carries_no_relayed_translator_turn(
    decision: Any, requirement: Any, derived_requirement: Any
) -> None:
    """The translator's own turns stay out of the planner's request, the
    guarantee this module's docstring states.

    ADK publishes a `single_turn` sub-agent's events on a sub-branch of its
    parent's (`google.adk.tools.agent_tool._SingleTurnAgentTool.run_async`),
    and `PlannerSeam._propose` stamps the planner's own run with
    `_PLANNER_BRANCH`, so `google.adk.flows.llm_flows.contents.
    _is_event_belongs_to_branch` drops the translator's events. Without that
    stamp the planner runs unbranched, an unbranched event belongs to every
    branch, and `google.adk.flows.llm_flows._fencing.
    _present_other_agent_message` relays the translator's `translate` call and
    its result back to the planner as fenced `role="user"` text: 1,327 chars of
    the planner's largest request on this fixture, and 24,683 of a 56,407-char
    history on the reference trace. Removing the branch stamp fails this test
    with four relayed parts on the planner's third generation.

    The typed `functionResponse: dfo_translator` is asserted present as well.
    A fix that blinds the planner to its own translator call would be worse
    than the relay it removed.
    """
    seam = _translator_seam(responses={"search": _NO_DATA_SEARCH_RESPONSE})
    draft = _gdp_plan(decision, requirement, derived_requirement)
    planner_llm = ScriptedPlannerLlm(
        script=[
            [("dfo_translator", {"request": "find gdp for guatemala"})],
            "stop",
        ]
    )
    translator_llm = ScriptedPlannerLlm(script=[[("translate", {})], "no data"])
    seam._agent.model = planner_llm
    cast(LlmAgent, seam._agent.sub_agents[0]).model = translator_llm

    with pytest.raises(PlannerSeamError):
        await _run_seam(
            seam,
            question="gdp guatemala",
            envelope={},
            prior_plan=draft.model_dump(mode="json"),
        )

    # Every generation, not just the last: the relay first appears on the one
    # after the translator's own first event.
    assert len(planner_llm.captured_requests) >= 3
    for index, request in enumerate(planner_llm.captured_requests):
        relayed = [
            part.text
            for content in request.contents
            for part in (content.parts or [])
            if part.text and "<<<BEGIN_QUOTED_AGENT_CONTENT>>>" in part.text
        ]
        assert not relayed, (
            f"generation {index} carries {len(relayed)} relayed foreign-agent part(s) "
            f"({sum(len(text) for text in relayed)} chars): {relayed!r}"
        )

    last = planner_llm.captured_requests[-1]
    assert "FR:dfo_translator" in _call_and_response_names(last), (
        "the planner lost its own dfo_translator function response, so the fix "
        "blinded it to the call rather than dropping the relay"
    )
    assert "Question: gdp guatemala" in _prompt_text(last)

    # Every function call and response in every request still has its
    # counterpart. This is what proves the branch fix drops whole relayed
    # events rather than cutting a call from its response: the synthesized
    # `dfo_translator` function response carries the *planner's* branch
    # (`dfo_planner`, the branch the calling agent's event is enriched with;
    # only the translator's own nested events take its sub-branch), so it
    # survives alongside the call it answers. An orphaned call here is a
    # request a real model rejects, which would be worse than the relay.
    for index, request in enumerate(planner_llm.captured_requests):
        assert not _unpaired(request), (
            f"generation {index} has orphaned function parts: {_unpaired(request)}"
        )


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
