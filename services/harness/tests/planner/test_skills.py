"""Tests for `planner.skills`: the planner's `get_skill` and `get_skill_file`.

Each tool calls data-mcp's tool of the same name through the surface it was
built over and answers with data-mcp's text, a refusal's included.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest
from compiler.surface import (
    ProgressCallback,
    ToolCallResult,
    ToolSurfaceUnavailableError,
)
from google.adk.tools.function_tool import FunctionTool
from planner.agent import create_planner_agent
from planner.skills import build_skill_tools
from planner.workflow import ToolId


class _RecordingSurface:
    """A `ToolSurface` that records each call and answers with one fixed result."""

    def __init__(self, answer: ToolCallResult | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[ToolId, dict[str, object]]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del on_progress
        self.calls.append((tool, dict(arguments)))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _tools(surface: _RecordingSurface) -> dict[str, Any]:
    return {tool.name: tool.func for tool in build_skill_tools(surface)}


_SKILL_TEXT = "---\nname: oda-concepts\n---\n\n# ODA concepts\n"


def test_a_planner_built_with_a_surface_carries_the_skill_tools_and_names_dfo_faq() -> None:
    agent = create_planner_agent(
        model_name="m", surface=_RecordingSurface(ToolCallResult(is_error=False))
    )

    names = {tool.name for tool in agent.tools if isinstance(tool, FunctionTool)}
    assert {"get_skill", "get_skill_file"} <= names
    instruction = str(agent.instruction)
    assert 'get_skill("dfo-faq")' in instruction
    assert "get_skill_file" in instruction


def test_a_planner_built_without_a_surface_has_no_skill_tools_and_no_skill_paragraph() -> None:
    agent = create_planner_agent(model_name="m")

    names = {tool.name for tool in agent.tools if isinstance(tool, FunctionTool)}
    assert names.isdisjoint({"get_skill", "get_skill_file"})
    instruction = str(agent.instruction)
    assert "get_skill" not in instruction
    assert "dfo-faq" not in instruction


async def test_get_skill_calls_data_mcps_get_skill_and_returns_its_text() -> None:
    surface = _RecordingSurface(
        ToolCallResult(is_error=False, text=_SKILL_TEXT, structured_content={"uri": "x"})
    )

    answer = await _tools(surface)["get_skill"](skill_id="oda-concepts")

    assert surface.calls == [
        (ToolId(server="data-mcp", name="get_skill"), {"skill_id": "oda-concepts"})
    ]
    assert answer == _SKILL_TEXT


async def test_get_skill_file_passes_path_and_offset_and_leaves_an_unset_digest_out() -> None:
    surface = _RecordingSurface(ToolCallResult(is_error=False, text="# Core ODA\n"))

    answer = await _tools(surface)["get_skill_file"](
        skill_id="oda-concepts", path="references/core-oda.md"
    )

    assert surface.calls == [
        (
            ToolId(server="data-mcp", name="get_skill_file"),
            {"skill_id": "oda-concepts", "path": "references/core-oda.md", "offset": 0},
        )
    ]
    assert answer == "# Core ODA\n"


async def test_get_skill_file_passes_a_continuations_offset_and_digest() -> None:
    surface = _RecordingSurface(ToolCallResult(is_error=False, text="page two"))

    await _tools(surface)["get_skill_file"](
        skill_id="oda-concepts", path="SKILL.md", offset=31844, digest="sha256:" + "a" * 64
    )

    assert surface.calls[0][1] == {
        "skill_id": "oda-concepts",
        "path": "SKILL.md",
        "offset": 31844,
        "digest": "sha256:" + "a" * 64,
    }


@pytest.mark.parametrize("tool", ["get_skill", "get_skill_file"])
async def test_a_refusal_reaches_the_model_as_data_mcps_text(tool: str) -> None:
    refusal = (
        "capability_unavailable: this deployment serves no skills. Carry on without the skill."
    )
    surface = _RecordingSurface(
        ToolCallResult(
            is_error=True, text=refusal, structured_content={"code": "capability_unavailable"}
        )
    )
    arguments = {"skill_id": "dfo-faq"} | ({"path": "SKILL.md"} if tool == "get_skill_file" else {})

    answer = await _tools(surface)[tool](**arguments)

    assert answer == refusal


async def test_a_call_that_does_not_complete_answers_with_one_sentence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    surface = _RecordingSurface(ToolSurfaceUnavailableError("could not call 'get_skill'"))

    with caplog.at_level(logging.WARNING, logger="planner.skills"):
        answer = await _tools(surface)["get_skill"](skill_id="dfo-faq")

    assert answer == (
        "get_skill could not reach data-mcp just now. Draft the plan without this skill."
    )
    [record] = caplog.records
    assert record.getMessage() == (
        "planner get_skill({'skill_id': 'dfo-faq'}) did not complete; drafting without it: "
        "could not call 'get_skill'"
    )
