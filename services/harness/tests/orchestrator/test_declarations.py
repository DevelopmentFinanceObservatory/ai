"""The orchestrator's nine declarations, built without ADK, against the bytes ADK sent.

`fixtures/orchestrator_requests/adk_declarations.json` holds each tool's
declaration as ADK's `FunctionTool` built it, recorded before the orchestrator
left ADK. `tool_declaration` must give the same JSON for every tool, and the
same bytes as the recorded requests carried them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from harness.orchestrator_tools import TOOL_NAMES, build_orchestrator_tools
from orchestrator.declarations import tool_declaration

_RECORDED = Path(__file__).parent.parent / "fixtures" / "orchestrator_requests"


async def _unused(**kwargs: Any) -> Any:
    raise AssertionError("no call expected")


def _declarations() -> dict[str, dict[str, Any]]:
    tools = build_orchestrator_tools(resolver=_unused, executor=_unused)
    return {
        tool.__name__: tool_declaration(tool).model_dump(mode="json", exclude_none=True)
        for tool in tools
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_each_declaration_is_the_one_adk_built(name: str) -> None:
    recorded = json.loads((_RECORDED / "adk_declarations.json").read_text())

    assert _canonical(_declarations()[name]) == _canonical(recorded[name])


@pytest.mark.parametrize("scenario", ["answer", "plan", "ask"])
def test_each_declaration_is_the_one_every_recorded_request_sent(scenario: str) -> None:
    requests = json.loads((_RECORDED / f"adk_{scenario}.json").read_text())
    built = _declarations()

    for request in requests:
        (tool,) = request["config"]["tools"]
        sent = tool["function_declarations"]
        assert [declaration["name"] for declaration in sent] == list(TOOL_NAMES)
        assert [_canonical(d) for d in sent] == [_canonical(built[n]) for n in TOOL_NAMES]


def test_a_changed_description_is_caught() -> None:
    """Control: the comparison sees a one-word change."""
    recorded = json.loads((_RECORDED / "adk_declarations.json").read_text())
    changed = {**_declarations()["fetch"]}
    changed["description"] += " Changed."

    assert _canonical(changed) != _canonical(recorded["fetch"])
