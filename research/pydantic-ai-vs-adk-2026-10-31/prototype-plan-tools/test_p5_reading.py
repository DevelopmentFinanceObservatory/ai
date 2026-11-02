"""P5: what an outside model reads before its first proposal, beside what DFO's planner
reads today, and what Anthropic's and OpenAI's tool APIs would reject.

`measurements.json` beside this file holds the numbers the test computes. Tokens are
estimated at four characters a token.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest
from checks import call, compact, opening, recorded_server, tokens
from entry import load_tasks, quiet_logs
from fastmcp import Client
from google.adk.tools.base_tool import BaseTool
from schema_rules import faults, summary

TASKS = load_tasks()
OUT = Path(__file__).with_name("measurements.json")


@pytest.fixture(autouse=True, scope="module")
def _quiet() -> None:
    quiet_logs()


def _row(name: str, chars: int) -> dict[str, Any]:
    return {"name": name, "chars": chars, "tokens": tokens(chars)}


async def _measure() -> dict[str, Any]:
    messages: list[int] = []
    declarations: list[dict[str, Any]] = []
    guidance = ""
    planner: dict[str, Any] = {}
    api: dict[str, Any] = {}
    for index, task in enumerate(TASKS):
        recorded = await recorded_server(task)
        async with Client(recorded.server.mcp) as client:
            opened = await call(client, "start_plan", opening(task))
            messages.append(len(opened["planner_message"]))
            if index:
                continue
            guidance = opened["guidance"]
            for tool in await client.list_tools():
                schema = tool.input_schema
                declarations.append(
                    {
                        "name": tool.name,
                        "description": len(tool.description or ""),
                        "input_schema": compact(schema),
                        "chars": len(tool.name) + len(tool.description or "") + compact(schema),
                    }
                )
                api[tool.name] = summary(faults(tool.name, schema))
        seam = recorded.server.service._seam
        tools = [tool for tool in seam._agent.tools if isinstance(tool, BaseTool)]
        planner = {
            "instruction": len(str(seam._agent.instruction)),
            "declarations": {
                tool.name: compact(
                    tool._get_declaration().model_dump(mode="json", exclude_none=True)  # type: ignore[union-attr]
                )
                for tool in tools
            },
        }
    message = {
        "median": int(statistics.median(messages)),
        "max": max(messages),
        "min": min(messages),
    }
    outside = sum(d["chars"] for d in declarations) + len(guidance) + message["median"]
    today = planner["instruction"] + sum(planner["declarations"].values()) + message["median"]
    return {
        "plan_mcp": {
            "declarations": declarations,
            "guidance": _row("start_plan guidance", len(guidance)),
            "planner_message": message,
            "before_first_proposal": _row("total, median task", outside),
        },
        "planner_today": {
            **planner,
            "first_message": message,
            "before_first_proposal": _row("total, median task", today),
        },
        "api_faults": api,
    }


async def test_p5_what_an_outside_model_reads_and_what_the_apis_reject() -> None:
    measured = await _measure()
    OUT.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(measured, indent=2))
    names = [d["name"] for d in measured["plan_mcp"]["declarations"]]
    assert names == ["start_plan", "propose_plan", "record_user_answer", "run_plan"]
    # Every declaration passes both APIs' non-strict rules.
    assert all(row["both"]["faults"] == 0 for row in measured["api_faults"].values())


def test_p5_control_the_checker_flags_what_each_api_rejects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "default": "x"},
            "node": {"$ref": "#/$defs/Node"},
        },
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
    }
    found = faults("bad name!", schema)
    assert found["both"] == ["name 'bad name!'"]
    assert "recursive $defs ['Node']" in found["anthropic_strict"]
    assert "#/properties/name: minLength" in found["anthropic_strict"]
    assert "#/properties/name: default" in found["openai_strict"]
    assert "#: optional properties" in found["openai_strict"]
    assert faults("fine", {"type": "object", "properties": {}, "additionalProperties": False}) == {
        "both": [],
        "anthropic_strict": [],
        "openai_strict": [],
    }
