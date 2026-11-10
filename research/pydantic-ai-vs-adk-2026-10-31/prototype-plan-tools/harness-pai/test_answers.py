"""Pass line 2: what the Pydantic AI model reads back is what arm B's model read.

On the two calls `harness-ab/measure.results` measures, the part the Pydantic AI model
reads is a `ToolReturnPart` whose content equals B's answer, as a value and as compact
JSON characters. The control marks the same answer `isError`; it reaches the model as a
retried `ToolReturnPart` (`outcome="retried"`), which Pydantic AI 2.64 sends in place of the
deprecated `RetryPromptPart`, and the model reads a different text.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest
import server as plan_server
from fastmcp.tools.base import ToolResult
from mcp.types import TextContent
from measure import chars
from pai_plan_mcp import PlanDeps, open_plan, plan_agent, plan_toolsets
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from rig import TASKS, Rig, Script, b_client, b_response, b_run, call, hand_off_of, say

COMBINATION = "combination_unasked (sum-oda-oof-ethiopia reference)"
INVALID_PLAN = "invalid_plan (growth-total-oda-ghana, period.start 'twenty twelve')"
CASES = [COMBINATION, INVALID_PLAN]


def body_of(case: str) -> tuple[str, dict[str, Any]]:
    if case == COMBINATION:
        task = "sum-oda-oof-ethiopia"
        return task, copy.deepcopy(TASKS[task].reference_payload)
    task = "growth-total-oda-ghana"
    body = copy.deepcopy(TASKS[task].reference_payload)
    body["evidence_requirements"][0]["period"] = {"start": "twenty twelve", "end": 2021}
    return task, body


def returned(messages: list[ModelMessage], tool: str) -> ToolReturnPart:
    """The part that answers the model's one call of `tool`."""
    (part,) = (
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool
    )
    return part


def reads_what_b_read(part: ToolReturnPart, b: Mapping[str, Any]) -> bool:
    return (
        part.outcome == "success"
        and part.content == b
        and part.model_response_str() == json.dumps(b, separators=(",", ":"), ensure_ascii=False)
    )


async def b_answer(rig: Rig, task_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
    async with b_client(rig) as client:
        run = await b_run(rig, client, task_id, [[("propose_plan", body)]])
    return b_response(run, "propose_plan")


async def pai_part(
    rig: Rig,
    task_id: str,
    body: Mapping[str, Any],
    *,
    before_propose: Callable[[], None] = lambda: None,
) -> ToolReturnPart:
    """What the Pydantic AI model reads after it proposes `body` as `task_id`'s plan."""
    task = rig.use(task_id)
    asks, searches = hand_off_of(task)
    mcp, view = plan_toolsets(rig.source)
    async with mcp:
        opened = await open_plan(mcp, brief=task.question, asks=asks, searches=searches)
        before_propose()
        script = Script([call("propose_plan", body, id="c1"), say("done")])
        result = await plan_agent(script.model, view, instruction="").run(
            opened["planner_message"], deps=PlanDeps(opened["plan_session"])
        )
    return returned(result.all_messages(), "propose_plan")


@pytest.mark.parametrize("case", CASES)
async def test_the_model_reads_what_b_read(rig: Rig, case: str) -> None:
    task_id, body = body_of(case)
    b = await b_answer(rig, task_id, body)
    part = await pai_part(rig, task_id, body)
    print(case, "| PAI", chars(part.content), "chars | B", chars(b), "chars")
    assert reads_what_b_read(part, b)
    assert b["outcome"] != "ok"


@pytest.mark.parametrize("case", CASES)
async def test_control_the_same_answer_marked_is_error_is_a_retried_return(
    rig: Rig, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id, body = body_of(case)
    b = await b_answer(rig, task_id, body)

    def mark_every_answer() -> None:
        def result(answer: Mapping[str, Any]) -> ToolResult:
            text = json.dumps(answer, separators=(",", ":"), default=str)
            return ToolResult(
                content=[TextContent(type="text", text=text)],
                structured_content=dict(answer),
                is_error=True,
            )

        monkeypatch.setattr(plan_server, "_result", result)

    part = await pai_part(rig, task_id, body, before_propose=mark_every_answer)
    read = part.model_response_str()
    print("control", case, "|", part.outcome, part.retry, "|", len(read), "chars |", read[:100])
    assert part.outcome == "retried"
    assert not reads_what_b_read(part, b)
