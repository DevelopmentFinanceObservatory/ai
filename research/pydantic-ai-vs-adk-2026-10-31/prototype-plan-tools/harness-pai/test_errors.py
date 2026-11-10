"""Pass line 3: every way a plan-mcp tool call comes back `isError` to this client.

plan-mcp's tools never set `isError` themselves: a refusal is an answer
(`outcome: refused`), and `server._PlanTool.run` turns a bad argument into one. FastMCP
sets `isError` when a tool body raises or the tool name is unknown. The tests below take
each path to the Pydantic AI client and show what the model then reads and whether the
call uses up a retry.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from typing import Any

import pytest
from mcp_planner import HarnessPlanMcp
from outward import ASK_TOOL
from pai_plan_mcp import PlanDeps, open_plan, plan_agent, plan_toolsets
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from rig import TASKS, Rig, Script, call, hand_off_of, say

TASK = "growth-total-oda-ghana"
SERVER_FAULT = "service fault"


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


async def opened(rig: Rig, mcp: Any) -> tuple[PlanDeps, str]:
    task = rig.use(TASK)
    asks, searches = hand_off_of(task)
    plan = await open_plan(mcp, brief=task.question, asks=asks, searches=searches)
    return PlanDeps(plan["plan_session"]), plan["planner_message"]


def mutations(body: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """`body` with each top-level field, and each field of each evidence requirement,
    replaced by a value of the wrong shape or removed."""
    values = (None, [], {}, 5, "x", True, -1, 1e99, [None], {"a": None}, "twenty")
    out = []
    for key in body:
        out.extend(
            (f"{key}={value!r}", {**copy.deepcopy(dict(body)), key: value}) for value in values
        )
        out.append((f"-{key}", {k: v for k, v in body.items() if k != key}))
    for index, requirement in enumerate(body["evidence_requirements"]):
        for key in requirement:
            for value in values:
                changed = copy.deepcopy(dict(body))
                changed["evidence_requirements"][index][key] = value
                out.append((f"requirement {index} {key}={value!r}", changed))
    return out


async def test_a_malformed_proposal_is_an_answer_and_never_is_error(rig: Rig) -> None:
    mcp, _ = plan_toolsets(rig.source)
    async with mcp:
        deps, _ = await opened(rig, mcp)
        sent = mutations(TASKS[TASK].reference_payload)
        errors = {}
        for label, body in sent:
            result = await mcp.client.call_tool(
                "propose_plan", {"plan_session": deps.plan_session, **body}, raise_on_error=False
            )
            if result.is_error:
                errors[label] = result.content[0].text
        print(f"{len(sent)} malformed proposals, {len(errors)} isError")
    assert len(sent) > 200
    assert errors == {}


async def test_a_session_the_server_does_not_hold_is_an_answer(rig: Rig) -> None:
    rig.use(TASK)
    mcp, _ = plan_toolsets(rig.source)
    async with mcp:
        answer = await mcp.direct_call_tool("propose_plan", {"plan_session": "ps_none"})
    assert answer["code"] == "plan_session_unknown"


async def test_a_bad_answer_recorded_by_the_harness_is_a_model_retry_raised_in_the_harness(
    rig: Rig,
) -> None:
    """`record_user_answer` raises in the server on an `answer` that is not an object, and
    the stock client turns that into `ModelRetry`, which no model is there to read."""
    mcp, _ = plan_toolsets(rig.source)
    async with mcp:
        deps, _ = await opened(rig, mcp)
        sent = {"plan_session": deps.plan_session, "message": "m", "answer": "not an object"}
        raw = await mcp.client.call_tool(ASK_TOOL, sent, raise_on_error=False)
        with pytest.raises(ModelRetry) as raised:
            await mcp.direct_call_tool(ASK_TOOL, sent)
    print("isError text:", raw.content[0].text)
    assert raw.is_error
    assert ASK_TOOL in str(raised.value)


@pytest.fixture
def faulty_propose(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fault inside the server's `propose_plan`, as a store or service failure raises."""

    async def propose(self: HarnessPlanMcp, arguments: Mapping[str, Any], **kwargs: Any) -> Any:
        raise RuntimeError(SERVER_FAULT)

    monkeypatch.setattr(HarnessPlanMcp, "propose", propose)


async def test_a_server_fault_on_the_models_call_is_a_retried_return_that_uses_a_retry(
    rig: Rig, faulty_propose: None
) -> None:
    mcp, view = plan_toolsets(rig.source)
    async with mcp:
        deps, message = await opened(rig, mcp)
        faults = [call("propose_plan", {"question": "q"}, id=f"c{i}") for i in (1, 2)]
        script = Script([faults[0], say("gave up")])
        result = await plan_agent(script.model, view, instruction="").run(message, deps=deps)
        (part,) = tool_returns(result.all_messages())
        print("model reads:", part.outcome, part.retry, part.model_response_str())
        assert part.outcome == "retried"
        assert part.retry is not None and part.retry.attempt == 1 == part.retry.max_retries
        assert SERVER_FAULT in part.model_response_str()

        # A second fault in a row has no retry left, and the run ends.
        script = Script([*faults, say("never reached")])
        with pytest.raises(UnexpectedModelBehavior, match="exceeded max retries"):
            await plan_agent(script.model, view, instruction="").run(message, deps=deps)


async def test_with_tool_error_behavior_failed_the_return_is_failed_and_uses_no_retry(
    rig: Rig, faulty_propose: None
) -> None:
    mcp, view = plan_toolsets(rig.source, tool_error_behavior="failed")
    async with mcp:
        deps, message = await opened(rig, mcp)
        faults = [call("propose_plan", {"question": "q"}, id=f"c{i}") for i in (1, 2, 3)]
        script = Script([*faults, say("gave up")])
        result = await plan_agent(script.model, view, instruction="").run(message, deps=deps)
    parts = tool_returns(result.all_messages())
    print("model reads:", [(p.outcome, p.model_response_str()) for p in parts][:1])
    assert [p.outcome for p in parts] == ["failed"] * 3


async def test_a_call_past_the_read_timeout_is_a_retried_return(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_timeout` is the client's request deadline. The model reads the timeout as a
    retried return; over HTTP, where the same setting is the transport's read timeout,
    closing the held connection afterwards raises `httpx2.ReadTimeout` as well."""

    async def slow(self: HarnessPlanMcp, arguments: Mapping[str, Any], **kwargs: Any) -> Any:
        await asyncio.sleep(2)

    monkeypatch.setattr(HarnessPlanMcp, "propose", slow)
    mcp, view = plan_toolsets(rig.source, read_timeout=0.3)
    script = Script([call("propose_plan", {"question": "q"}, id="c1"), say("gave up")])
    seen: list[ModelMessage] = []

    async def run() -> None:
        async with mcp:
            deps, message = await opened(rig, mcp)
            result = await plan_agent(script.model, view, instruction="").run(message, deps=deps)
            seen.extend(result.all_messages())

    if rig.transport == "inproc":
        await run()
    else:
        with pytest.raises(BaseException) as raised:
            await run()
        leaf = raised.value
        while isinstance(leaf, BaseExceptionGroup):
            leaf = leaf.exceptions[0]
        print("closing after a timeout over HTTP:", type(leaf).__module__, type(leaf).__name__)
        assert type(leaf).__name__ == "ReadTimeout"
    (part,) = tool_returns(seen)
    print("timeout, model reads:", part.outcome, part.model_response_str())
    assert part.outcome == "retried"


async def test_a_tool_the_view_hides_is_refused_by_the_client_and_never_reaches_the_server(
    rig: Rig,
) -> None:
    mcp, view = plan_toolsets(rig.source)
    async with mcp:
        deps, message = await opened(rig, mcp)
        before = len(rig.server.server_times)
        script = Script([call("run_plan", {"version": 1}, id="c1"), say("gave up")])
        result = await plan_agent(script.model, view, instruction="").run(message, deps=deps)
    (part,) = tool_returns(result.all_messages())
    print("hidden tool, model reads:", part.outcome, part.retry, part.model_response_str())
    assert part.outcome == "retried"
    assert part.retry is not None and part.retry.cause == "unknown_tool"
    assert len(rig.server.server_times) == before


async def test_a_server_that_cannot_be_reached_ends_the_run_with_the_transport_error() -> None:
    _, view = plan_toolsets("http://127.0.0.1:9/mcp")
    script = Script([say("never reached")])
    with pytest.raises(Exception) as raised:
        await plan_agent(script.model, view, instruction="").run("brief", deps=PlanDeps("ps_x"))
    print("unreachable:", type(raised.value).__module__, type(raised.value).__name__)
    assert not isinstance(raised.value, ModelRetry | UnexpectedModelBehavior)
    assert script.requests == []
