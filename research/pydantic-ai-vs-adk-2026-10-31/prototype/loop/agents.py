"""Two Pydantic AI agents on scripted models.

Each script is a pure function of the message history, so a conversation that is
stored, reloaded and continued in another process gets the same next move.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    CallDeferred,
    CustomEvent,
    DeferredToolRequests,
    RunContext,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)

SCOPE_OPTIONS = ["Ghana only", "West Africa"]


class Step(BaseModel):
    title: str
    detail: str = ""


@dataclass(kw_only=True)
class PlanProgress(CustomEvent):
    """What the translator reports while it checks a draft plan."""

    done: int
    total: int
    note: str


@dataclass(kw_only=True)
class Deps:
    """Test and driver controls shared by both agents.

    `gate` is awaited by `propose_plan` call number `gate_call` (every call when
    `gate_call` is None) right after its first progress event.
    `tool_log` receives one line per tool body entered, so a test can tell whether a
    body ran twice.
    """

    progress: int = 3
    delay: float = 0.02
    gate: asyncio.Event | None = None
    gate_call: int | None = None
    tool_log: Path | None = None


def log_tool(deps: Deps, line: str) -> None:
    if deps.tool_log is not None:
        with deps.tool_log.open("a") as handle:
            handle.write(line + "\n")


def plan_hash(steps: Sequence[Step]) -> str:
    canonical = json.dumps(
        [step.model_dump() for step in steps], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def result_of(part: ToolReturnPart) -> dict[str, Any]:
    """The dict a tool of this module returned."""
    assert isinstance(part.content, dict)
    return part.content


def _parts(messages: list[ModelMessage]) -> list[Any]:
    return [part for message in messages for part in message.parts]


def _returns(messages: list[ModelMessage], tool: str) -> list[ToolReturnPart]:
    return [
        part
        for part in _parts(messages)
        if isinstance(part, ToolReturnPart) and part.tool_name == tool
    ]


def _calls(messages: list[ModelMessage], tool: str) -> list[ToolCallPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == tool
    ]


def _user_prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def accepted_plans(messages: list[ModelMessage]) -> list[tuple[dict, list[Step]]]:
    """Every `propose_plan` result with outcome ok, with the steps it accepted."""
    steps_by_call = {
        call.tool_call_id: [Step(**s) for s in call.args_as_dict()["steps"]]
        for call in _calls(messages, "propose_plan")
    }
    return [
        (result_of(ret), steps_by_call[ret.tool_call_id])
        for ret in _returns(messages, "propose_plan")
        if result_of(ret)["outcome"] == "ok"
    ]


async def _words(text: str) -> AsyncIterator[str]:
    words = text.split(" ")
    for index, word in enumerate(words):
        yield word if index == len(words) - 1 else word + " "


# Planner -------------------------------------------------------------------


def _draft_steps(brief: str, scope: str | None, amendment: str | None) -> list[Step]:
    steps = [Step(title="Fetch the series", detail=f"{brief} ({scope or 'default'})")]
    steps.append(Step(title="Aggregate", detail="Sum by year"))
    if amendment is not None:
        steps.append(Step(title="Amendment", detail=amendment))
    return steps


async def planner_script(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[DeltaToolCalls | str]:
    last = messages[-1]
    last_returns = [p for p in last.parts if isinstance(p, ToolReturnPart)]
    prompts = _user_prompts(messages)
    asked = _returns(messages, "ask_user")
    scope = str(asked[-1].content) if asked else None
    proposal_no = len(_calls(messages, "propose_plan")) + 1

    if last_returns and last_returns[0].tool_name == "propose_plan":
        outcome = result_of(last_returns[0])
        if outcome["outcome"] == "refused":
            yield {
                0: DeltaToolCall(
                    name="ask_user",
                    json_args=json.dumps(
                        {"message": outcome["message"], "options": outcome["options"]}
                    ),
                    tool_call_id=f"ask-{len(asked) + 1}",
                )
            }
            return
        async for chunk in _words(
            f"Plan revision {outcome['revision']} is ready: {outcome['plan_hash']}."
        ):
            yield chunk
        return

    if last_returns and last_returns[0].tool_name == "ask_user":
        steps = _draft_steps(prompts[0], scope, None)
    elif accepted_plans(messages):
        steps = _draft_steps(prompts[0], scope, prompts[-1])
    else:
        steps = _draft_steps(prompts[0], None, None)
        async for chunk in _words("Drafting a plan."):
            yield chunk

    yield {
        0: DeltaToolCall(
            name="propose_plan",
            json_args=json.dumps(
                {
                    "question": prompts[0],
                    "steps": [step.model_dump() for step in steps],
                }
            ),
            tool_call_id=f"propose-{proposal_no}",
        )
    }


planner = Agent(
    FunctionModel(stream_function=planner_script),
    deps_type=Deps,
    output_type=str | DeferredToolRequests,
    name="planner",
)


@planner.tool
async def ask_user(ctx: RunContext[Deps], message: str, options: list[str]) -> str:
    log_tool(ctx.deps, "ask_user")
    raise CallDeferred


@planner.tool
async def propose_plan(ctx: RunContext[Deps], question: str, steps: list[Step]) -> dict[str, Any]:
    """Stand-in for the translator: report progress, then accept or refuse."""
    deps = ctx.deps
    call_no = len(_returns(ctx.messages, "propose_plan")) + 1
    revision = len(accepted_plans(ctx.messages)) + 1
    log_tool(deps, f"propose_plan call={call_no} start")

    for done in range(1, deps.progress + 1):
        note = f"check {done} of {deps.progress}"
        await ctx.emit(PlanProgress(done=done, total=deps.progress, note=note))
        if done == 1 and deps.gate is not None and deps.gate_call in (None, call_no):
            await deps.gate.wait()
        await asyncio.sleep(deps.delay)

    log_tool(deps, f"propose_plan call={call_no} end")
    if not _returns(ctx.messages, "ask_user"):
        return {
            "outcome": "refused",
            "code": "choose_scope",
            "message": "The question fits two scopes. Which one?",
            "options": SCOPE_OPTIONS,
        }
    return {"outcome": "ok", "revision": revision, "plan_hash": plan_hash(steps)}


# Orchestrator --------------------------------------------------------------


async def orchestrator_script(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[DeltaToolCalls | str]:
    plan_returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
    if plan_returns:
        result = result_of(plan_returns[0])
        text = f"Plan {result['plan_hash'][:12]} ran and returned {result['rows']} rows."
        async for chunk in _words(text):
            yield chunk
        return
    yield {
        0: DeltaToolCall(
            name="plan",
            json_args=json.dumps({"brief": _user_prompts(messages)[-1]}),
            tool_call_id=f"plan-{len(_calls(messages, 'plan')) + 1}",
        )
    }


orchestrator = Agent(
    FunctionModel(stream_function=orchestrator_script),
    deps_type=Deps,
    output_type=str | DeferredToolRequests,
    name="orchestrator",
)


@orchestrator.tool
async def plan(ctx: RunContext[Deps], brief: str) -> str:
    log_tool(ctx.deps, "plan")
    raise CallDeferred


def text_of(messages: list[ModelMessage]) -> str:
    """The text of the last response, for a final answer."""
    last = messages[-1]
    assert isinstance(last, ModelResponse)
    return "".join(p.content for p in last.parts if isinstance(p, TextPart))
