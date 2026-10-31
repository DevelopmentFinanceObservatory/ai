"""The planner: a Pydantic AI agent on a scripted `FunctionModel`.

The script is a pure function of the message history, so a conversation stored,
reloaded and continued in another process gets the same next move.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, CallDeferred, CustomEvent, DeferredToolRequests, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

SCOPE_OPTIONS = ["Ghana only", "West Africa"]


@dataclass(kw_only=True)
class PlanProgress(CustomEvent):
    """One check of a proposed plan, reported while `propose_plan` runs."""

    call: int
    done: int
    total: int


@dataclass(kw_only=True)
class Deps:
    """Controls shared by the script and the tools.

    `gate` is the signal that the client received a progress event: `propose_plan`
    call `gate_call` (every call when `None`) waits on it after its first progress
    event, for at most `gate_timeout` seconds, then carries on. `fail_in_call` makes
    that `propose_plan` call raise. `log` receives one line per model call and per
    tool body entered and left, in memory and, when `log_file` is set, on disk.
    """

    progress: int = 3
    delay: float = 0.0
    gate: asyncio.Event | None = None
    gate_call: int | None = None
    gate_timeout: float = 10.0
    fail_in_call: int | None = None
    log: list[str] = field(default_factory=list)
    log_file: Path | None = None

    def record(self, line: str) -> None:
        self.log.append(line)
        if self.log_file is not None:
            with self.log_file.open("a") as handle:
                handle.write(line + "\n")


class PlannerToolError(RuntimeError):
    """The failure `propose_plan` raises when `Deps.fail_in_call` names its call."""


def _calls(messages: list[ModelMessage], tool: str) -> list[ToolCallPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == tool
    ]


def _returns(messages: list[ModelMessage], tool: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool
    ]


def _brief(messages: list[ModelMessage]) -> str:
    return next(
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )


def plan_of(messages: list[ModelMessage]) -> dict[str, Any]:
    """The plan the last accepted `propose_plan` call returned."""
    accepted: list[dict[str, Any]] = []
    for part in _returns(messages, "propose_plan"):
        content: dict[str, Any] = part.model_response_object()
        if content["outcome"] == "ok":
            accepted.append(content["plan"])
    return accepted[-1]


def build_planner(deps: Deps) -> Agent[Deps, str | DeferredToolRequests]:
    """The planner agent, its script recording each model call on `deps`."""

    async def script(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        last_returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        asked = _returns(messages, "ask_user")
        proposals = len(_calls(messages, "propose_plan"))
        deps.record(f"model call after {len(messages)} messages")

        if last_returns and last_returns[0].tool_name == "propose_plan":
            outcome = last_returns[0].model_response_object()
            assert isinstance(outcome, dict)
            if outcome["outcome"] == "refused":
                args = {"message": outcome["message"], "options": SCOPE_OPTIONS}
                yield {
                    0: DeltaToolCall(
                        name="ask_user",
                        json_args=json.dumps(args),
                        tool_call_id=f"ask-{len(asked) + 1}",
                    )
                }
                return
            yield f"Plan {outcome['plan']['hash'][:12]} is ready."
            return

        scope = str(asked[-1].content) if asked else None
        args = {"brief": _brief(messages), "scope": scope}
        yield {
            0: DeltaToolCall(
                name="propose_plan",
                json_args=json.dumps(args),
                tool_call_id=f"propose-{proposals + 1}",
            )
        }

    agent = Agent(
        FunctionModel(stream_function=script),
        deps_type=Deps,
        output_type=str | DeferredToolRequests,
        name="planner",
    )

    @agent.tool
    async def ask_user(ctx: RunContext[Deps], message: str, options: list[str]) -> str:
        """Ask the user to choose; the answer arrives as a deferred result."""
        ctx.deps.record("ask_user body")
        raise CallDeferred

    @agent.tool
    async def propose_plan(ctx: RunContext[Deps], brief: str, scope: str | None) -> dict[str, Any]:
        """Check a draft plan, reporting progress, then accept or refuse it."""
        deps = ctx.deps
        call = len(_returns(ctx.messages, "propose_plan")) + 1
        deps.record(f"propose_plan call={call} start")
        for done in range(1, deps.progress + 1):
            await ctx.emit(PlanProgress(call=call, done=done, total=deps.progress))
            if done == 1 and deps.gate is not None and deps.gate_call in (None, call):
                try:
                    await asyncio.wait_for(deps.gate.wait(), deps.gate_timeout)
                except TimeoutError:
                    deps.record(f"propose_plan call={call} gate timed out")
                except asyncio.CancelledError:
                    deps.record(f"propose_plan call={call} cancelled")
                    raise
            if done == 1 and deps.fail_in_call == call:
                raise PlannerToolError(f"propose_plan call {call} failed")
            await asyncio.sleep(deps.delay)
        deps.record(f"propose_plan call={call} end")
        if scope is None:
            return {
                "outcome": "refused",
                "message": "The question fits two scopes. Which one?",
            }
        steps = [f"Fetch {brief} for {scope}", "Sum by year"]
        digest = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
        return {"outcome": "ok", "plan": {"steps": steps, "hash": digest}}

    return agent
