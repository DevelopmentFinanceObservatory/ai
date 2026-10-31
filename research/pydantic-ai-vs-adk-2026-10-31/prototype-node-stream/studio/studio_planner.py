"""The scripted Pydantic AI planner a Studio turn drafts with.

`propose_plan` reports progress through `ctx.emit` (stage A's `PlanProgress`, the
event stage A's relay forwards), can wait on a gate after its first report, and
returns the canned plan `harness.scripted_plan.scripted_plan` builds, the plan
`ScriptedPlannerModel` drafts on dev. `ask_user` defers, so the run ends with a
`DeferredToolRequests` the node turns into its own interrupt.

`SENTINEL` is written into everything the planner says that only its own
messages hold: the refusal it reads, its ask and its final text. A request to the
orchestrator's model that contains it carries the planner's messages.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from harness.scripted_plan import scripted_plan
from planner_agent import PlanProgress, _calls, _returns
from pydantic_ai import Agent, CallDeferred, DeferredToolRequests, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

SENTINEL = "PAI-PLANNER-ONLY"
SCOPE_OPTIONS = ["Ghana only", "West Africa"]


@dataclass(kw_only=True)
class StudioDeps:
    """Controls shared by the script and the tools.

    `ask` makes the first proposal refuse for want of a scope, so the planner asks.
    `gate` holds `propose_plan` call `gate_call` after its first report until the
    test sets it, for at most `gate_timeout` seconds. `log` holds one line per model
    call and per tool body entered, left or cancelled.
    """

    ask: bool = True
    progress: int = 3
    gate: asyncio.Event | None = None
    gate_call: int = 1
    gate_timeout: float = 10.0
    log: list[str] = field(default_factory=list)


def _brief(messages: list[ModelMessage]) -> str:
    return next(
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )


async def report_progress(
    deps: StudioDeps, *, call: int, emit: Any, label: str = "propose_plan"
) -> None:
    """Send `deps.progress` reports for `call` through `emit`, waiting on the gate
    after the first one; log the call's start, end or cancellation."""
    deps.log.append(f"{label} call={call} start")
    try:
        for done in range(1, deps.progress + 1):
            await emit(PlanProgress(call=call, done=done, total=deps.progress))
            if done == 1 and deps.gate is not None and deps.gate_call == call:
                try:
                    await asyncio.wait_for(deps.gate.wait(), deps.gate_timeout)
                except TimeoutError:
                    deps.log.append(f"{label} call={call} gate timed out")
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        deps.log.append(f"{label} call={call} cancelled")
        raise
    deps.log.append(f"{label} call={call} end")


def build_studio_planner(deps: StudioDeps) -> Agent[StudioDeps, str | DeferredToolRequests]:
    """The planner agent, its script a pure function of the message history."""

    async def script(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        deps.log.append(f"model call after {len(messages)} messages")
        last_returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        asked = _returns(messages, "ask_user")
        if last_returns and last_returns[0].tool_name == "propose_plan":
            outcome = last_returns[0].model_response_object()
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
            yield f"The plan is ready. {SENTINEL}"
            return
        scope = str(asked[-1].content) if asked else (None if deps.ask else "as asked")
        args = {"brief": _brief(messages), "scope": scope}
        proposals = len(_calls(messages, "propose_plan"))
        yield {
            0: DeltaToolCall(
                name="propose_plan",
                json_args=json.dumps(args),
                tool_call_id=f"propose-{proposals + 1}",
            )
        }

    agent = Agent(
        FunctionModel(stream_function=script),
        deps_type=StudioDeps,
        output_type=str | DeferredToolRequests,
        name="planner",
    )

    @agent.tool
    async def ask_user(ctx: RunContext[StudioDeps], message: str, options: list[str]) -> str:
        """Ask the user to choose; the answer arrives as a deferred result."""
        ctx.deps.log.append("ask_user body")
        raise CallDeferred

    @agent.tool
    async def propose_plan(
        ctx: RunContext[StudioDeps], brief: str, scope: str | None
    ) -> dict[str, Any]:
        """Check a draft plan, reporting progress, then accept or refuse it."""
        call = len(_returns(ctx.messages, "propose_plan")) + 1
        await report_progress(ctx.deps, call=call, emit=ctx.emit)
        if scope is None:
            return {
                "outcome": "refused",
                "message": f"Which scope should the plan cover? {SENTINEL}",
            }
        ctx.deps.log.append(f"propose_plan call={call} scope={scope}")
        plan = scripted_plan(brief, spec_id=None)
        return {"outcome": "ok", "plan": plan.model_dump(mode="json")}

    return agent
