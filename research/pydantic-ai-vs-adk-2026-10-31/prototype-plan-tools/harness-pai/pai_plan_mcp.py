"""A Pydantic AI planner whose tools are plan-mcp's, through Pydantic AI's own MCP client.

The planner is an `Agent` over `MCPToolset`, the stock MCP client, with three
stock hooks and no other harness code in the tool path:

- `process_tool_call` fills `plan_session` from the run's deps on every call the
  model makes;
- `PreparedToolset` (`.prepared`) is the model's view of plan-mcp's tools: it
  leaves `start_plan` and `run_plan` out, hides `plan_session` from
  `propose_plan`, and declares `record_user_answer` less `plan_session` and
  `answer` as a deferred tool (`kind="external"`);
- the deferred tool is the pause: the run returns `DeferredToolRequests`, and
  `resume` records the user's answer on plan-mcp, `by: harness`, before the run
  continues with `DeferredToolResults`.

`open_plan` and `record_answers` call plan-mcp through the same client, outside
the model's view, as the harness's seam does in `harness-ab/mcp_planner.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from mcp_planner import PROPOSE_PLAN, START_PLAN, mcp_instruction, user_answer, without
from outward import ASK_TOOL
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPToolset, MCPToolsetClient, ProcessToolCallback
from pydantic_ai.models import Model
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets import PreparedToolset

__all__ = [
    "PlanDeps",
    "mcp_instruction",
    "open_plan",
    "plan_agent",
    "plan_toolsets",
    "record_answers",
]

# What `plan_session` and `answer` hide: the harness fills the first, and the
# user's answer reaches plan-mcp when the run resumes.
_PROPOSE_HIDDEN = ("plan_session",)
_ASK_HIDDEN = ("plan_session", "answer")


@dataclass(frozen=True)
class PlanDeps:
    """What a run carries beside the model: the plan session it drafts in."""

    plan_session: str


def _fill_session(meta: Callable[[], Mapping[str, str]]) -> ProcessToolCallback:
    """The hook that adds `plan_session` and the request `_meta` `meta()` names to a call."""

    async def fill(
        ctx: RunContext[PlanDeps], call_tool: CallToolFunc, name: str, args: dict[str, Any]
    ) -> Any:
        return await call_tool(
            name, {"plan_session": ctx.deps.plan_session, **args}, metadata=dict(meta()) or None
        )

    return fill


def _model_view(ctx: RunContext[PlanDeps], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    del ctx
    shown = []
    for tool in tools:
        if tool.name == PROPOSE_PLAN:
            schema = without(tool.parameters_json_schema, _PROPOSE_HIDDEN)
            shown.append(replace(tool, parameters_json_schema=schema))
        elif tool.name == ASK_TOOL:
            schema = without(tool.parameters_json_schema, _ASK_HIDDEN)
            shown.append(replace(tool, parameters_json_schema=schema, kind="external"))
    return shown


def plan_toolsets(
    source: MCPToolsetClient, *, meta: Callable[[], Mapping[str, str]] = dict, **options: Any
) -> tuple[MCPToolset[PlanDeps], PreparedToolset[PlanDeps]]:
    """The MCP client of plan-mcp at `source`, and the model's view of its tools.

    The client is the harness's own handle (`open_plan`, `record_answers`); the view
    is what an agent is given. `meta` gives the `_meta` each of the model's calls
    carries (the harness's `common.telemetry.turn_call_meta`). `options` go to
    `MCPToolset`.
    """
    client = MCPToolset(source, process_tool_call=_fill_session(meta), **options)
    return client, client.prepared(_model_view)


def plan_agent(
    model: Model, toolset: PreparedToolset[PlanDeps], *, instruction: str
) -> Agent[PlanDeps, str | DeferredToolRequests]:
    return Agent(
        model,
        deps_type=PlanDeps,
        output_type=[str, DeferredToolRequests],
        instructions=instruction,
        toolsets=[toolset],
    )


async def open_plan(
    client: MCPToolset[PlanDeps],
    *,
    brief: str,
    asks: Sequence[Any],
    searches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """`start_plan`: the session id and the planner's first message."""
    return await client.direct_call_tool(
        START_PLAN, {"brief": brief, "asks": list(asks), "searches": list(searches)}
    )


async def record_answers(
    client: MCPToolset[PlanDeps],
    plan_session: str,
    requests: DeferredToolRequests,
    response: Mapping[str, Any],
) -> None:
    """Record each paused ask and the user's `response` on plan-mcp, `by: harness`."""
    for call in requests.calls:
        await client.direct_call_tool(
            ASK_TOOL,
            {"plan_session": plan_session, **call.args_as_dict(), "answer": user_answer(response)},
        )
