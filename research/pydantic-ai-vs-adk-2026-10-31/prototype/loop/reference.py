"""The same turn with no store and no pause: the answers are resolved inline.

The conversations stay in memory between runs, so this is the reference a stored,
killed and resumed turn must reproduce.
"""

from typing import Any

from agents import Deps, accepted_plans, orchestrator, planner
from loop import execute_stub
from pydantic_ai import Agent, Conversation
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent


async def _run(agent: Agent[Deps, Any], *args: Any, **kwargs: Any) -> AgentRunResult[Any]:
    """Run through the streaming path, the one the scripted models answer on."""
    async with agent.run_stream_events(*args, **kwargs) as events:
        async for event in events:
            if isinstance(event, AgentRunResultEvent):
                return event.result
    raise RuntimeError("run ended without a result")


async def run_reference(
    conversation_id: str, user_text: str, scope: str, deps: Deps
) -> tuple[Conversation, Conversation, str]:
    """Returns the orchestrator and planner conversations and the plan hash."""
    first = await _run(
        orchestrator,
        user_text,
        deps=deps,
        conversation=Conversation(conversation_id=f"{conversation_id}:orchestrator"),
    )
    (plan_call,) = first.output.calls
    brief = plan_call.args_as_dict()["brief"]

    drafted = await _run(
        planner,
        brief,
        deps=deps,
        conversation=Conversation(conversation_id=f"{conversation_id}:planner"),
    )
    (ask,) = drafted.output.calls
    answers = drafted.output.build_results(calls={ask.tool_call_id: scope})
    planned = await _run(
        planner, None, deps=deps, conversation=drafted.conversation, deferred_tool_results=answers
    )
    plan_hash = _plan_hash_of(planned.conversation)

    results: Any = first.output.build_results(
        calls={plan_call.tool_call_id: execute_stub(plan_hash)}
    )
    last = await _run(
        orchestrator,
        None,
        deps=deps,
        conversation=first.conversation,
        deferred_tool_results=results,
    )
    return last.conversation, planned.conversation, plan_hash


def _plan_hash_of(conversation: Conversation) -> str:
    return accepted_plans(conversation.messages)[-1][0]["plan_hash"]
