"""A Gemini or Luna client on Pydantic AI that answers a task's question through DFO's public
`/mcp` and plan-mcp, with a simulated user answering from the task's user script.

The client is an `Agent` with both servers as `MCPToolset`s. Every tool call goes
through `rig.call_tool`, so the model reads what `dfo` prints (text blocks, and
`structuredContent` only on an error result) and the call lands in the run's log.
The simulated user (`simuser.py`) holds `users/<task>.json`; it
answers the client's questions, approves or turns down the plan card, and ends
the conversation once the client has answered or stopped.

Caps per run: 40 tool calls and 12 `propose_plan` calls. Reaching one ends the
run, logged as `cap`.

Both toolsets declare MCP elicitation (Pydantic AI's `elicitation_handler`) and
every call carries the same handler (`rig.simulated_user`), so plan-mcp asks the
simulated user its questions and the plan card directly; `--no-elicit` runs the
client as a host without elicitation, where the model relays both. A tool whose
`_meta` says `strict: false` is declared to the model non-strict.

    uv run --with 'pydantic-ai-slim[google,anthropic,mcp]==2.64.0' --env-file .env \
        --package harness python client.py --reps 2 --concurrency 4

`--model luna` runs `gpt-6-luna` through the local claude-relay on :8787, which speaks
the Anthropic Messages API and reads the Zen key from opencode's own auth file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, replace
from typing import Any

import simuser
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.mcp import CallToolFunc, MCPToolset
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits
from rig import SERVERS, TOKEN, RunLog, call_tool, caller_for, simulated_user
from tasks import SAMPLE

CLIENT_MODELS = ("gemini", "luna")
RELAY_URL = "http://127.0.0.1:8787"
MAX_TOOL_CALLS = 40
MAX_PROPOSALS = 12
MAX_USER_TURNS = 8

CLIENT_INSTRUCTION = """\
You are a general assistant. Answer the user's question with Development Finance
Observatory (DFO) data, using the tools you have. You may call `get_skill` and read
the guidance the tools point you to. Build the answer as a plan: search, open a plan
with `start_plan`, propose it with `propose_plan`, show the user the plan card, and
run it with `run_plan` once they approve. When you need a choice only the user can
make, ask them and wait for their reply. Answer with the figures the run returns.
"""


def client_model(model: str) -> tuple[str, Model | str]:
    """The model's name for the run log, and the Pydantic AI model to run."""
    if model == "luna":
        provider = AnthropicProvider(base_url=RELAY_URL, api_key="relay")
        return "gpt-6-luna", AnthropicModel("gpt-6-luna", provider=provider)
    return "google:gemini-3.8-flash", "google:gemini-3.8-flash"


class CapReachedError(RuntimeError):
    """A run reached its tool-call or proposal cap."""


@dataclass
class Counters:
    tool_calls: int = 0
    proposals: int = 0


def toolset(
    server: str, *, run: str, counters: Counters, model: str, elicitation: Any = None
) -> MCPToolset:
    """`server` as an `MCPToolset` whose calls go through `rig.call_tool` under the caps,
    with `elicitation` answering what the server asks the user."""

    async def process(
        ctx: RunContext[Any], call: CallToolFunc, name: str, args: dict[str, Any]
    ) -> str:
        del ctx, call
        if counters.tool_calls >= MAX_TOOL_CALLS:
            raise CapReachedError(f"{MAX_TOOL_CALLS} tool calls")
        if name == "propose_plan" and counters.proposals >= MAX_PROPOSALS:
            raise CapReachedError(f"{MAX_PROPOSALS} proposals")
        counters.tool_calls += 1
        counters.proposals += name == "propose_plan"
        return await call_tool(
            server=server, tool=name, args=args, run=run, client=model, elicitation=elicitation
        )

    return MCPToolset(
        SERVERS[server],
        id=server,
        auth=TOKEN,
        headers={"x-dfo-caller": caller_for(run)},
        process_tool_call=process,
        read_timeout=300,
        elicitation_handler=elicitation,
    )


async def not_strict(ctx: RunContext[Any], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """Each tool whose MCP `_meta` says `strict: false`, declared non-strict."""
    del ctx
    return [
        replace(tool, strict=False)
        if ((tool.metadata or {}).get("meta") or {}).get("strict") is False
        else tool
        for tool in tools
    ]


def usage_dict(usage: RunUsage) -> dict[str, int]:
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "output_tokens": usage.output_tokens,
        "thinking_tokens": usage.output_reasoning_tokens,
        "tool_calls": usage.tool_calls,
    }


async def run_task(
    task_id: str, rep: int, *, prefix: str, model: str, elicit: bool = True
) -> dict[str, Any]:
    """One conversation on `task_id`, logged to `runs/<prefix>-<task>-r<rep>.jsonl`.
    With `elicit`, plan-mcp asks the simulated user directly."""
    run = f"{prefix}-{task_id}-r{rep}"
    log = RunLog(run)
    log.path.unlink(missing_ok=True)
    counters = Counters()
    model_name, agent_model = client_model(model)
    turns: list[tuple[str, str]] = []
    elicitation = (
        simulated_user(task_id, run, lambda: turns, lambda who, text: turns.append((who, text)))
        if elicit
        else None
    )
    client = Agent(
        agent_model,
        instructions=CLIENT_INSTRUCTION,
        toolsets=[
            toolset(name, run=run, counters=counters, model=model, elicitation=elicitation)
            for name in ("dfo", "plan")
        ],
        capabilities=[PrepareTools(not_strict)],
    )
    log.write(
        {
            "type": "start",
            "task": task_id,
            "rep": rep,
            "client_model": model_name,
            "elicits": elicit,
        }
    )
    started = time.monotonic()
    history: list[Any] = []
    total = RunUsage()
    message = simuser.opening(task_id)
    ending = "user_turns"
    async with client:
        for _ in range(MAX_USER_TURNS):
            turns.append(("User", message))
            try:
                result = await client.run(
                    message,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=80),
                )
            except CapReachedError as cap:
                ending = f"cap: {cap}"
                log.write({"type": "turn", "user": message, "assistant": None, "ending": ending})
                break
            except Exception as exc:  # a client failure ends the run, logged
                ending = f"error: {type(exc).__name__}: {exc}"[:600]
                log.write({"type": "turn", "user": message, "assistant": None, "ending": ending})
                break
            history = result.all_messages()
            total = total + result.usage
            turns.append(("Assistant", result.output))
            log.write(
                {
                    "type": "turn",
                    "user": message,
                    "assistant": result.output,
                    "usage": usage_dict(result.usage),
                }
            )
            try:
                answered = await simuser.reply(task_id, turns)
            except Exception as exc:  # a user-model failure ends the run, logged
                ending = f"user error: {type(exc).__name__}: {exc}"[:600]
                break
            if answered.done:
                ending = "done"
                break
            message = answered.reply
    summary = {
        "type": "end",
        "ending": ending,
        "usage": usage_dict(total),
        "tool_calls": counters.tool_calls,
        "proposals": counters.proposals,
        "wall_s": round(time.monotonic() - started, 1),
    }
    log.write(summary)
    return {"run": run, **summary}


def finished(run: str) -> bool:
    """Whether `run` has a log that ended without an error, so a rerun skips it."""
    path = RunLog(run).path
    if not path.exists():
        return False
    ends = [json.loads(line) for line in path.read_text().splitlines() if '"type": "end"' in line]
    return any(not str(end["ending"]).startswith(("error", "user error")) for end in ends)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--only", help="comma-separated task ids; default the sample")
    parser.add_argument("--model", choices=CLIENT_MODELS, default="gemini")
    parser.add_argument("--prefix", default="gemini")
    parser.add_argument("--no-elicit", action="store_true", help="a host without elicitation")
    args = parser.parse_args()
    tasks = args.only.split(",") if args.only else list(SAMPLE)
    gate = asyncio.Semaphore(args.concurrency)

    async def one(task_id: str, rep: int) -> None:
        async with gate:
            print(
                json.dumps(
                    await run_task(
                        task_id,
                        rep,
                        prefix=args.prefix,
                        model=args.model,
                        elicit=not args.no_elicit,
                    )
                ),
                flush=True,
            )

    jobs = [
        (task_id, rep)
        for rep in range(1, args.reps + 1)
        for task_id in tasks
        if not finished(f"{args.prefix}-{task_id}-r{rep}")
    ]
    async with asyncio.TaskGroup() as group:
        for task_id, rep in jobs:
            group.create_task(one(task_id, rep))
    print(f"ran {len(jobs)} runs", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
