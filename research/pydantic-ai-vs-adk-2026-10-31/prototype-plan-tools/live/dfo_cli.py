"""Call any tool on DFO's public `/mcp` or on plan-mcp and print what a model sees.

    dfo tools <dfo|plan> [--schemas]
    dfo call <dfo|plan> <tool> '<json args>' --run <run> [--no-elicit]
    dfo calls --run <run>

One call per process. Every call of a run sends the run's own `x-dfo-caller`,
and is appended with its result to `runs/<run>.jsonl`. A call declares MCP
elicitation, so plan-mcp asks the run's simulated user (`simuser`) its questions
and the plan card directly, unless `--no-elicit` says the host cannot; the
exchange is logged beside the run's own turns. `calls` prints how many
tool calls the run has made. A run stops at 40 tool calls and 12 `propose_plan`
calls: a call past either cap is refused without reaching the server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from fastmcp import Client
from rig import SERVERS, RunLog, call_tool, caller_for, simulated_user, transport
from user_cli import turns


async def list_tools(server: str, *, schemas: bool) -> str:
    async with Client(transport(server, caller=caller_for("tools")), timeout=60) as mcp:
        tools = await mcp.list_tools()
    lines = []
    for tool in tools:
        lines.append(f"## {tool.name}\n{tool.description or ''}")
        if schemas:
            lines.append(json.dumps(tool.input_schema, indent=1))
    return "\n\n".join(lines)


MAX_TOOL_CALLS = 40
MAX_PROPOSALS = 12


def logged_calls(run: str) -> list[dict]:
    path = RunLog(run).path
    if not path.exists():
        return []
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [line for line in lines if line.get("type") == "call"]


def cap_reached(run: str, tool: str) -> str | None:
    """The cap a further call of `tool` would pass, if any."""
    calls = logged_calls(run)
    if len(calls) >= MAX_TOOL_CALLS:
        return f"{MAX_TOOL_CALLS} tool calls"
    proposals = sum(call["tool"] == "propose_plan" for call in calls)
    if tool == "propose_plan" and proposals >= MAX_PROPOSALS:
        return f"{MAX_PROPOSALS} proposals"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="dfo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tools = sub.add_parser("tools")
    tools.add_argument("server", choices=sorted(SERVERS))
    tools.add_argument("--schemas", action="store_true")
    call = sub.add_parser("call")
    call.add_argument("server", choices=sorted(SERVERS))
    call.add_argument("tool")
    call.add_argument("args", nargs="?", default="{}")
    call.add_argument("--run", required=True)
    call.add_argument("--no-elicit", action="store_true", help="declare no elicitation")
    calls = sub.add_parser("calls")
    calls.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "tools":
        print(asyncio.run(list_tools(args.server, schemas=args.schemas)))
        return 0
    if args.command == "calls":
        print(len(logged_calls(args.run)))
        return 0
    try:
        arguments = json.loads(args.args)
    except json.JSONDecodeError as exc:
        print(f"arguments are not valid JSON: {exc}", file=sys.stderr)
        return 64
    if cap := cap_reached(args.run, args.tool):
        print(f"cap reached: this run has made {cap}; no more calls. End the run.")
        return 3
    elicitation = None
    if not args.no_elicit:
        task, _, _ = turns(args.run)
        elicitation = simulated_user(task, args.run, lambda: turns(args.run)[1])
    seen = asyncio.run(
        call_tool(
            server=args.server,
            tool=args.tool,
            args=arguments,
            run=args.run,
            client="cli",
            elicitation=elicitation,
        )
    )
    print(seen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
