"""The two MCP servers an outside client uses, what a model sees of a call, and the run log.

- `dfo` is the harness's public `/mcp` (`search`, `get_skill`, the data and
  analysis tools), `plan` is plan-mcp (`start_plan`, `propose_plan`,
  `record_user_answer`, `run_plan`).
- `render` is the model's view of a result: its text blocks, and
  `structuredContent` only on an error result, as
  `harness.tool_results.make_structured_content_stripper` leaves it.
- `RunLog` appends every call and its result to `runs/<run>.jsonl`, which
  `score.py` reads.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SERVERS = {
    "dfo": os.environ.get("DFO_PUBLIC_MCP_URL", "http://127.0.0.1:8080/mcp"),
    "plan": os.environ.get("DFO_PLAN_MCP_URL", "http://127.0.0.1:8091/mcp"),
}
# The bearer both servers were started with (`MCP_API_TOKEN=local-dev`). Its own
# name, since `--env-file .env` sets `MCP_API_TOKEN` to the checkout's value.
TOKEN = os.environ.get("RIG_MCP_TOKEN", "local-dev")
CALLER_HEADER = "x-dfo-caller"


def caller_for(run: str) -> str:
    """The `x-dfo-caller` every call of `run` sends, so its plan session stays its own."""
    return f"key:outside-{run}"


def transport(server: str, *, caller: str) -> StreamableHttpTransport:
    return StreamableHttpTransport(SERVERS[server], auth=TOKEN, headers={CALLER_HEADER: caller})


def render(result: Any) -> str:
    """What a model reads of a tool result."""
    is_error = bool(getattr(result, "is_error", False))
    lines = [block.text for block in result.content or [] if getattr(block, "text", None)]
    structured = getattr(result, "structured_content", None)
    if is_error and structured is not None:
        lines.append("structuredContent: " + json.dumps(structured, sort_keys=True))
    return "\n".join(lines)


@dataclass
class RunLog:
    """The JSON-lines log of one run: one line per tool call, plus client lines."""

    run: str

    @property
    def path(self) -> Path:
        return RUNS / f"{self.run}.jsonl"

    def write(self, record: Mapping[str, Any]) -> None:
        RUNS.mkdir(exist_ok=True)
        line = {"at": datetime.now(UTC).isoformat(), "run": self.run, **record}
        with self.path.open("a") as out:
            out.write(json.dumps(line, default=str) + "\n")

    def call(
        self,
        *,
        server: str,
        tool: str,
        args: Mapping[str, Any],
        result: Any,
        elapsed_ms: int,
        client: str,
    ) -> None:
        self.write(
            {
                "type": "call",
                "client": client,
                "server": server,
                "tool": tool,
                "args": dict(args),
                "is_error": bool(getattr(result, "is_error", False)),
                "seen": render(result),
                "structured": getattr(result, "structured_content", None),
                "elapsed_ms": elapsed_ms,
            }
        )


async def call_tool(
    *, server: str, tool: str, args: Mapping[str, Any], run: str, client: str
) -> str:
    """Call `tool` on `server` as `run`'s caller, log it, and answer the model's view."""
    started = time.monotonic()
    async with Client(transport(server, caller=caller_for(run)), timeout=300) as mcp:
        result = await mcp.call_tool(tool, dict(args), raise_on_error=False)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    RunLog(run).call(
        server=server,
        tool=tool,
        args=args,
        result=result,
        elapsed_ms=elapsed_ms,
        client=client,
    )
    return render(result)
