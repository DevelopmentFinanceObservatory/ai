"""One task end to end by hand, with its reference plan: public `search`,
`start_plan`, `propose_plan`, `run_plan`. Writes `runs/smoke-<task>.jsonl`.

    python smoke.py series-gdp-ghana
"""

from __future__ import annotations

import asyncio
import json
import sys

from rig import RunLog, call_tool
from tasks import load_task


async def smoke(task_id: str) -> None:
    task = load_task(task_id)
    run = f"smoke-{task_id}"
    RunLog(run).path.unlink(missing_ok=True)

    async def call(server: str, tool: str, args: dict) -> str:
        seen = await call_tool(server=server, tool=tool, args=args, run=run, client="smoke")
        print(f"--- {tool}\n{seen[:1200]}\n")
        return seen

    for search in task["searches"]:
        await call("dfo", "search", {"parts": search["parts"]})
    opened = json.loads(
        await call(
            "plan",
            "start_plan",
            {
                "brief": task["brief"],
                "asks": task["asks"],
                "searches": [{"query": s["query"], "parts": s["parts"]} for s in task["searches"]],
            },
        )
    )
    session = opened["plan_session"]
    proposed = json.loads(
        await call("plan", "propose_plan", {"plan_session": session, **task["reference_payload"]})
    )
    await call(
        "plan",
        "run_plan",
        {"plan_session": session, "version": proposed["version"], "wait_s": 40},
    )


if __name__ == "__main__":
    asyncio.run(smoke(sys.argv[1] if len(sys.argv) > 1 else "series-gdp-ghana"))
