"""One task end to end by hand, with its reference plan: public `search`,
`start_plan` with the spec_ids it returned, `propose_plan` with the reference
plan (its one dataset pinned to the one spec_id, when it has one), and
`run_plan` approved on the client's word. Writes
`runs/smoke-<task>.jsonl`.

    python smoke.py series-gdp-ghana
"""

from __future__ import annotations

import asyncio
import json
import re
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

    spec_ids: list[str] = []
    for search in task["searches"]:
        seen = await call("dfo", "search", {"parts": search["parts"]})
        spec_ids += re.findall(r"\[(spec_[0-9a-f]{32})\]", seen)
    opened = json.loads(
        await call(
            "plan",
            "start_plan",
            {"brief": task["brief"], "asks": task["asks"], "spec_ids": spec_ids},
        )
    )
    session = opened["plan_session"]
    body = json.loads(json.dumps(task["reference_payload"]))
    datasets = [r for r in body["evidence_requirements"] if r["kind"] == "dataset"]
    if len(datasets) == 1 and len(spec_ids) == 1:
        datasets[0]["source_spec_id"] = spec_ids[0]
    proposed = json.loads(await call("plan", "propose_plan", {"plan_session": session, **body}))
    await call(
        "plan",
        "run_plan",
        {"plan_session": session, "version": proposed["version"], "approved": True, "wait_s": 40},
    )


if __name__ == "__main__":
    asyncio.run(smoke(sys.argv[1] if len(sys.argv) > 1 else "series-gdp-ghana"))
