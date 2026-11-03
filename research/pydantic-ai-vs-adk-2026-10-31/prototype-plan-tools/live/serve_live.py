"""plan-mcp over Streamable HTTP beside a running harness, behind the harness's bearer
gate, recording each accepted plan for the scorer.

`serve.build` wires plan-mcp to the harness's data-mcp and analysis-mcp
mounts, whose clients (`harness.mcp_client`) send `common.auth.bearer_token()`.
That token resolves only once `apply_gate` runs, as `harness.server.create_app`
runs it for every mount; without it every call to the mounts answers 401. This
runs it on plan-mcp too, so plan-mcp asks the same bearer the public `/mcp`
asks.

Each accepted `propose_plan` appends the plan on record to
`runs/plans.jsonl`, keyed on the plan session and version: an amendment's
arguments carry only what changed, so only the stored plan says what kinds an
accepted version uses. The client never sees this record.

    DFO_APIS_URL=http://localhost:8083 MCP_API_TOKEN=local-dev \
        uv run --env-file .env --package harness python live/serve_live.py --port 8091
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from common.auth import apply_gate
from fastmcp import FastMCP
from harness.settings import Settings
from plan_tools import PlanningService, PlanSession
from planner.plan import load_plan
from serve import build

PLANS = Path(__file__).resolve().parent / "runs" / "plans.jsonl"


def record_accepted_plans(service: PlanningService, path: Path = PLANS) -> None:
    """Append each plan `service.propose` accepts to `path`."""
    propose = service.propose

    async def recording(session: PlanSession, arguments: Mapping[str, Any]) -> dict[str, Any]:
        answer = await propose(session, arguments)
        plan = load_plan(session.state)
        if "summary" in answer and plan is not None:
            line = {
                "plan_session": session.session_id,
                "caller": session.caller,
                "version": plan.version,
                "kinds": sorted({r.kind for r in plan.evidence_requirements}),
                "plan": plan.model_dump(mode="json", exclude={"translation", "workflow"}),
            }
            path.parent.mkdir(exist_ok=True)
            with path.open("a") as out:
                out.write(json.dumps(line) + "\n")
        return answer

    service.propose = recording  # ty: ignore[invalid-assignment]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    settings = Settings()
    # The harness clients read the token when built, so the process-wide policy
    # resolves first, on a server that is never served; plan-mcp's own gate
    # then takes the resolved policy.
    gate = {"token": settings.mcp_api_token, "disabled": settings.mcp_auth_disabled}
    apply_gate(FastMCP("auth-policy"), **gate)
    plan = build(settings)
    record_accepted_plans(plan.service)
    apply_gate(plan.mcp, **gate)
    plan.mcp.run(transport="http", host=args.host, port=args.port, path="/mcp")


if __name__ == "__main__":
    main()
