"""Serve plan-mcp over Streamable HTTP beside a running harness.

The planner, translation and executor reach data-mcp and analysis-mcp at the
harness's own mounts (`Settings.data_mcp_url`, `Settings.analysis_mcp_url`,
`http://localhost:8080/mcp/...` by default), with the bearer token and caller
headers `harness.mcp_client` sends. Search goes through `McpResolver`, the
translation is production's (`translator.tool.build_translate`), and an approved
plan runs on dev's `execute` and `plan_done` nodes with `McpExecutor`, its
derived results published through `McpExecutor.publish_result`, its runs kept in
memory.

    uv run --env-file .env --package harness python serve.py --port 8091
"""

from __future__ import annotations

import argparse

from execution import dev_run_graph
from executor.ledger import InMemoryRunLedger
from harness.agents import build_tool_surface
from harness.conversation import current_conversation_id
from harness.mcp_client import McpExecutor, McpResolver
from harness.provenance import ProvenanceStore
from harness.settings import Settings
from plan_tools import PlanningService
from server import PlanMcp
from translator.tool import build_translate


def build(settings: Settings) -> PlanMcp:
    """plan-mcp wired to the harness `settings` names."""
    surface = build_tool_surface(settings=settings)
    executor = McpExecutor(settings=settings, provenance=ProvenanceStore())
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    service = PlanningService(
        resolver=McpResolver(settings=settings),
        surface=surface,
        translate=build_translate(surface=surface),
        run_graph=dev_run_graph(
            executor=executor,
            tool_surface=surface,
            run_ledger=ledger,
            publisher=executor.publish_result,
            max_concurrent_steps=settings.executor_max_concurrent_steps,
        ),
    )
    return PlanMcp(service)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    build(Settings()).mcp.run(transport="http", host=args.host, port=args.port, path="/mcp")


if __name__ == "__main__":
    main()
