"""Serve plan-mcp over Streamable HTTP beside a running harness.

The planner, translation and executor reach data-mcp and analysis-mcp at the
harness's own mounts (`Settings.data_mcp_url`, `Settings.analysis_mcp_url`,
`http://localhost:8080/mcp/...` by default), with the bearer token and caller
headers `harness.mcp_client` sends. Search goes through `McpResolver`, the
translation is production's (`translator.tool.build_translate`), and an approved
plan runs on dev's `execute` and `plan_done` nodes with `McpExecutor`, its
derived results published through `McpExecutor.publish_result`, its runs kept in
memory.

A spec_id a client's `search` returned is read as `get_data` reads a bare id
(`held_identity`), a run's rows as `make_charts` reads a frame
(`harness.orchestrator_tools._frame_work`), and sessions persist in the SQLite
file `--sessions` names.

    uv run --env-file .env --package harness python serve.py --port 8091
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from common.materialisation.factory import install_store_if_unset
from data_mcp.client_holder import get_client
from data_mcp.deadline import CallDeadline
from data_mcp.definition_store import DefinitionStoreUnavailable, get_store
from data_mcp.errors import UpstreamError
from data_mcp.settings import Settings as DataSettings
from execution import dev_run_graph
from executor.ledger import InMemoryRunLedger
from harness.agents import build_tool_surface
from harness.conversation import current_conversation_id
from harness.mcp_client import McpExecutor, McpResolver
from harness.orchestrator_tools import _frame_work
from harness.provenance import ProvenanceStore
from harness.settings import Settings
from plan_tools import DefinitionUnavailableError, PlanningService
from server import PlanMcp
from store import SqliteSessions
from translator.tool import build_translate

# The registry read `get_data` makes for an id its definition store does not
# hold (`data_mcp.get_data_tool._registry_identity_or_refuse`), and its deadline.
_OPTIONS_ROUTE = "/api/datasets/{id}/options"
_REGISTRY_DEADLINE_S = 20.0
_SCHEMA_VERSION = 1
DEFAULT_SESSIONS = Path(__file__).resolve().parent / "live" / "plan-sessions.sqlite3"


async def held_identity(spec_id: str) -> Mapping[str, Any] | None:
    """The identity of `spec_id` as `get_data` finds it for a bare id: data-mcp's
    definition store, else dfo-apis' registry, which holds every definition Search
    minted. `None` when neither holds it, or the registry serves it under another
    spec_id."""
    try:
        stored = await get_store().get(spec_id, schema_version=_SCHEMA_VERSION)
    except DefinitionStoreUnavailable:
        stored = None
    if stored is not None:
        return json.loads(stored.canonical_bytes)
    try:
        response = await get_client().request(
            "GET",
            f"/api/datasets/{spec_id}/options",
            route=_OPTIONS_ROUTE,
            deadline=CallDeadline.after(_REGISTRY_DEADLINE_S),
        )
    except UpstreamError as exc:
        if exc.code == "spec_not_found" or exc.status == 404:
            return None
        raise DefinitionUnavailableError(str(exc)) from exc
    definition = response.json().get("definition") or {}
    if definition.get("spec_id") != spec_id:
        return None
    return definition.get("identity")


async def frame_rows(ref: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The rows of the frame `ref` names, from the materialisation store or, for a
    derived frame, from dfo-apis' analysis runner."""
    return await _frame_work(ref, lambda read: read.rows)


def build(settings: Settings, *, sessions: Path | None = None) -> PlanMcp:
    """plan-mcp wired to the harness `settings` names, with sessions in `sessions`."""
    surface = build_tool_surface(settings=settings)
    executor = McpExecutor(settings=settings, provenance=ProvenanceStore())
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    service = PlanningService(
        resolver=McpResolver(settings=settings),
        surface=surface,
        translate=build_translate(surface=surface),
        definitions=held_identity,
        store=SqliteSessions(sessions) if sessions is not None else None,
        read_rows=frame_rows,
        run_graph=dev_run_graph(
            executor=executor,
            tool_surface=surface,
            run_ledger=ledger,
            publisher=executor.publish_result,
            max_concurrent_steps=settings.executor_max_concurrent_steps,
        ),
    )
    return PlanMcp(service)


def install_frame_store() -> None:
    """The materialisation store data-mcp serves frames from, as its own lifespan
    installs it: the shared bucket when one is configured, so this process reads
    the harness's frames."""
    data = DataSettings()
    install_store_if_unset(
        bucket_name=data.frame_store_bucket,
        verify_content_version_on_read=data.frame_store_verify_content_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    args = parser.parse_args()
    install_frame_store()
    build(Settings(), sessions=args.sessions).mcp.run(
        transport="http", host=args.host, port=args.port, path="/mcp"
    )


if __name__ == "__main__":
    main()
