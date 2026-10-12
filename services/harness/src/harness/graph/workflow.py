"""The root workflow: builds each node and wires the edges between them."""

from __future__ import annotations

from compiler.surface import ToolSurface
from executor.run import (
    FetchSeriesStats,
    default_fetch_series_stats,
)
from google.adk.agents import LlmAgent
from google.adk.workflow import START, Edge, Workflow, node

from harness.graph.execute import execute_node
from harness.graph.orchestrate import ask, orchestrate_node, route
from harness.graph.plan_done import plan_done_node
from harness.graph.planning import approve, plan_start, planner_node
from harness.graph.seams import (
    Executor,
    PlannerModel,
)
from harness.graph.seed import seed_node
from harness.graph.state import (
    AGENT_NAME,
    GraphState,
)


def create_root_workflow(
    *,
    planner: PlannerModel,
    orchestrator: LlmAgent,
    executor: Executor,
    tool_surface: ToolSurface | None,
    definition_surface: ToolSurface,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
) -> Workflow:
    """Build the root workflow with its external seams injected.

    The turn opens with `orchestrator`, the agent (`orchestrator.agent`, with
    `harness.orchestrator_tools`), after `seed` hands it any datasets the user
    brought from Search. Its `plan` tool reaches the `planner`, `approve` and
    `execute` nodes through `plan_start` and `plan_done`, and its `ask_user`
    tool reaches `ask`. `planner` is the model the `planner` node drafts plans
    with, for approval. `executor` is the recorder the nodes' own calls run
    through. `tool_surface` is the servers the `execute` node runs an approved
    workflow against; it carries no default, so a caller states its absence
    (`None`, refused loudly at `execute`) rather than falling into it
    unnoticed. `definition_surface` reaches the data-mcp the `seed` node
    resolves each seed's identity again through
    (`translator.discovery.resolve_definition`), a call the Activity feed
    does not show. `max_concurrent_steps` bounds how many of one plan's
    wave's tool calls the `execute` node's run may hold in flight against
    `tool_surface` at once (`executor.run.execute_approved_plan`'s own
    keyword of the same name). `fetch_series_stats` is that same call's own
    keyword: the real signed-URL fetch by default, injectable for a caller
    whose `tool_surface` is itself fully virtualised.
    """
    receive_seeds = seed_node(executor=executor, definition_surface=definition_surface)
    orchestrate = orchestrate_node(orchestrator=orchestrator)
    draft_plan = planner_node(planner=planner)
    execute = execute_node(
        tool_surface=tool_surface,
        max_concurrent_steps=max_concurrent_steps,
        fetch_series_stats=fetch_series_stats,
    )
    plan_done = plan_done_node(executor=executor)

    seed_n = node(receive_seeds, name="seed")
    orchestrate_n = node(orchestrate, name="orchestrate", rerun_on_resume=True)
    route_n = node(route, name="route")
    ask_n = node(ask, name="ask", rerun_on_resume=True)
    plan_start_n = node(plan_start, name="plan_start")
    # `rerun_on_resume=True`: a real `PlannerModel` seam
    # (`harness.planner_seam.PlannerSeam`) calls `ctx.run_node` from inside
    # this node, and ADK raises `ValueError` at that call unless the
    # *calling* node itself carries this flag (`agents/context.py`'s
    # `_run_node_internal`).
    planner_n = node(draft_plan, name="planner", rerun_on_resume=True)
    approve_n = node(approve, name="approve", rerun_on_resume=True)
    execute_n = node(execute, name="execute")
    plan_done_n = node(plan_done, name="plan_done")
    return Workflow(
        name=AGENT_NAME,
        state_schema=GraphState,
        edges=[
            (START, seed_n),
            (seed_n, orchestrate_n),
            (orchestrate_n, route_n),
            Edge(from_node=route_n, to_node=ask_n, route="ask"),
            (ask_n, orchestrate_n),
            Edge(from_node=route_n, to_node=plan_start_n, route="plan"),
            (plan_start_n, planner_n),
            Edge(from_node=planner_n, to_node=approve_n, route=["ok", "pending_refusal"]),
            Edge(from_node=planner_n, to_node=plan_done_n, route="refused"),
            Edge(from_node=approve_n, to_node=execute_n, route="approved"),
            Edge(from_node=approve_n, to_node=planner_n, route="amend"),
            Edge(from_node=execute_n, to_node=planner_n, route="replan"),
            Edge(from_node=execute_n, to_node=plan_done_n, route="ok"),
            (plan_done_n, orchestrate_n),
        ],
    )
