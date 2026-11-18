"""The turn's edges (`EDGES`), the graph built over its nodes (`turn_graph`),
and the Studio turn's graph with every node built (`studio_graph`).

`harness.graph` describes the turn the edges make and what they guarantee.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import FunctionType, MappingProxyType
from typing import Any

from compiler.surface import ToolSurface
from executor.completion import ResultPublisher
from executor.ledger import RunLedger
from executor.run import (
    FetchSeriesStats,
    LabelRefs,
    default_fetch_series_stats,
    default_label_refs,
)
from orchestrator.agent import Orchestrator
from pydantic_ai.models import Model

from harness.early_mint import EarlyMints
from harness.graph import planning
from harness.graph.approve import approve_node
from harness.graph.events import GraphEvent
from harness.graph.execute import execute_node
from harness.graph.interfaces import Executor, PlannerModel
from harness.graph.orchestrate import ask, orchestrate_node
from harness.graph.plan_done import plan_done_node
from harness.graph.resume import run_resumer
from harness.graph.seed import chip_fetch, seed_node
from harness.graph.state import ORCHESTRATOR_SPECS_KEY
from harness.graph.switch import switch_node
from harness.turn.loop import END, Graph
from harness.turn.node import Next, Node, NodeCtx, Pause

# The node every turn opens at.
START = "seed"

# The node each edge `(node, label)` reaches. A node with one way on leaves
# along `next`.
EDGES: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        ("seed", "next"): "orchestrate",
        ("orchestrate", "plan"): "plan_start",
        ("orchestrate", "ask"): "ask",
        ("orchestrate", "answered"): END,
        ("ask", "next"): "orchestrate",
        ("plan_start", "next"): "planner",
        ("planner", "ok"): "approve",
        ("planner", "pending_refusal"): "approve",
        ("planner", "refused"): "plan_done",
        ("approve", "approved"): "execute",
        ("approve", "amend"): "planner",
        ("approve", "switch"): "switch",
        ("switch", "offer"): "approve",
        ("execute", "replan"): "planner",
        ("execute", "ok"): "plan_done",
        ("plan_done", "next"): "orchestrate",
    }
)


@contextlib.contextmanager
def _specs_in_state(ctx: NodeCtx[Any]) -> Iterator[None]:
    """Hold the turn's spec blocks under `ORCHESTRATOR_SPECS_KEY` in
    `ctx.state` while the block runs, and put what the key holds after it
    back in `ctx.specs`, in order."""
    ctx.state[ORCHESTRATOR_SPECS_KEY] = dict(ctx.specs)
    try:
        yield
    finally:
        specs = dict(ctx.state.pop(ORCHESTRATOR_SPECS_KEY, None) or {})
        ctx.specs.clear()
        ctx.specs.update(specs)


def with_specs[E](node: Node[E]) -> Node[E]:
    """`node`, reading and writing the turn's spec blocks in its state.

    The nodes and the orchestrator's tools read and write the turn's spec
    blocks under `ORCHESTRATOR_SPECS_KEY`. The loop stores them apart from
    the state, one row per spec id (`harness.turn.store`), so the key holds
    them only while a node runs.
    """

    async def run(ctx: NodeCtx[E], node_input: Any) -> Next | Pause:
        with _specs_in_state(ctx):
            return await node.run(ctx, node_input)

    if node.resume is None:
        return Node(run=run)
    resume = node.resume

    async def resumed(ctx: NodeCtx[E], node_input: Any, answer: dict[str, Any]) -> Next | Pause:
        with _specs_in_state(ctx):
            return await resume(ctx, node_input, answer)

    return Node(run=run, resume=resumed)


def turn_graph(nodes: Mapping[str, Node[GraphEvent]]) -> Graph[GraphEvent]:
    """The turn's graph over `nodes`, by name, each holding the turn's spec
    blocks in its state (`with_specs`), with `EDGES` between them.

    Raises:
        ValueError: An edge joins a node `nodes` lacks.
    """
    return Graph(
        start=START,
        nodes={name: with_specs(node) for name, node in nodes.items()},
        edges=EDGES,
    )


def studio_graph(
    *,
    planner: PlannerModel,
    orchestrator: Orchestrator,
    model: Callable[[], Model],
    tools: Sequence[FunctionType],
    executor: Executor,
    tool_surface: ToolSurface | None,
    definition_surface: ToolSurface,
    publisher: ResultPublisher | None,
    run_ledger: RunLedger,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    label_refs: LabelRefs = default_label_refs,
    early_mints: EarlyMints | None = None,
) -> Graph[GraphEvent]:
    """The Studio turn's graph (`turn_graph`) with every node built.

    `orchestrate` runs `orchestrator` on the model `model` builds for each
    run, with `tools`, the functions the agent was built on, among which
    `seed` finds the `fetch` a chip's data comes through (`chip_fetch`).
    `planner` drafts plans at `planner` and revises them at `switch`.
    `executor` is the recorder the nodes' own calls run through.
    `tool_surface` is the servers `execute` runs an approved workflow
    against; with `None`, `execute` refuses and `seed` carries no run on.
    `run_ledger` keeps every run `execute` starts, and `seed` carries on the
    unfinished ones (`harness.graph.resume`). `publisher` publishes each
    run's derived results with their replay recipes before it completes
    (`executor.completion`); with `None`, a run with a derived result never
    completes. `definition_surface` reaches the data-mcp `seed` resolves
    each seed's identity again through, and that `switch` and the chip post
    a switched scope choice through. `max_concurrent_steps`,
    `fetch_series_stats` and `label_refs` are
    `executor.run.execute_approved_plan`'s keywords of those names.
    `early_mints` starts each card's retrievals when `approve` offers it
    (`harness.early_mint`); with `None` nothing is fetched before approval.
    """
    nodes: dict[str, Node[GraphEvent]] = {
        "seed": seed_node(
            executor=executor,
            definition_surface=definition_surface,
            fetch=chip_fetch(list(tools)),
            resume=run_resumer(
                run_ledger=run_ledger,
                publisher=publisher,
                executor=executor,
                tool_surface=tool_surface,
                max_concurrent_steps=max_concurrent_steps,
                fetch_series_stats=fetch_series_stats,
                label_refs=label_refs,
            ),
        ),
        "orchestrate": orchestrate_node(orchestrator=orchestrator, model=model),
        "ask": ask,
        "plan_start": planning.plan_start,
        "planner": planning.planner_node(planner=planner),
        "approve": approve_node(early_mints=early_mints),
        "switch": switch_node(planner=planner, definition_surface=definition_surface),
        "execute": execute_node(
            tool_surface=tool_surface,
            run_ledger=run_ledger,
            publisher=publisher,
            max_concurrent_steps=max_concurrent_steps,
            fetch_series_stats=fetch_series_stats,
            label_refs=label_refs,
        ),
        "plan_done": plan_done_node(executor=executor, run_ledger=run_ledger),
    }
    return turn_graph(nodes)
