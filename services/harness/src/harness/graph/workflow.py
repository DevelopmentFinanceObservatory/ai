"""The turn's edges (`EDGES`) and the graph built over its nodes (`turn_graph`).

`harness.graph` describes the turn the edges make and what they guarantee.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from harness.graph.events import GraphEvent
from harness.graph.state import ORCHESTRATOR_SPECS_KEY
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
