"""Turns on the graph's first nodes, `seed`, `orchestrate` and `ask`, for tests.

`opening_graph` builds a test-only graph over the real nodes: the edges of
`harness.graph.workflow.EDGES` that leave them, with each edge that reaches a
node the graph lacks ending the turn there, so a `plan` hand-off ends the turn
on `orchestrate`'s `plan` edge. A `given` node opens it when a test hands the
turn spec blocks or state an earlier part of the turn would have written,
since a turn opens with none. Run it with `tests.turn_driver.TurnDriver`.

`agent_read` reads a request the orchestrator sent back as the agent reads it:
each text part, and each hand-back as its text, its structured fields and its
plan record, one per line.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from compiler.surface import ToolSurface
from google.genai import types
from harness.graph.interfaces import Executor
from harness.graph.orchestrate import ask_node, orchestrate_node
from harness.graph.resume import ResumeRuns
from harness.graph.seed import Fetch, seed_node
from harness.graph.state import ORCHESTRATOR_SPECS_KEY
from harness.graph.workflow import EDGES, START
from harness.turn.events import NodeDone, TurnEvent
from harness.turn.loop import END, Graph
from harness.turn.node import Next, Node, NodeCtx, TurnKey
from orchestrator.agent import Orchestrator

from .data_mcp_definitions import DataMcpDefinitions
from .scripted_genai import Request
from .scripted_tools import no_executor
from .turn_driver import ScriptedModels

GIVEN = "given"


def given(
    *, specs: Mapping[str, Any] | None = None, state: Mapping[str, Any] | None = None
) -> Node:
    """A node that writes `specs` and `state` into the turn, as its earlier
    searches would have, and goes on to `seed`."""

    async def run(ctx: NodeCtx, node_input: Any) -> Next:
        del node_input
        ctx.state[ORCHESTRATOR_SPECS_KEY] = {
            **ctx.state[ORCHESTRATOR_SPECS_KEY],
            **json.loads(json.dumps(dict(specs or {}))),
        }
        ctx.state.update(json.loads(json.dumps(dict(state or {}))))
        return Next("next")

    return Node(run=run)


def no_carried(_key: TurnKey) -> None:
    """A conversation store that holds no carried conversation, for graphs
    whose turns open on the digest."""


def opening_nodes(
    orchestrator: Orchestrator,
    models: ScriptedModels,
    *,
    executor: Executor = no_executor,
    definition_surface: ToolSurface | None = None,
    fetch: Fetch | None = None,
    resume: ResumeRuns | None = None,
) -> dict[str, Node]:
    """`seed`, `orchestrate` and `ask`, `orchestrate` running `orchestrator`
    on the orchestrator's scripted model, `seed` resolving seeds through
    `definition_surface` (a data-mcp over a Search that serves nothing by
    default), recording steps through `executor`, applying a chip with
    `fetch` and carrying runs on with `resume`. `ask` re-reads a unit answer's
    parts through `definition_surface` too."""
    surface = definition_surface or DataMcpDefinitions()
    return {
        "seed": seed_node(
            executor=executor,
            definition_surface=surface,
            resume=resume,
            fetch=fetch,
        ),
        "orchestrate": orchestrate_node(
            orchestrator=orchestrator, model=models.orchestrator_model, carried=no_carried
        ),
        "ask": ask_node(definition_surface=surface),
    }


def opening_graph(nodes: Mapping[str, Node], *, before: Node | None = None) -> Graph:
    """A graph over `nodes` with the edges of `EDGES` that leave them, an
    edge to a node `nodes` lacks ending the turn. With `before`, the turn
    opens at it, as `GIVEN`, and goes on to `START`."""
    edges = {
        (source, label): target if target == END or target in nodes else END
        for (source, label), target in EDGES.items()
        if source in nodes
    }
    every = dict(nodes)
    start = START
    if before is not None:
        every[GIVEN] = before
        edges[(GIVEN, "next")] = START
        start = GIVEN
    return Graph(start=start, nodes=every, edges=edges)


def visited(events: list[TurnEvent]) -> list[str]:
    """The nodes that finished, in order, `GIVEN` left out."""
    return [event.node for event in events if isinstance(event, NodeDone) and event.node != GIVEN]


# The calls the `orchestrate` node answers with a hand-back.
_HANDED_BACK_TO = frozenset({"plan", "ask_user"})


def agent_read(request: Request) -> str:
    """Every text part of `request` and every hand-back it shows as the
    result of a `plan` or `ask_user` call (`hand_back_text`), one per line."""
    return "\n".join(
        _part_text(part) for content in request.content_list for part in content.parts or []
    )


def _part_text(part: types.Part) -> str:
    response = part.function_response
    if response is not None and response.name in _HANDED_BACK_TO:
        return hand_back_text(response.response or {})
    return part.text or ""


def hand_back_text(result: Mapping[str, Any]) -> str:
    """A hand-back's result (`harness.graph.orchestrate.AgentHandBack.response`)
    as lines: its text, its structured fields as one JSON object, then its
    plan record."""
    if "result" not in result:
        return ""
    fields = {key: value for key, value in result.items() if key not in ("result", "plan_record")}
    lines = [str(result["result"])]
    if fields:
        lines.append(json.dumps(fields))
    if "plan_record" in result:
        lines.append(str(result["plan_record"]))
    return "\n".join(lines)
