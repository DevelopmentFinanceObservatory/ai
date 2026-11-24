"""Turns that run a plan on the turn loop, for tests.

`plan_nodes` builds every node a plan call passes through: `seed`,
`orchestrate` and `ask` (`tests.graph_turns.opening_nodes`), then
`plan_start`, `planner`, `approve`, `switch`, `execute` and `plan_done`,
each the real node, the planner drafting and revising with a
`ScriptedPlannerModel` by default, and `seed` carrying runs on through the
real run resumer. Build the graph with `tests.graph_turns.opening_graph`
and run it with `tests.turn_driver.TurnDriver`.

`approve_answer` is the answer approving a card, and `approve_turn` runs a
question to its card and approves it on the same instance.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from compiler.surface import ToolSurface
from executor.completion import ResultPublisher
from executor.ledger import RunLedger
from executor.run import default_label_refs
from harness.early_mint import EarlyMints
from harness.graph import planning
from harness.graph.approve import approve_node
from harness.graph.execute import execute_node
from harness.graph.interfaces import Executor, PlannerModel
from harness.graph.plan_done import plan_done_node
from harness.graph.resume import run_resumer
from harness.graph.switch import switch_node
from harness.scripted import ScriptedPlannerModel, scripted_publish_result
from harness.scripted_plan import scripted_fetch_series_stats
from harness.turn.node import Node
from orchestrator.agent import Orchestrator

from .data_mcp_definitions import DataMcpDefinitions
from .graph_turns import opening_nodes
from .turn_driver import ScriptedModels, SegmentRun, TurnDriver


def approve_answer(revision: object, *, at: datetime | None = None) -> dict[str, Any]:
    """An answer approving the plan card that names `revision`, given at `at`
    (now by default), as the studio's interrupt route builds it."""
    return {"approved": revision, "at": (at or datetime.now(UTC)).isoformat()}


def plan_nodes(
    orchestrator: Orchestrator,
    models: ScriptedModels,
    *,
    run_ledger: RunLedger,
    tool_surface: ToolSurface | None,
    executor: Executor,
    planner: PlannerModel | None = None,
    publisher: ResultPublisher | None = scripted_publish_result,
    early_mints: EarlyMints | None = None,
    definition_surface: ToolSurface | None = None,
) -> dict[str, Node]:
    """Every node a plan call passes through: `orchestrate` running
    `orchestrator` on its scripted model, `planner` drafting with `planner`,
    and the real `approve` (offering to `early_mints`), `switch` (revising
    with `planner`), `execute` (running against `tool_surface`, publishing
    through `publisher`) and `plan_done` (describing through `executor`),
    every run kept in `run_ledger`, whose unfinished runs `seed` carries on.
    `seed` resolves seeds and `switch` posts switches through
    `definition_surface`, a data-mcp over a Search that serves nothing when
    `None`."""
    planner = planner or ScriptedPlannerModel()
    definition_surface = definition_surface or DataMcpDefinitions()
    resume = run_resumer(
        run_ledger=run_ledger,
        publisher=publisher,
        executor=executor,
        tool_surface=tool_surface,
        max_concurrent_steps=4,
        fetch_series_stats=scripted_fetch_series_stats,
        label_refs=default_label_refs,
    )
    return {
        **opening_nodes(
            orchestrator,
            models,
            executor=executor,
            definition_surface=definition_surface,
            resume=resume,
        ),
        "plan_start": planning.plan_start,
        "planner": planning.planner_node(planner=planner),
        "approve": approve_node(early_mints=early_mints),
        "switch": switch_node(planner=planner, definition_surface=definition_surface),
        "execute": execute_node(
            tool_surface=tool_surface,
            run_ledger=run_ledger,
            publisher=publisher,
            max_concurrent_steps=4,
            fetch_series_stats=scripted_fetch_series_stats,
            label_refs=default_label_refs,
        ),
        "plan_done": plan_done_node(executor=executor, run_ledger=run_ledger),
    }


async def approve_turn(driver: TurnDriver, message: str, *, turn_index: int = 1) -> SegmentRun:
    """Run `message` as turn `turn_index` to its plan card and approve the
    card; the segment that ran the approved plan."""
    paused = await driver.turn(message, turn_index=turn_index)
    card = paused.pause
    assert card.kind == "approve", card
    return await driver.answer(approve_answer(card.payload["revision"]), to=card)
