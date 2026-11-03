"""An approved plan run on dev's own `execute` and `plan_done` nodes.

`dev_run_graph` builds the two nodes as `harness.graph.workflow.create_root_workflow`
builds them, wired `execute` to `plan_done` on the `"ok"` route, and runs them
under an ADK `Runner` on a session seeded from the plan session's state with the
approved plan under the graph's `plan` key. What `plan_done` yields is the
`AgentHandBack` dev's `orchestrate` node answers the orchestrator's `plan` call
with (`_call_result`); the run answers its `response()`, and the frames
`plan_done` registered (`ORCHESTRATOR_FRAMES_KEY`), each with the pair it is
described by and the pair the materialisation store holds it under.

An `execute` that routes `"replan"`, a refusal the planner would draft again
for, ends the run with that refusal, for the outside model to propose again.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from compiler.surface import ToolSurface
from executor.completion import ResultPublisher
from executor.ledger import RunLedger
from executor.run import FetchSeriesStats, LabelRefs, default_fetch_series_stats
from executor.run import default_label_refs as dev_label_refs
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Edge, Workflow, node
from google.genai import types
from harness.conversation import current_conversation_id
from harness.graph.execute import execute_node
from harness.graph.orchestrate import AgentHandBack
from harness.graph.plan_done import plan_done_node
from harness.graph.seams import Executor
from harness.graph.state import ORCHESTRATOR_FRAMES_KEY
from plan_tools import PlanSession
from planner.plan import Plan

_APP = "plan_mcp_run"
_RUN_MESSAGE = "Run the approved plan."


def dev_run_graph(
    *,
    executor: Executor,
    tool_surface: ToolSurface,
    run_ledger: RunLedger,
    publisher: ResultPublisher | None,
    max_concurrent_steps: int = 4,
    fetch_series_stats: FetchSeriesStats = default_fetch_series_stats,
    label_refs: LabelRefs = dev_label_refs,
) -> Callable[[PlanSession, Plan], Awaitable[dict[str, Any]]]:
    """The run `PlanningService` hands an approved plan to."""
    execute = node(
        execute_node(
            tool_surface=tool_surface,
            run_ledger=run_ledger,
            publisher=publisher,
            max_concurrent_steps=max_concurrent_steps,
            fetch_series_stats=fetch_series_stats,
            label_refs=label_refs,
        ),
        name="execute",
    )
    plan_done = node(plan_done_node(executor=executor, run_ledger=run_ledger), name="plan_done")
    workflow = Workflow(
        name="plan_run",
        edges=[(START, execute), Edge(from_node=execute, to_node=plan_done, route="ok")],
    )
    sessions = InMemorySessionService()
    runner = Runner(node=workflow, app_name=_APP, session_service=sessions)

    async def run(session: PlanSession, approved: Plan) -> dict[str, Any]:
        state = {**session.state, "plan": approved.model_dump_json(), "replans": 0}
        graph_session = await sessions.create_session(
            app_name=_APP, user_id=session.caller or "outside", state=state
        )
        token = current_conversation_id.set(session.session_id)
        outputs: dict[str, Any] = {}
        try:
            async for event in runner.run_async(
                user_id=graph_session.user_id,
                session_id=graph_session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=_RUN_MESSAGE)]
                ),
            ):
                if event.node_info is not None and event.output is not None:
                    outputs[_node_name(event)] = (event.output, event.actions.route)
        finally:
            current_conversation_id.reset(token)
        if "plan_done" in outputs:
            hand_back = AgentHandBack.model_validate(outputs["plan_done"][0])
            ended = await sessions.get_session(
                app_name=_APP, user_id=graph_session.user_id, session_id=graph_session.id
            )
            frames = dict((ended.state if ended else {}).get(ORCHESTRATOR_FRAMES_KEY) or {})
            return {"status": "finished", "hand_back": hand_back.response(), "frames": frames}
        output, routed = outputs.get("execute", (None, None))
        return {"status": "replan" if routed == "replan" else "stopped", "execute": output}

    return run


def _node_name(event: Any) -> str:
    """The workflow node an event came from, as `harness.studio_translate` reads it."""
    path = event.node_info.path if event.node_info else ""
    return path.split("/")[-1].split("@")[0]
