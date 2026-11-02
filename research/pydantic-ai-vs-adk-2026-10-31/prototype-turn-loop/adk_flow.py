"""The same stub nodes as `loop.py`, as an ADK 2.13 `Workflow` with the shape of
dev's root workflow (`harness/graph/workflow.py`): `route` and `plan_start`
fold into `orchestrate`, and `switch` is left out.

`version="v2"` adds `vet` between `planner` and `approve`. With `guards=True`
the nodes carry dev's guards: `seed`'s `_turn_ran_before` (copied verbatim),
`approve`'s `awaiting_approve` flag, and on `vet` the same kind of guard a
deploy would have to add to it. With `guards=False` each node is the plain
version a first draft would write.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import runlog
from google.adk import Context, Event
from google.adk.apps import App
from google.adk.events import RequestInput
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from google.adk.workflow import START, Edge, Workflow, node

APP_NAME = "turn_loop_adk"
RUNTIME = "adk"
APPROVE_INTERRUPT = "approve"
MAX_ROUNDS = 5


def _turn_ran_before(ctx: Context) -> bool:
    """`harness.graph.seed._turn_ran_before`, verbatim."""
    own = ctx.node_path
    return any(
        event.invocation_id == ctx.invocation_id
        and event.node_info is not None
        and bool(event.node_info.path)
        and event.node_info.path != own
        and not event.node_info.path.startswith(f"{own}/")
        for event in ctx.session.events
    )


def _ran_in_invocation(ctx: Context, name: str) -> bool:
    """Whether node `name` already has an event in this invocation."""
    return any(
        event.invocation_id == ctx.invocation_id
        and event.node_info is not None
        and (event.node_info.path or "").split("/")[-1].split("@")[0] == name
        for event in ctx.session.events
    )


def build_runner(
    service: BaseSessionService, *, version: str, guards: bool, log: Path | None
) -> Runner:
    def ran(ctx: Context, name: str, **extra: Any) -> None:
        runlog.record(log, runtime=RUNTIME, turn=ctx.session.id, node=name, what="run", **extra)

    def seed(ctx: Context, node_input: Any) -> None:
        del node_input
        if guards and _turn_ran_before(ctx):
            runlog.record(log, runtime=RUNTIME, turn=ctx.session.id, node="seed", what="skip")
            return
        ran(ctx, "seed")

    def orchestrate(ctx: Context, node_input: Any) -> Event:
        del node_input
        ran(ctx, "orchestrate")
        if ctx.state.get("reply_ready"):
            return Event(output={"reply": "done"}, route="reply")
        if ctx.state.get("ask") and "clarified" not in ctx.state:
            return Event(output={}, route="ask")
        return Event(output={}, route="plan")

    async def ask(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        del node_input
        answer = ctx.resume_inputs.get("clarify")
        ran(ctx, "ask", answered=answer is not None)
        if answer is None:
            yield RequestInput(interrupt_id="clarify", message="Which years?")
            return
        ctx.state["clarified"] = answer
        yield Event(output={})

    def planner(ctx: Context, node_input: Any) -> Event:
        del node_input
        draft = ctx.state.get("round", 0) + 1
        if draft > MAX_ROUNDS:
            raise RuntimeError(f"planner drafted {draft} rounds in one turn")
        ran(ctx, "planner", round=draft)
        ctx.state["round"] = draft
        ctx.state["plan"] = {"round": draft}
        return Event(output={"plan": {"round": draft}}, route="ok")

    def vet(ctx: Context, node_input: Any) -> Any:
        if guards and _ran_in_invocation(ctx, "approve"):
            runlog.record(log, runtime=RUNTIME, turn=ctx.session.id, node="vet", what="skip")
            return node_input
        ran(ctx, "vet")
        ctx.state["vetted"] = ctx.state.get("round")
        return node_input

    async def approve(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        round_ = ctx.state.get("round")
        answer = ctx.resume_inputs.get(APPROVE_INTERRUPT)
        fresh = answer is None or (guards and not ctx.state.get("awaiting_approve"))
        if fresh:
            if guards:
                ctx.state["awaiting_approve"] = True
            ran(ctx, "approve", phase="offer", round=round_)
            yield RequestInput(
                interrupt_id=APPROVE_INTERRUPT, message="Approve?", payload=node_input
            )
            return
        if guards:
            ctx.state["awaiting_approve"] = False
        ran(
            ctx,
            "approve",
            phase="answer",
            round=round_,
            answer_round=answer.get("round"),
            decision=answer["decision"],
        )
        if guards and answer.get("round") != round_:
            # Dev's approve routes an answer naming another revision to amend.
            yield Event(output={"stale": answer}, route="amend")
            return
        yield Event(output={}, route="amend" if answer["decision"] == "amend" else "approved")

    async def execute(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        del node_input
        ran(ctx, "execute")
        for n in range(ctx.state["steps"]):
            runlog.record(
                log, runtime=RUNTIME, turn=ctx.session.id, node="execute", what="step_start", n=n
            )
            yield Event(custom_metadata={"step_start": n})
            await asyncio.sleep(ctx.state["step_s"])
            runlog.record(
                log, runtime=RUNTIME, turn=ctx.session.id, node="execute", what="step", n=n
            )
            yield Event(custom_metadata={"step": n})
        yield Event(output={}, route="ok")

    def plan_done(ctx: Context, node_input: Any) -> dict[str, Any]:
        del node_input
        ran(ctx, "plan_done")
        ctx.state["reply_ready"] = True
        return {}

    seed_n = node(seed, name="seed")
    orchestrate_n = node(orchestrate, name="orchestrate", rerun_on_resume=True)
    ask_n = node(ask, name="ask", rerun_on_resume=True)
    planner_n = node(planner, name="planner", rerun_on_resume=True)
    approve_n = node(approve, name="approve", rerun_on_resume=True)
    execute_n = node(execute, name="execute")
    plan_done_n = node(plan_done, name="plan_done")
    edges: list[Any] = [
        (START, seed_n),
        (seed_n, orchestrate_n),
        Edge(from_node=orchestrate_n, to_node=ask_n, route="ask"),
        Edge(from_node=orchestrate_n, to_node=planner_n, route="plan"),
        (ask_n, orchestrate_n),
        Edge(from_node=approve_n, to_node=execute_n, route="approved"),
        Edge(from_node=approve_n, to_node=planner_n, route="amend"),
        Edge(from_node=execute_n, to_node=plan_done_n, route="ok"),
        (plan_done_n, orchestrate_n),
    ]
    if version == "v2":
        vet_n = node(vet, name="vet")
        edges += [Edge(from_node=planner_n, to_node=vet_n, route="ok"), (vet_n, approve_n)]
    else:
        edges.append(Edge(from_node=planner_n, to_node=approve_n, route="ok"))
    workflow = Workflow(name=APP_NAME, edges=edges)
    return Runner(app=App(name=APP_NAME, root_agent=workflow), session_service=service)
