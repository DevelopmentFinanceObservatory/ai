"""The own turn loop: an enum of states, one async function per state, and a
stored cursor saying where the turn stopped.

The nodes are stubs with the shape of dev's root workflow (`seed`,
`orchestrate`, `ask`, `planner`, `approve`, `execute`, `plan_done`, with the
same loop-back edges). Each logs its runs to the shared run log. `V2` adds
`vet` between `planner` and `approve`, standing for a node a deploy adds.

`drive` dispatches on the cursor's state until the turn pauses or ends,
moving the cursor after every node. `run_turn` wraps it with the lease
heartbeat and the cross-instance cancel (`LISTEN` on the cancel channel).

The controls are keyword arguments: `from_start` ignores the cursor and
replays from `seed`; `checkpoint_between_pauses=False` writes the cursor and
records only when the turn pauses or ends; `cross_instance_cancel=False` leaves only the
in-process stop.
"""

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import asyncpg
import runlog
from store import CANCEL_CHANNEL, DONE, Cursor, Store

RUNTIME = "own"


class S(StrEnum):
    SEED = "seed"
    ORCHESTRATE = "orchestrate"
    ASK = "ask"
    PLANNER = "planner"
    VET = "vet"
    APPROVE = "approve"
    EXECUTE = "execute"
    PLAN_DONE = "plan_done"
    DONE = DONE


EDGES_V1: dict[tuple[S, str], S] = {
    (S.SEED, "next"): S.ORCHESTRATE,
    (S.ORCHESTRATE, "plan"): S.PLANNER,
    (S.ORCHESTRATE, "ask"): S.ASK,
    (S.ORCHESTRATE, "reply"): S.DONE,
    (S.ASK, "next"): S.ORCHESTRATE,
    (S.PLANNER, "ok"): S.APPROVE,
    (S.APPROVE, "approved"): S.EXECUTE,
    (S.APPROVE, "amend"): S.PLANNER,
    (S.EXECUTE, "ok"): S.PLAN_DONE,
    (S.PLAN_DONE, "next"): S.ORCHESTRATE,
}
EDGES_V2 = EDGES_V1 | {(S.PLANNER, "ok"): S.VET, (S.VET, "next"): S.APPROVE}
EDGES = {"v1": EDGES_V1, "v2": EDGES_V2}


@dataclass
class Route:
    label: str
    round: int | None = None


@dataclass
class Pause:
    interrupt_id: str
    payload: dict[str, Any]


@dataclass
class NodeCtx:
    store: Store
    cursor: Cursor
    owner: str
    log: Path | None
    answer: dict[str, Any] | None
    emit: Callable[[dict[str, Any]], None]
    checkpoint_between_pauses: bool
    stop: asyncio.Event
    data: dict[str, Any] = field(default_factory=dict)
    records: dict[str, Any] = field(default_factory=dict)

    def ran(self, node: str, **extra: Any) -> None:
        runlog.record(
            self.log, runtime=RUNTIME, turn=self.cursor.turn_id, node=node, what="run", **extra
        )

    async def checkpoint(self) -> None:
        """Write the records where the cursor stands, under the lease."""
        self.cursor = await self.store.advance(
            self.cursor,
            owner=self.owner,
            state=self.cursor.state,
            round=self.cursor.round,
            data=self.data,
            records=self.records,
        )


type Node = Callable[[NodeCtx], Awaitable[Route | Pause]]


async def seed(ctx: NodeCtx) -> Route:
    ctx.ran("seed")
    return Route("next")


async def orchestrate(ctx: NodeCtx) -> Route:
    ctx.ran("orchestrate")
    if ctx.data.get("reply_ready"):
        return Route("reply")
    if ctx.data.get("ask") and "clarified" not in ctx.data:
        return Route("ask")
    return Route("plan")


async def ask(ctx: NodeCtx) -> Route | Pause:
    ctx.ran("ask", answered=ctx.answer is not None)
    if ctx.answer is None:
        return Pause("clarify", {"question": "Which years?"})
    ctx.data["clarified"] = ctx.answer
    return Route("next")


async def planner(ctx: NodeCtx) -> Route:
    draft = ctx.cursor.round + 1
    ctx.ran("planner", round=draft)
    history = ctx.records.get("planner", {"drafts": []})
    ctx.records["planner"] = {"drafts": [*history["drafts"], draft]}
    ctx.data["plan"] = {"round": draft}
    return Route("ok", round=draft)


async def vet(ctx: NodeCtx) -> Route:
    ctx.ran("vet")
    ctx.data["vetted"] = ctx.cursor.round
    return Route("next")


async def approve(ctx: NodeCtx) -> Route | Pause:
    if ctx.answer is None:
        ctx.ran("approve", phase="offer", round=ctx.cursor.round)
        return Pause("approve", {"plan": ctx.data["plan"]})
    ctx.ran(
        "approve",
        phase="answer",
        round=ctx.cursor.round,
        answer_round=ctx.answer.get("round"),
        decision=ctx.answer["decision"],
    )
    return Route("amend" if ctx.answer["decision"] == "amend" else "approved")


async def execute(ctx: NodeCtx) -> Route:
    """Run `steps` steps of `step_s` seconds, checkpointing each finished step
    when `checkpoint_between_pauses`. Stops between steps on the in-process flag."""
    ctx.ran("execute")
    done: list[int] = list(ctx.records.get("execute", {}).get("done", []))
    for n in range(ctx.data["steps"]):
        if n in done:
            continue
        if ctx.stop.is_set():
            raise asyncio.CancelledError
        runlog.record(
            ctx.log,
            runtime=RUNTIME,
            turn=ctx.cursor.turn_id,
            node="execute",
            what="step_start",
            n=n,
        )
        ctx.emit({"type": "step_start", "n": n})
        await asyncio.sleep(ctx.data["step_s"])
        done.append(n)
        ctx.records["execute"] = {"done": done}
        runlog.record(
            ctx.log, runtime=RUNTIME, turn=ctx.cursor.turn_id, node="execute", what="step", n=n
        )
        if ctx.checkpoint_between_pauses:
            await ctx.checkpoint()
        ctx.emit({"type": "step", "n": n})
    return Route("ok")


async def plan_done(ctx: NodeCtx) -> Route:
    ctx.ran("plan_done")
    ctx.data["reply_ready"] = True
    return Route("next")


NODES: dict[S, Node] = {
    S.SEED: seed,
    S.ORCHESTRATE: orchestrate,
    S.ASK: ask,
    S.PLANNER: planner,
    S.VET: vet,
    S.APPROVE: approve,
    S.EXECUTE: execute,
    S.PLAN_DONE: plan_done,
}


@dataclass
class Options:
    version: str = "v1"
    from_start: bool = False
    checkpoint_between_pauses: bool = True
    cross_instance_cancel: bool = True


async def drive(
    store: Store,
    cursor: Cursor,
    *,
    owner: str,
    options: Options,
    log: Path | None,
    emit: Callable[[dict[str, Any]], None],
    stop: asyncio.Event,
    answer: dict[str, Any] | None = None,
) -> Cursor:
    """Dispatch on the cursor's state until the turn pauses or ends. `answer`
    goes to the state the cursor stood at, the one that paused."""
    edges = EDGES[options.version]
    answer_state = S(cursor.state)
    state = S.SEED if options.from_start else S(cursor.state)
    while state is not S.DONE:
        ctx = NodeCtx(
            store=store,
            cursor=cursor,
            owner=owner,
            log=log,
            answer=answer if state is answer_state else None,
            emit=emit,
            checkpoint_between_pauses=options.checkpoint_between_pauses,
            stop=stop,
            data=dict(cursor.data),
            records=dict(cursor.records),
        )
        if state is answer_state:
            answer = None
        result = await NODES[state](ctx)
        if isinstance(result, Pause):
            return await store.advance(
                ctx.cursor,
                owner=owner,
                state=state,
                round=ctx.cursor.round,
                data=ctx.data,
                records=ctx.records,
                pending=result.interrupt_id,
            )
        state = edges[(state, result.label)]
        round_ = result.round if result.round is not None else ctx.cursor.round
        if options.checkpoint_between_pauses or state is S.DONE:
            cursor = await store.advance(
                ctx.cursor,
                owner=owner,
                state=state,
                round=round_,
                data=ctx.data,
                records=ctx.records,
            )
        else:
            cursor = dataclasses.replace(
                ctx.cursor, state=state, round=round_, data=ctx.data, records=ctx.records
            )
    await store.release(cursor.turn_id, owner=owner)
    return cursor


@dataclass
class Outcome:
    kind: str  # "paused", "done" or "stopped"
    cursor: Cursor | None
    stopped_at: float | None = None


async def run_turn(
    store: Store,
    cursor: Cursor,
    *,
    dsn: str,
    owner: str,
    options: Options,
    log: Path | None,
    emit: Callable[[dict[str, Any]], None],
    answer: dict[str, Any] | None = None,
) -> Outcome:
    """`drive` under the lease heartbeat and, with `cross_instance_cancel`, a
    `LISTEN` on the cancel channel. A cancel or a lost lease cancels the
    running node; finished steps stay checkpointed and the turn stays
    resumable where its cursor stands."""
    stop = asyncio.Event()
    task = asyncio.create_task(
        drive(
            store,
            cursor,
            owner=owner,
            options=options,
            log=log,
            emit=emit,
            stop=stop,
            answer=answer,
        )
    )

    def cancel_on(conn: object, pid: int, channel: str, payload: str) -> None:
        del conn, pid, channel
        if payload == cursor.turn_id:
            task.cancel()

    listener: asyncpg.Connection | None = None
    if options.cross_instance_cancel:
        listener = await asyncpg.connect(dsn)
        await listener.add_listener(CANCEL_CHANNEL, cancel_on)
        # A cancel written before the LISTEN started sent its notice to no one.
        if (await store.load(cursor.turn_id)).cancel_requested:
            task.cancel()

    async def heartbeat() -> None:
        while not task.done():
            await asyncio.sleep(store.ttl / 3)
            flagged = await store.renew(cursor.turn_id, owner=owner)
            if flagged is None:
                # Released by the turn's own pause or end, or taken over.
                holder = (await store.load(cursor.turn_id)).lease_owner
                if holder is not None and holder != owner:
                    task.cancel()
                return
            if flagged and options.cross_instance_cancel:
                task.cancel()

    beat = asyncio.create_task(heartbeat())
    try:
        final = await task
        return Outcome("done" if final.state == DONE else "paused", final)
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
        stopped_at = time.time()
        await store.release(cursor.turn_id, owner=owner)
        return Outcome("stopped", None, stopped_at)
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
        if listener is not None:
            await listener.close()
