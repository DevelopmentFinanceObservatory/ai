"""The turn loop (`harness.turn.loop`) on Postgres, with toy nodes.

Each `StudioStore` stands for one instance: it claims the conversation's
lease as the routes do, and its heartbeat renewals carry the cancel flag
another instance's `request_cancel` sets.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from harness.studio_store import StudioStore
from harness.turn import transcript as entries
from harness.turn.events import NodeDone, Paused, TurnEvent, TurnFailed
from harness.turn.loop import END, Answer, Graph, Outcome, TurnExpired, run_segment
from harness.turn.node import Next, Node, NodeCtx, Pause
from harness.turn.store import AgentRecord, Cursor, CursorStale, TurnStatus, TurnStore
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .store_clock import StoreClock

PRINCIPAL = "user-1"
CONVERSATION = "conv-1"
BEAT = 0.02


class _NoAdk:
    """The `AdkBridge` of a store whose conversations ADK never held."""

    async def reimport(self, principal: str, conversation_id: str) -> None:
        del principal, conversation_id

    async def moved(self, principal: str, conversation_id: str) -> bool:
        del principal, conversation_id
        return False

    async def take_copy(self, principal: str, conversation_id: str) -> None:
        del principal, conversation_id


async def _no_turn_specs(principal: str, session_id: str) -> dict[str, Any]:
    del principal, session_id
    return {}


async def _instance(engine: AsyncEngine, clock: StoreClock) -> StudioStore:
    return StudioStore(engine, turn_specs=_no_turn_specs, adk=_NoAdk(), **clock.store_kwargs)


async def _claim(lease: StudioStore) -> str:
    """Claim the conversation's lease, creating the conversation on first use."""
    if await lease.load(PRINCIPAL, CONVERSATION) is None:
        await lease.create(PRINCIPAL, CONVERSATION)
    segment_id = lease.claim_segment(PRINCIPAL, CONVERSATION)
    await lease.save(PRINCIPAL, CONVERSATION)
    return segment_id


async def _open(store: TurnStore, segment_id: str, graph: Graph, *, node: str = "") -> Cursor:
    return await store.open_turn(
        PRINCIPAL,
        CONVERSATION,
        segment_id=segment_id,
        turn_index=1,
        node=node or graph.start,
        inputs={"message": "aid to Kenya"},
        state={},
        records=[entries.user_message("aid to Kenya")],
    )


async def _segment(
    store: TurnStore,
    lease: StudioStore,
    cursor: Cursor,
    segment_id: str,
    graph: Graph,
    events: list[TurnEvent],
    answer: Answer | None = None,
) -> Any:
    return await run_segment(
        store,
        lease,
        cursor,
        principal=PRINCIPAL,
        conversation_id=CONVERSATION,
        segment_id=segment_id,
        graph=graph,
        emit=events.append,
        answer=answer,
        heartbeat_seconds=BEAT,
    )


async def _open_cursor(store: TurnStore) -> Cursor:
    cursor = await store.open_cursor(PRINCIPAL, CONVERSATION)
    assert cursor is not None
    return cursor


async def _until(check: Callable[[], Awaitable[bool]]) -> None:
    for _ in range(500):
        if await check():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError


def _stays(node: str, *, ran: list[str]) -> Node:
    """A node that records its run, writes `state[node]` and appends a
    planner record, then waits until it is cancelled."""

    async def run(ctx: NodeCtx, node_input: Any) -> Next:
        ran.append(node)
        ctx.state[node] = "unwritten"
        ctx.transcript.append(AgentRecord(agent="planner", kind="response", body={}))
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    return Node(run=run)


@pytest.fixture
async def clock(pg_engine: AsyncEngine) -> StoreClock:
    return await StoreClock.start(pg_engine)


async def test_a_turn_writes_each_boundary_before_it_reports_it(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    stored_at_report: list[tuple[str, dict[str, Any]]] = []

    async def seed(ctx: NodeCtx, node_input: Any) -> Next:
        ctx.state["seeded"] = True
        ctx.specs["spec_a"] = {"spec_id": "spec_a"}
        ctx.transcript.append(AgentRecord(agent="planner", kind="response", body={"n": 1}))

        async def read_back() -> None:
            stored = await _open_cursor(store)
            stored_at_report.append((stored.node, stored.state))

        ctx.after_commit(read_back)
        return Next("next", {"brief": "b"})

    async def answer(ctx: NodeCtx, node_input: Any) -> Next:
        assert node_input == {"brief": "b"}
        ctx.state["answered"] = True
        return Next("done", "reply")

    graph = Graph(
        start="seed",
        nodes={"seed": Node(run=seed), "answer": Node(run=answer)},
        edges={("seed", "next"): "answer", ("answer", "done"): END},
    )
    events: list[TurnEvent] = []
    segment = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, events
    )

    assert segment.outcome is Outcome.DONE
    assert stored_at_report == [("answer", {"seeded": True})]
    assert events == [
        NodeDone(node="seed", label="next", output={"brief": "b"}),
        NodeDone(node="answer", label="done", output="reply"),
    ]
    assert (segment.cursor.status, segment.cursor.state) == (
        TurnStatus.DONE,
        {"seeded": True, "answered": True},
    )
    assert await store.specs(segment.cursor) == {"spec_a": {"spec_id": "spec_a"}}
    assert [r.kind for r in await store.records(segment.cursor)] == [entries.USER, "response"]


def _asking(calls: list[str]) -> Node:
    async def ask(ctx: NodeCtx, node_input: Any) -> Pause:
        calls.append("run")
        ctx.state["asked"] = node_input
        return Pause(kind="ask", payload={"question": "which donor?"}, message="Which donor?")

    async def answered(ctx: NodeCtx, node_input: Any, answer: dict[str, Any]) -> Next:
        calls.append("resume")
        assert ctx.state["asked"] == node_input
        return Next("next", answer)

    return Node(run=ask, resume=answered)


@pytest.mark.parametrize("start_moved", [False, True], ids=["same-graph", "new-start-node"])
async def test_a_paused_turn_resumes_at_its_node_in_another_instance(
    pg_engine: AsyncEngine, clock: StoreClock, start_moved: bool
) -> None:
    """Instance B resumes the turn instance A paused, at the node it paused
    at. A graph whose start node changed since the turn opened does not run
    the new start on the resume."""
    store = TurnStore(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    calls: list[str] = []
    received: list[Any] = []

    async def reply(ctx: NodeCtx, node_input: Any) -> Next:
        received.append(node_input)
        return Next("done")

    graph = Graph(
        start="ask",
        nodes={"ask": _asking(calls), "reply": Node(run=reply)},
        edges={("ask", "next"): "reply", ("reply", "done"): END},
    )
    segment_a = await _claim(lease_a)
    events: list[TurnEvent] = []
    paused = await _segment(
        store, lease_a, await _open(store, segment_a, graph), segment_a, graph, events
    )
    await lease_a.drop_lease(PRINCIPAL, CONVERSATION, segment_a)

    [event] = events
    assert isinstance(event, Paused)
    assert paused.outcome is Outcome.PAUSED
    assert (event.kind, event.round, event.message) == ("ask", 1, "Which donor?")
    assert event.interrupt_id.startswith("int_")
    assert paused.cursor.pause is not None
    assert paused.cursor.pause.interrupt_id == event.interrupt_id

    if start_moved:

        async def vet(ctx: NodeCtx, node_input: Any) -> Next:
            calls.append("vet")
            return Next("next")

        graph = Graph(
            start="vet",
            nodes={**graph.nodes, "vet": Node(run=vet)},
            edges={**graph.edges, ("vet", "next"): "ask"},
        )
    segment_b = await _claim(lease_b)
    answer = Answer(interrupt_id=event.interrupt_id, round=1, value={"donor": "DAC"})
    done = await _segment(
        store, lease_b, await _open_cursor(store), segment_b, graph, [], answer=answer
    )

    assert done.outcome is Outcome.DONE
    assert calls == ["run", "resume"]
    assert received == [{"donor": "DAC"}]
    assert await store.answer(done.cursor, interrupt_id=event.interrupt_id, round=1) == {
        "donor": "DAC"
    }
    assert [r.kind for r in await store.records(done.cursor)] == [
        entries.USER,
        entries.PAUSE,
        entries.ANSWER,
    ]


async def test_one_of_two_answers_racing_on_one_read_resumes_the_pause(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    calls: list[str] = []
    graph = Graph(start="ask", nodes={"ask": _asking(calls)}, edges={("ask", "next"): END})
    segment_id = await _claim(lease)
    events: list[TurnEvent] = []
    paused = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, events
    )
    [event] = events
    assert isinstance(event, Paused)

    results = await asyncio.gather(
        *(
            _segment(
                store,
                lease,
                paused.cursor,
                segment_id,
                graph,
                [],
                answer=Answer(interrupt_id=event.interrupt_id, round=1, value={"pick": pick}),
            )
            for pick in ("a", "b")
        ),
        return_exceptions=True,
    )

    assert sorted(type(result).__name__ for result in results) == ["CursorStale", "Segment"]
    assert calls == ["run", "resume"]


@pytest.mark.parametrize("paused", [False, True], ids=["unknown-node", "paused-without-resume"])
async def test_a_node_the_running_graph_lacks_expires_the_turn(
    pg_engine: AsyncEngine, clock: StoreClock, paused: bool
) -> None:
    """A turn opened at a node the running graph has removed, and a turn
    paused at a node that no longer resumes, each close `abandoned` with a
    `TurnExpired` error, and the conversation's next turn can open."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    calls: list[str] = []
    asking = _asking(calls)
    running = Graph(start="ask", nodes={"ask": Node(run=asking.run)}, edges={})
    answer: Answer | None = None
    if paused:
        before = Graph(start="ask", nodes={"ask": asking}, edges={("ask", "next"): END})
        events: list[TurnEvent] = []
        await _segment(
            store, lease, await _open(store, segment_id, before), segment_id, before, events
        )
        [event] = events
        assert isinstance(event, Paused)
        answer = Answer(interrupt_id=event.interrupt_id, round=1, value={})
        cursor = await _open_cursor(store)
    else:
        cursor = await _open(store, segment_id, running, node="retired")
    events = []

    segment = await _segment(store, lease, cursor, segment_id, running, events, answer=answer)

    [failed] = events
    assert isinstance(failed, TurnFailed) and isinstance(failed.error, TurnExpired)
    assert segment.outcome is Outcome.ABANDONED
    assert segment.cursor.status is TurnStatus.ABANDONED
    assert segment.cursor.error is not None and segment.cursor.error["type"] == "TurnExpired"
    assert calls == (["run"] if paused else [])
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) is None


async def test_a_node_that_raises_closes_the_turn_failed_with_the_last_written_state(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)

    async def seed(ctx: NodeCtx, node_input: Any) -> Next:
        ctx.state["seeded"] = True
        return Next("next")

    async def plan_start(ctx: NodeCtx, node_input: Any) -> Next:
        ctx.state["half"] = True
        raise RuntimeError("plan_start broke")

    graph = Graph(
        start="seed",
        nodes={"seed": Node(run=seed), "plan_start": Node(run=plan_start)},
        edges={("seed", "next"): "plan_start", ("plan_start", "next"): END},
    )
    events: list[TurnEvent] = []
    segment = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, events
    )

    assert segment.outcome is Outcome.FAILED
    assert isinstance(events[-1], TurnFailed) and events[-1].node == "plan_start"
    assert segment.cursor.error == {
        "node": "plan_start",
        "type": "RuntimeError",
        "message": "plan_start broke",
    }
    assert segment.cursor.state == {"seeded": True}
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) is None


async def _lease_cancel(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        return (await conn.execute(text("SELECT lease_cancel FROM conversations"))).scalar_one()


async def test_the_first_stop_ends_only_the_scoped_work_and_the_second_ends_the_turn(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    store = TurnStore(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)
    switching = asyncio.Event()
    ran: list[str] = []

    async def switch(ctx: NodeCtx, node_input: Any) -> Next:
        async with ctx.stop.open() as stopped:
            switching.set()
            revision = asyncio.ensure_future(asyncio.sleep(3600, result={"revised": True}))
            waiting = asyncio.ensure_future(stopped.wait())
            await asyncio.wait({revision, waiting}, return_when=asyncio.FIRST_COMPLETED)
            revision.cancel()
            waiting.cancel()
        return Next("offer", {"stopped": stopped.is_set()})

    graph = Graph(
        start="switch",
        nodes={"switch": Node(run=switch), "approve": _stays("approve", ran=ran)},
        edges={("switch", "offer"): "approve", ("approve", "next"): END},
    )
    events: list[TurnEvent] = []
    running = asyncio.create_task(
        _segment(store, lease_a, await _open(store, segment_id, graph), segment_id, graph, events)
    )
    await switching.wait()

    assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    await _until(lambda: _at(store, "approve"))
    assert events == [NodeDone(node="switch", label="offer", output={"stopped": True})]
    assert await _lease_cancel(pg_engine) is False
    assert not running.done()

    assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    segment = await asyncio.wait_for(running, 5)

    assert segment.outcome is Outcome.CANCELLED
    assert segment.cursor.status is TurnStatus.CANCELLED
    assert ran == ["approve"]
    left = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert left == []


async def _at(store: TurnStore, node: str) -> bool:
    cursor = await store.open_cursor(PRINCIPAL, CONVERSATION)
    return cursor is not None and cursor.node == node


@pytest.mark.parametrize(
    ("dies_in", "reruns"),
    [("work", ["work"]), ("ask", ["resume", "work"])],
    ids=["mid-run-after-the-answer", "mid-resume"],
)
async def test_a_node_cut_off_mid_run_runs_again_from_its_start_in_the_next_segment(
    pg_engine: AsyncEngine, clock: StoreClock, dies_in: str, reruns: list[str]
) -> None:
    """Instance A answers the pause and dies inside a node, in the resumed
    `ask` or in `work` after it. Instance B takes the turn over once A's
    lease lapses, and runs that node again from its start: `ask` from
    `resume` with the answer the turn accepted, `work` from `run`, each
    with the state and transcript as last written."""
    store = TurnStore(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    calls: list[str] = []
    started = asyncio.Event()
    seen: list[tuple[dict[str, Any], int]] = []
    asking = _asking(calls)

    async def resume(ctx: NodeCtx, node_input: Any, answer: dict[str, Any]) -> Next | Pause:
        assert asking.resume is not None
        result = await asking.resume(ctx, node_input, answer)
        if dies_in == "ask" and not started.is_set():
            ctx.state["unwritten"] = True
            started.set()
            await asyncio.sleep(3600)
        return result

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        calls.append("work")
        seen.append((dict(ctx.state), len(ctx.transcript.records)))
        if dies_in == "work" and not started.is_set():
            ctx.state["unwritten"] = True
            ctx.transcript.append(AgentRecord(agent="planner", kind="response", body={}))
            started.set()
            await asyncio.sleep(3600)
        return Next("done", node_input)

    graph = Graph(
        start="ask",
        nodes={"ask": Node(run=asking.run, resume=resume), "work": Node(run=work)},
        edges={("ask", "next"): "work", ("work", "done"): END},
    )
    segment_a = await _claim(lease_a)
    events: list[TurnEvent] = []
    await _segment(store, lease_a, await _open(store, segment_a, graph), segment_a, graph, events)
    [event] = events
    assert isinstance(event, Paused)
    answer = Answer(interrupt_id=event.interrupt_id, round=1, value={"donor": "DAC"})
    dying = asyncio.create_task(
        _segment(store, lease_a, await _open_cursor(store), segment_a, graph, [], answer=answer)
    )
    await started.wait()
    dying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dying
    assert (await _open_cursor(store)).status is TurnStatus.RUNNING
    calls_before = list(calls)

    await clock.advance(60)
    segment_b = await _claim(lease_b)
    segment = await _segment(store, lease_b, await _open_cursor(store), segment_b, graph, [])

    assert segment.outcome is Outcome.DONE
    assert calls[len(calls_before) :] == reruns
    assert segment.cursor.state == {"asked": None}
    # `work`'s last run saw the state and transcript as last written: the
    # user's message, the pause and the answer.
    assert seen[-1] == ({"asked": None}, 3)


async def test_a_lease_taken_by_another_segment_ends_the_segment_lost(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    ran: list[str] = []
    graph = Graph(start="work", nodes={"work": _stays("work", ran=ran)}, edges={})
    cursor = await _open(store, segment_id, graph)
    async with pg_engine.begin() as conn:
        await conn.execute(text("UPDATE conversations SET lease_segment_id = 'segment-b'"))

    segment = await asyncio.wait_for(
        _segment(store, lease, cursor, segment_id, graph, []), timeout=5
    )

    assert segment.outcome is Outcome.LOST
    assert ran == ["work"]
    assert (await _open_cursor(store)).version == cursor.version


@pytest.mark.parametrize("taken_over", [False, True], ids=["one-segment", "after-a-death"])
async def test_a_second_visit_to_the_answered_node_in_the_same_round_runs_without_the_answer(
    pg_engine: AsyncEngine, clock: StoreClock, taken_over: bool
) -> None:
    """The planner's resume refuses, and the turn comes back to the planner
    through `plan_done` without pausing again. That second visit runs from
    `run`, also when the segment that took the turn over after a death on
    it starts there."""
    store = TurnStore(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    calls: list[str] = []
    second_visit = asyncio.Event()

    async def plan(ctx: NodeCtx, node_input: Any) -> Next | Pause:
        calls.append("run")
        if not ctx.state.get("answered"):
            return Pause(kind="planner_ask", payload={"question": "which donor?"})
        if taken_over and not second_visit.is_set():
            second_visit.set()
            await asyncio.sleep(3600)
        return Next("done")

    async def replan(ctx: NodeCtx, node_input: Any, answer: dict[str, Any]) -> Next:
        calls.append("resume")
        ctx.state["answered"] = answer
        return Next("refused")

    async def plan_done(ctx: NodeCtx, node_input: Any) -> Next:
        return Next("next")

    graph = Graph(
        start="planner",
        nodes={"planner": Node(run=plan, resume=replan), "plan_done": Node(run=plan_done)},
        edges={
            ("planner", "refused"): "plan_done",
            ("plan_done", "next"): "planner",
            ("planner", "done"): END,
        },
    )
    segment_a = await _claim(lease_a)
    events: list[TurnEvent] = []
    await _segment(store, lease_a, await _open(store, segment_a, graph), segment_a, graph, events)
    [event] = events
    assert isinstance(event, Paused)
    answer = Answer(interrupt_id=event.interrupt_id, round=1, value={"donor": "DAC"})
    answering = _segment(
        store, lease_a, await _open_cursor(store), segment_a, graph, [], answer=answer
    )
    if taken_over:
        dying = asyncio.create_task(answering)
        await second_visit.wait()
        dying.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dying
        cursor = await _open_cursor(store)
        assert (cursor.node, cursor.round, cursor.owed) == ("planner", 1, None)
        await clock.advance(60)
        segment_b = await _claim(lease_b)
        segment = await _segment(store, lease_b, cursor, segment_b, graph, [])
    else:
        segment = await answering

    assert segment.outcome is Outcome.DONE
    expected = ["run", "resume", "run"] + (["run"] if taken_over else [])
    assert calls == expected


_NAN = float("nan")


@pytest.mark.parametrize(
    ("ends", "error_type"),
    [
        ("nan-node-input", "ValueError"),
        ("unencodable-node-input", "TypeError"),
        ("nan-pause-payload", "ValueError"),
        ("nan-state-at-the-end", "ValueError"),
        ("nan-spec-block", "ValueError"),
    ],
)
async def test_a_boundary_write_that_cannot_store_its_values_closes_the_turn_failed(
    pg_engine: AsyncEngine, clock: StoreClock, ends: str, error_type: str
) -> None:
    """A node's outcome that JSON cannot hold fails the boundary's write,
    which closes the turn `failed` at that node with the error stored and
    reported, so no later segment reruns the node into the same failure."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)

    async def work(ctx: NodeCtx, node_input: Any) -> Next | Pause:
        match ends:
            case "nan-node-input":
                return Next("next", {"share": _NAN})
            case "unencodable-node-input":
                return Next("next", object())
            case "nan-pause-payload":
                return Pause(kind="ask", payload={"share": _NAN})
            case "nan-state-at-the-end":
                ctx.state["share"] = _NAN
                return Next("done")
            case _:
                ctx.specs["spec_a"] = {"share": _NAN}
                return Next("next")

    async def after(ctx: NodeCtx, node_input: Any) -> Next:
        raise AssertionError("the turn went past the failed write")

    graph = Graph(
        start="work",
        nodes={"work": Node(run=work), "after": Node(run=after)},
        edges={("work", "next"): "after", ("work", "done"): END, ("after", "next"): END},
    )
    events: list[TurnEvent] = []
    segment = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, events
    )

    assert segment.outcome is Outcome.FAILED
    [failed] = events
    assert isinstance(failed, TurnFailed) and failed.node == "work"
    assert segment.cursor.status is TurnStatus.FAILED
    assert segment.cursor.error is not None
    assert (segment.cursor.error["node"], segment.cursor.error["type"]) == ("work", error_type)
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) is None


async def test_a_cursor_moved_since_it_was_read_runs_no_node(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """A segment handed a cursor another write has moved since refuses it
    before the node runs, so no node repeats its work on stale state."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    ran: list[Any] = []

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        ran.append(node_input)
        return Next("done")

    graph = Graph(start="work", nodes={"work": Node(run=work)}, edges={("work", "done"): END})
    read = await _open(store, segment_id, graph)
    await store.advance(
        read, segment_id=segment_id, node="work", node_input="moved", state={}, specs={}
    )

    with pytest.raises(CursorStale):
        await _segment(store, lease, read, segment_id, graph, [])
    assert ran == []


async def test_the_next_node_takes_its_input_as_stored(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """The node an edge reaches gets its input as the cursor stores it, the
    same value a segment that reruns it after a crash reads back."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    received: list[Any] = []

    async def seed(ctx: NodeCtx, node_input: Any) -> Next:
        return Next("next", ("spec_a", 2))

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        received.append(node_input)
        return Next("done")

    graph = Graph(
        start="seed",
        nodes={"seed": Node(run=seed), "work": Node(run=work)},
        edges={("seed", "next"): "work", ("work", "done"): END},
    )
    await _segment(store, lease, await _open(store, segment_id, graph), segment_id, graph, [])

    assert received == [["spec_a", 2]]


class _HeldAfterClosing(TurnStore):
    """A store whose first `finish(DONE)` commits and then holds its caller,
    the window in which a Stop or a lost lease can cancel the segment after
    the turn closed."""

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)
        self.closed = asyncio.Event()

    async def finish(self, cursor: Cursor, **kwargs: Any) -> Cursor:
        finished = await super().finish(cursor, **kwargs)
        if kwargs["status"] is TurnStatus.DONE and not self.closed.is_set():
            self.closed.set()
            await asyncio.sleep(3600)
        return finished


@pytest.mark.parametrize("cut_by", ["stop", "lost-lease"])
async def test_a_segment_cut_off_after_the_turn_closed_reports_the_stored_close(
    pg_engine: AsyncEngine, clock: StoreClock, cut_by: str
) -> None:
    store = _HeldAfterClosing(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        return Next("done")

    graph = Graph(start="work", nodes={"work": Node(run=work)}, edges={("work", "done"): END})
    running = asyncio.create_task(
        _segment(store, lease_a, await _open(store, segment_id, graph), segment_id, graph, [])
    )
    await store.closed.wait()
    if cut_by == "stop":
        assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    else:
        async with pg_engine.begin() as conn:
            await conn.execute(text("UPDATE conversations SET lease_segment_id = 'segment-b'"))
    segment = await asyncio.wait_for(running, 5)

    assert segment.outcome is Outcome.DONE
    assert segment.cursor.status is TurnStatus.DONE


async def test_a_cursor_owing_an_answer_the_store_lacks_closes_the_turn_failed(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    calls: list[str] = []
    graph = Graph(start="ask", nodes={"ask": _asking(calls)}, edges={("ask", "next"): END})
    events: list[TurnEvent] = []
    paused = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, events
    )
    [event] = events
    assert isinstance(event, Paused)
    owing = await store.record_answer(
        paused.cursor, segment_id=segment_id, interrupt_id=event.interrupt_id, round=1, answer={}
    )
    async with pg_engine.begin() as conn:
        await conn.execute(text("DELETE FROM turn_answers"))
    events = []

    segment = await _segment(store, lease, owing, segment_id, graph, events)

    assert segment.outcome is Outcome.FAILED
    assert isinstance(events[-1], TurnFailed) and events[-1].node == "ask"
    assert calls == ["run"]


async def test_events_a_node_emits_reach_the_consumer_as_it_runs_and_before_its_pause(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """An event a node emits reaches the consumer while the node is still
    running, and the node's events arrive in order before its `Paused`."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    gate = asyncio.Event()
    first, second = NodeDone(node="draft", label="first"), NodeDone(node="draft", label="second")

    async def draft(ctx: NodeCtx, node_input: Any) -> Pause:
        ctx.emit(first)
        await gate.wait()
        ctx.emit(second)
        return Pause(kind="planner_ask", payload={"question": "which donor?"})

    async def answered(ctx: NodeCtx, node_input: Any, answer: dict[str, Any]) -> Next:
        return Next("next")

    graph = Graph(
        start="draft",
        nodes={"draft": Node(run=draft, resume=answered)},
        edges={("draft", "next"): END},
    )
    events: list[TurnEvent] = []
    running = asyncio.create_task(
        _segment(store, lease, await _open(store, segment_id, graph), segment_id, graph, events)
    )

    async def first_arrived() -> bool:
        return events == [first]

    await _until(first_arrived)
    assert not running.done()
    gate.set()
    segment = await asyncio.wait_for(running, 5)

    assert segment.outcome is Outcome.PAUSED
    assert events[:2] == [first, second]
    assert [type(event) for event in events] == [NodeDone, NodeDone, Paused]
