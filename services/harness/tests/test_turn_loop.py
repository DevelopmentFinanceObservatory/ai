"""The turn loop (`harness.turn.loop`) on Postgres, with toy nodes.

Each `StudioStore` stands for one instance: it claims the conversation's
lease as the routes do, and its heartbeat renewals carry the Stop request
another instance's `request_cancel` records.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import harness.provenance as provenance_module
import pytest
from harness.conversation import ledger_conversation_id
from harness.graph.state import ORCHESTRATOR_FRAMES_KEY, frame_store_keys
from harness.provenance import ProvenanceStore
from harness.studio_store import StudioStore
from harness.turn import transcript as entries
from harness.turn.events import NodeDone, Paused, TurnEvent, TurnFailed
from harness.turn.loop import (
    END,
    Answer,
    Graph,
    Outcome,
    TurnExpired,
    reject,
    run_segment,
    stop_turn,
)
from harness.turn.node import Next, Node, NodeCtx, Pause
from harness.turn.store import (
    SPECS_KEY,
    AgentRecord,
    Cursor,
    CursorStale,
    FrameLineage,
    PauseNotOpen,
    TurnStatus,
    TurnStore,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .store_clock import StoreClock

PRINCIPAL = "user-1"
CONVERSATION = "conv-1"
BEAT = 0.02
_SPEC_A = {"spec_a": {"spec_id": "spec_a"}}


async def _instance(engine: AsyncEngine, clock: StoreClock) -> StudioStore:
    return StudioStore(engine, **clock.store_kwargs)


async def _claim(lease: StudioStore) -> str:
    """Claim the conversation's lease, creating the conversation on first use
    with its turn index at 1, the turn `_open` opens."""
    if await lease.load(PRINCIPAL, CONVERSATION) is None:
        (await lease.create(PRINCIPAL, CONVERSATION)).begin_turn()
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


def _stays(node: str, *, ran: list[str], pauses: bool = False) -> Node:
    """A node that records its run, writes `state[node]` and appends a
    planner record, then waits until it is cancelled. With `pauses`, it is a
    node that can pause (it has a `resume`), as a stop edge must reach."""

    async def run(ctx: NodeCtx, node_input: Any) -> Next:
        ran.append(node)
        ctx.state[node] = "unwritten"
        ctx.transcript.append(AgentRecord(agent="planner", kind="response", body={}))
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    async def resume(ctx: NodeCtx, node_input: Any, answer: dict[str, Any]) -> Next:
        return await run(ctx, node_input)

    return Node(run=run, resume=resume if pauses else None)


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
        ctx.state[SPECS_KEY] = {"spec_a": {"spec_id": "spec_a"}}
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
    assert stored_at_report == [("answer", {"seeded": True, SPECS_KEY: _SPEC_A})]
    assert events == [
        NodeDone(node="seed", label="next", output={"brief": "b"}),
        NodeDone(node="answer", label="done", output="reply"),
    ]
    assert (segment.cursor.status, segment.cursor.state) == (
        TurnStatus.DONE,
        {"seeded": True, "answered": True, SPECS_KEY: _SPEC_A},
    )
    assert await store.specs(segment.cursor) == _SPEC_A
    assert [r.kind for r in await store.records(segment.cursor)] == [entries.USER, "response"]


@pytest.mark.parametrize("ends", ["raises", "goes-on"])
async def test_a_node_s_change_in_place_leaves_the_cursor_as_stored(
    pg_engine: AsyncEngine, clock: StoreClock, ends: str
) -> None:
    """A node works on a copy of the state. One that changes a spec block in
    place and raises leaves the closed cursor's state as stored; one that
    puts a tuple where a list is stored and goes on hands the next node the
    list a rerun would read."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    stored = {"spec_a": {"v": 1, "pair": [1, 2]}}
    seen: list[Any] = []

    async def seed(ctx: NodeCtx, node_input: Any) -> Next:
        ctx.state[SPECS_KEY] = stored
        return Next("next")

    async def change(ctx: NodeCtx, node_input: Any) -> Next:
        block = ctx.state[SPECS_KEY]["spec_a"]
        if ends == "raises":
            block["v"] = 2
            raise RuntimeError("change broke")
        block["pair"] = (1, 2)
        return Next("next")

    async def read(ctx: NodeCtx, node_input: Any) -> Next:
        seen.append(ctx.state[SPECS_KEY]["spec_a"]["pair"])
        return Next("done")

    graph = Graph(
        start="seed",
        nodes={"seed": Node(run=seed), "change": Node(run=change), "read": Node(run=read)},
        edges={("seed", "next"): "change", ("change", "next"): "read", ("read", "done"): END},
    )
    segment = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, []
    )

    assert segment.cursor.state[SPECS_KEY] == stored
    assert segment.cursor.spec_blocks == stored
    assert seen == ([] if ends == "raises" else [[1, 2]])


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
    `TurnExpired` error, and the conversation's next turn can open. The
    answer to the paused one is not accepted."""
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
    assert failed.error.args[0] == (
        "the running graph cannot resume 'ask'"
        if paused
        else "the running graph cannot run 'retired'"
    )
    if answer is not None:
        assert await store.answer(segment.cursor, interrupt_id=answer.interrupt_id, round=1) is None
        assert entries.ANSWER not in [r.kind for r in await store.records(segment.cursor)]


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
        "transient": False,
    }
    assert segment.cursor.state == {"seeded": True, SPECS_KEY: {}}
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) is None


class _Refused(Exception):  # noqa: N818 -- a stand-in for a refusal-bearing tool error
    """An error stating a refusal and the MCP call whose pinned timeout it
    hit, raised from a dropped connection."""

    def __init__(self) -> None:
        super().__init__("get_data timed out")
        self.refusal = {"code": "capability_unavailable", "retry_mode": "do_not_retry"}
        self.timeout_tool = "get_data"
        self.timeout_key = "get_data:spec_a"


async def test_a_failed_turn_records_what_its_error_frame_reads(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """The turn's failure carries the exception the node raised, and the
    cursor records what the Studio error frame reads off it: the refusal
    it states, the timed-out call, and whether a later attempt may not meet
    it, read through its cause."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    raised = _Refused()

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        raise raised from ConnectionResetError("peer closed")

    graph = Graph(start="work", nodes={"work": Node(run=work)}, edges={})
    events: list[TurnEvent] = []
    segment = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, events
    )

    [failed] = events
    assert isinstance(failed, TurnFailed) and failed.error is raised
    assert segment.cursor.error == {
        "node": "work",
        "type": "_Refused",
        "message": "get_data timed out",
        "transient": True,
        "refusal": {"code": "capability_unavailable", "retry_mode": "do_not_retry"},
        "timeout_tool": "get_data",
        "timeout_key": "get_data:spec_a",
    }


async def _paused(store: TurnStore, lease: StudioStore, calls: list[str]) -> tuple[Graph, Paused]:
    """A turn paused at `ask` by `lease`'s instance, whose lease it then drops."""
    graph = Graph(start="ask", nodes={"ask": _asking(calls)}, edges={("ask", "next"): END})
    segment_id = await _claim(lease)
    events: list[TurnEvent] = []
    await _segment(store, lease, await _open(store, segment_id, graph), segment_id, graph, events)
    await lease.drop_lease(PRINCIPAL, CONVERSATION, segment_id)
    [paused] = events
    assert isinstance(paused, Paused)
    return graph, paused


async def test_a_stop_while_the_turn_is_paused_closes_it_cancelled(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """A Stop finds no live segment for a paused turn; the instance that
    takes the lease closes the turn `cancelled`, and the answer to its
    pause is then refused."""
    store = TurnStore(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    calls: list[str] = []
    _, paused = await _paused(store, lease_a, calls)

    assert not await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    segment_id = await _claim(lease_b)
    stopped = await stop_turn(store, await _open_cursor(store), segment_id=segment_id)

    assert (stopped.status, stopped.pause, stopped.error) == (TurnStatus.CANCELLED, None, None)
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) is None
    with pytest.raises(PauseNotOpen):
        await store.record_answer(
            stopped,
            segment_id=segment_id,
            interrupt_id=paused.interrupt_id,
            round=paused.round,
            answer={"donor": "DAC"},
        )
    assert calls == ["run"]


async def test_a_reject_closes_the_paused_turn_done_without_resuming_its_node(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """A reject of the open pause closes the turn `done` with the answer in
    its transcript and runs nothing; one naming another round is refused and
    leaves the turn paused."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    calls: list[str] = []
    _, paused = await _paused(store, lease, calls)
    segment_id = await _claim(lease)
    cursor = await _open_cursor(store)

    with pytest.raises(PauseNotOpen):
        await reject(
            store,
            cursor,
            segment_id=segment_id,
            answer=Answer(interrupt_id=paused.interrupt_id, round=2, value={"rejected": True}),
        )
    assert (await _open_cursor(store)).status is TurnStatus.PAUSED

    rejected = await reject(
        store,
        cursor,
        segment_id=segment_id,
        answer=Answer(
            interrupt_id=paused.interrupt_id, round=paused.round, value={"rejected": True}
        ),
    )

    assert (rejected.status, rejected.pause) == (TurnStatus.DONE, None)
    assert await store.open_cursor(PRINCIPAL, CONVERSATION) is None
    records = await store.records(rejected)
    assert [r.kind for r in records] == [entries.USER, entries.PAUSE, entries.ANSWER]
    assert records[-1].body["answer"] == {"rejected": True}
    assert calls == ["run"]


async def _stop_request(engine: AsyncEngine) -> str | None:
    async with engine.connect() as conn:
        return (await conn.execute(text("SELECT stop_id FROM conversations"))).scalar_one()


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
        nodes={
            "switch": Node(run=switch, stop_edges=frozenset({"offer"})),
            "approve": _stays("approve", ran=ran, pauses=True),
        },
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
    # The scope took the Stop with no write, so it stays live until a pause
    # write records it taken: a release now would close the turn on it.
    assert await _stop_request(pg_engine) is not None
    assert await lease_b.stop_live(PRINCIPAL, CONVERSATION)
    assert not running.done()

    assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    segment = await asyncio.wait_for(running, 5)

    assert segment.outcome is Outcome.CANCELLED
    assert segment.cursor.status is TurnStatus.CANCELLED
    assert ran == ["approve"]
    left = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert left == []


async def test_the_pause_after_a_scope_took_a_stop_records_it_taken(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """The scope takes the first Stop and the node goes on to a node that
    pauses: the pause write records the Stop taken, so no Stop is live on
    the paused turn and the renewals in between did not cut the node."""
    store = TurnStore(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)
    switching = asyncio.Event()

    async def switch(ctx: NodeCtx, node_input: Any) -> Next:
        async with ctx.stop.open() as stopped:
            switching.set()
            await stopped.wait()
            # Renewals report the taken Stop meanwhile, and change nothing.
            await asyncio.sleep(BEAT * 5)
        return Next("offer")

    async def offer(ctx: NodeCtx, node_input: Any) -> Pause:
        return Pause(kind="approve", payload={"plan": "unchanged"})

    graph = Graph(
        start="switch",
        nodes={
            "switch": Node(run=switch, stop_edges=frozenset({"offer"})),
            "offer": Node(run=offer, resume=offer),
        },
        edges={("switch", "offer"): "offer"},
    )
    running = asyncio.create_task(
        _segment(store, lease_a, await _open(store, segment_id, graph), segment_id, graph, [])
    )
    await switching.wait()
    assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    segment = await asyncio.wait_for(running, 5)

    assert segment.outcome is Outcome.PAUSED
    stop_id = await _stop_request(pg_engine)
    async with pg_engine.connect() as conn:
        taken = (await conn.execute(text("SELECT stop_taken_id FROM turn_cursors"))).scalar_one()
    assert stop_id is not None and taken == stop_id
    assert not await lease_b.stop_live(PRINCIPAL, CONVERSATION)


class _CloseUnacknowledged(TurnStore):
    """A turn store whose first `cancelled` close commits and then raises,
    as a close whose acknowledgement was lost does."""

    lost = False

    async def finish(self, cursor: Cursor, **kwargs: Any) -> Cursor:
        closed = await super().finish(cursor, **kwargs)
        if kwargs["status"] is TurnStatus.CANCELLED and not self.lost:
            self.lost = True
            raise ConnectionError("the close's acknowledgement was lost")
        return closed


@pytest.mark.parametrize(
    ("leaves", "ran_after"),
    [
        pytest.param("end", [], id="ends-the-turn"),
        pytest.param("end-unacknowledged", [], id="ends-the-turn-close-unacknowledged"),
        pytest.param("other-edge", [], id="leaves-on-another-edge"),
        pytest.param("offer-goes-on", ["offer"], id="the-offer-goes-on-without-pausing"),
    ],
)
async def test_a_stop_the_scope_took_closes_the_turn_when_no_pause_records_it(
    pg_engine: AsyncEngine, clock: StoreClock, leaves: str, ran_after: list[str]
) -> None:
    """The scope takes the first Stop, and no pause records it: the node
    that took it ends the turn or leaves on an edge that is not a stop edge,
    or the node its stop edge leads to returns on. The loop closes the turn
    `cancelled` in place of that write and runs nothing past it, also when
    the close commits and its acknowledgement is lost."""
    store = (
        _CloseUnacknowledged(pg_engine) if leaves == "end-unacknowledged" else TurnStore(pg_engine)
    )
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)
    switching = asyncio.Event()
    ran: list[str] = []

    async def switch(ctx: NodeCtx, node_input: Any) -> Next:
        async with ctx.stop.open() as stopped:
            switching.set()
            await stopped.wait()
        return Next("offer" if leaves == "offer-goes-on" else "on")

    async def offer(ctx: NodeCtx, node_input: Any) -> Next:
        ran.append("offer")
        return Next("on")

    async def offered(ctx: NodeCtx, node_input: Any, answer: dict[str, Any]) -> Next:
        return Next("on")

    graph = Graph(
        start="switch",
        nodes={
            "switch": Node(run=switch, stop_edges=frozenset({"offer"})),
            "offer": Node(run=offer, resume=offered),
            "after": _stays("after", ran=ran),
        },
        edges={
            ("switch", "offer"): "offer",
            ("switch", "on"): "after" if leaves == "other-edge" else END,
            ("offer", "on"): "after",
        },
    )
    running = asyncio.create_task(
        _segment(store, lease_a, await _open(store, segment_id, graph), segment_id, graph, [])
    )
    await switching.wait()
    assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    segment = await asyncio.wait_for(running, 5)

    assert segment.outcome is Outcome.CANCELLED
    assert segment.cursor.status is TurnStatus.CANCELLED
    assert ran == ran_after
    assert not await lease_b.stop_live(PRINCIPAL, CONVERSATION)


@pytest.mark.parametrize(
    "reaches", ["work", "end"], ids=["a-node-that-cannot-pause", "the-turns-end"]
)
def test_a_stop_edge_that_reaches_no_node_that_pauses_fails_the_graph(reaches: str) -> None:
    """A node that holds the turn's `StopScope` open names its stop edges;
    one that reaches a node with no `resume`, or the turn's end, is refused
    when the graph is built, since no pause there would record the Stop."""

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        return Next("on")

    with pytest.raises(ValueError, match="stop edge"):
        Graph(
            start="switch",
            nodes={
                "switch": Node(run=work, stop_edges=frozenset({"offer"})),
                "work": Node(run=work),
            },
            edges={("switch", "offer"): "work" if reaches == "work" else END},
        )


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
    assert segment.cursor.state == {"asked": None, SPECS_KEY: {}}
    # `work`'s last run saw the state and transcript as last written: the
    # user's message, the pause and the answer.
    assert seen[-1] == ({"asked": None, SPECS_KEY: {}}, 3)


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
        ("nan-node-input", "NonFiniteJsonError"),
        ("unencodable-node-input", "TypeError"),
        ("nan-pause-payload", "NonFiniteJsonError"),
        ("nan-state-at-the-end", "NonFiniteJsonError"),
        ("nan-spec-block", "NonFiniteJsonError"),
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
                ctx.state[SPECS_KEY] = {"spec_a": {"share": _NAN}}
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


async def test_a_boundary_write_whose_lineage_the_database_refuses_closes_the_turn_failed(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """The turn's frame lineage commits with the boundary's cursor write, so
    a refusal of its rows by the database (here disk_full, 53100, raised by a
    trigger) fails that write, and the turn closes `failed` at the node as
    for any other boundary write that fails."""
    provenance = ProvenanceStore()
    scope = ledger_conversation_id(PRINCIPAL, CONVERSATION)
    frame = {"spec_id": "spec_a", "content_version": "cv_a"}
    store = TurnStore(
        pg_engine, lineage=FrameLineage(source=provenance, frame_keys=frame_store_keys)
    )
    lease = await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE FUNCTION refuse_lineage() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'refused' USING ERRCODE = '53100'; END $$"
            )
        )
        await conn.execute(
            text(
                "CREATE TRIGGER refuse_lineage BEFORE INSERT OR UPDATE ON turn_frame_lineage "
                "FOR EACH ROW EXECUTE FUNCTION refuse_lineage()"
            )
        )

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        # The node's fetch stores the frame's lineage, as `fetch` does.
        provenance.put(conversation=scope, manifest={}, **frame)
        ctx.state[ORCHESTRATOR_FRAMES_KEY] = {"frame_a": {"ref": frame}}
        return Next("next")

    async def after(ctx: NodeCtx, node_input: Any) -> Next:
        raise AssertionError("the turn went past the failed write")

    graph = Graph(
        start="work",
        nodes={"work": Node(run=work), "after": Node(run=after)},
        edges={("work", "next"): "after", ("after", "next"): END},
    )
    events: list[TurnEvent] = []
    segment = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, events
    )

    assert segment.outcome is Outcome.FAILED
    [failed] = events
    assert isinstance(failed, TurnFailed) and failed.node == "work"
    assert segment.cursor.status is TurnStatus.FAILED
    async with pg_engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM turn_frame_lineage"))).scalar() == 0


async def test_a_segments_frames_count_against_the_provenance_bound_once_it_ends(
    pg_engine: AsyncEngine, clock: StoreClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frames a segment named stay outside the provenance store's
    per-conversation bound only while it runs: once it pauses, later stores
    in the conversation evict them like any other entry."""
    provenance = ProvenanceStore()
    scope = ledger_conversation_id(PRINCIPAL, CONVERSATION)
    frame = {"spec_id": "spec_a", "content_version": "cv_a"}
    store = TurnStore(
        pg_engine, lineage=FrameLineage(source=provenance, frame_keys=frame_store_keys)
    )
    lease = await _instance(pg_engine, clock)
    segment_id = await _claim(lease)

    async def work(ctx: NodeCtx, node_input: Any) -> Pause:
        provenance.put(conversation=scope, manifest={}, **frame)
        ctx.state[ORCHESTRATOR_FRAMES_KEY] = {"frame_a": {"ref": frame}}
        return Pause(kind="ask", payload={})

    graph = Graph(start="work", nodes={"work": Node(run=work)}, edges={})
    segment = await _segment(
        store, lease, await _open(store, segment_id, graph), segment_id, graph, []
    )
    assert segment.outcome is Outcome.PAUSED
    monkeypatch.setattr(provenance_module, "_MAX_MANIFESTS_PER_CONVERSATION", 1)
    for n in (1, 2):
        provenance.put(conversation=scope, manifest={}, spec_id=f"spec_{n}", content_version="cv")

    assert provenance.get(conversation=scope, **frame) is None


@pytest.mark.parametrize("since", ["moved", "closed"])
async def test_a_cursor_moved_since_it_was_read_runs_no_node(
    pg_engine: AsyncEngine, clock: StoreClock, since: str
) -> None:
    """A segment handed a cursor another write has moved since, or a cursor
    of a turn that is closed, refuses it before the node runs, so no node
    repeats its work on stale state."""
    store, lease = TurnStore(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    ran: list[Any] = []

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        ran.append(node_input)
        return Next("done")

    graph = Graph(start="work", nodes={"work": Node(run=work)}, edges={("work", "done"): END})
    read = await _open(store, segment_id, graph)
    if since == "moved":
        await store.advance(
            read,
            segment_id=segment_id,
            node="work",
            node_input="moved",
            state={},
            lineage=None,
        )
    else:
        read = await store.turn(
            (await store.finish(read, segment_id=segment_id, status=TurnStatus.DONE)).id
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


class _ReportsAfterClosing(StudioStore):
    """A lease store whose renewal that reads a Stop reports it only once
    `closed` is set: the Stop was live when the renewal read it, and the
    turn's close commits before the heartbeat acts on it."""

    def __init__(self, engine: AsyncEngine, *, closed: asyncio.Event, **kwargs: Any) -> None:
        super().__init__(engine, **kwargs)
        self.closed = closed
        self.read_stop = asyncio.Event()

    async def renew(self, *args: Any, **kwargs: Any) -> Any:
        renewal = await super().renew(*args, **kwargs)
        if renewal.stop_id is not None:
            self.read_stop.set()
            await self.closed.wait()
        return renewal


@pytest.mark.parametrize("cut_by", ["stop", "lost-lease"])
async def test_a_segment_cut_off_after_the_turn_closed_reports_the_stored_close(
    pg_engine: AsyncEngine, clock: StoreClock, cut_by: str
) -> None:
    store = _HeldAfterClosing(pg_engine)
    lease_b = await _instance(pg_engine, clock)
    lease_a = _ReportsAfterClosing(pg_engine, closed=store.closed, **clock.store_kwargs)
    segment_id = await _claim(lease_a)

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        if cut_by == "stop":
            # A renewal reads the Stop while the turn runs, and reports it
            # once the close below has committed.
            assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
            await lease_a.read_stop.wait()
        return Next("done")

    graph = Graph(start="work", nodes={"work": Node(run=work)}, edges={("work", "done"): END})
    running = asyncio.create_task(
        _segment(store, lease_a, await _open(store, segment_id, graph), segment_id, graph, [])
    )
    await store.closed.wait()
    if cut_by == "lost-lease":
        async with pg_engine.begin() as conn:
            await conn.execute(text("UPDATE conversations SET lease_segment_id = 'segment-b'"))
    segment = await asyncio.wait_for(running, 5)

    assert segment.outcome is Outcome.DONE
    assert segment.cursor.status is TurnStatus.DONE


class _HeldAfterPausing(TurnStore):
    """A store whose first `pause` commits and then holds its caller, the
    window in which a Stop or a lost lease can cancel the segment after the
    pause is stored and before the loop reports it."""

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)
        self.paused = asyncio.Event()

    async def pause(self, cursor: Cursor, **kwargs: Any) -> Cursor:
        paused = await super().pause(cursor, **kwargs)
        if not self.paused.is_set():
            self.paused.set()
            await asyncio.sleep(3600)
        return paused


@pytest.mark.parametrize("cut_by", ["stop", "lost-lease"])
async def test_a_segment_cut_off_after_its_pause_committed_settles_on_the_cut(
    pg_engine: AsyncEngine, clock: StoreClock, cut_by: str
) -> None:
    """A Stop after the pause's write closes the turn `cancelled` with no
    question shown: the user stopped it. After a lost lease the client gets
    the question the store holds, and the turn stays paused for the answer."""
    store = _HeldAfterPausing(pg_engine)
    lease_a, lease_b = await _instance(pg_engine, clock), await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)
    graph = Graph(start="ask", nodes={"ask": _asking([])}, edges={("ask", "next"): END})
    events: list[TurnEvent] = []
    running = asyncio.create_task(
        _segment(store, lease_a, await _open(store, segment_id, graph), segment_id, graph, events)
    )
    await store.paused.wait()
    if cut_by == "stop":
        assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    else:
        async with pg_engine.begin() as conn:
            await conn.execute(text("UPDATE conversations SET lease_segment_id = 'segment-b'"))
    segment = await asyncio.wait_for(running, 5)

    if cut_by == "stop":
        assert events == []
        assert segment.outcome is Outcome.CANCELLED
        assert (await store.turn(segment.cursor.id)).status is TurnStatus.CANCELLED
        return
    stored = await _open_cursor(store)
    [paused] = events
    assert isinstance(paused, Paused)
    assert (paused.kind, paused.round, paused.message) == ("ask", 1, "Which donor?")
    assert stored.pause is not None and stored.pause.interrupt_id == paused.interrupt_id
    assert segment.outcome is Outcome.PAUSED
    assert segment.cursor.status is TurnStatus.PAUSED


class _ClosesNothing(TurnStore):
    """A store whose `finish` cannot reach the database."""

    async def finish(self, cursor: Cursor, **kwargs: Any) -> Cursor:
        raise ConnectionError("the database is unreachable")


@pytest.mark.parametrize("paused", [False, True], ids=["running", "paused-without-resume"])
async def test_a_close_that_cannot_be_written_still_reports_the_nodes_error(
    pg_engine: AsyncEngine, clock: StoreClock, paused: bool
) -> None:
    """The node's exception, or the expiry of a pause its node can no longer
    resume, reaches the client, and the segment ends `unclosed`. The turn
    stays open, running or paused as last written, for a Stop or the
    conversation's next turn to close."""
    store, lease = _ClosesNothing(pg_engine), await _instance(pg_engine, clock)
    segment_id = await _claim(lease)
    refused = RuntimeError("plan_start broke")

    async def plan_start(ctx: NodeCtx, node_input: Any) -> Next:
        raise refused

    graph = Graph(start="plan_start", nodes={"plan_start": Node(run=plan_start)}, edges={})
    answer: Answer | None = None
    if paused:
        asking = _asking([])
        before = Graph(start="ask", nodes={"ask": asking}, edges={("ask", "next"): END})
        asked: list[TurnEvent] = []
        await _segment(
            store, lease, await _open(store, segment_id, before), segment_id, before, asked
        )
        [event] = asked
        assert isinstance(event, Paused)
        answer = Answer(interrupt_id=event.interrupt_id, round=1, value={})
        graph = Graph(start="ask", nodes={"ask": Node(run=asking.run)}, edges={})
        cursor = await _open_cursor(store)
    else:
        cursor = await _open(store, segment_id, graph)
    events: list[TurnEvent] = []
    segment = await _segment(store, lease, cursor, segment_id, graph, events, answer=answer)

    [failed] = events
    assert isinstance(failed, TurnFailed)
    assert isinstance(failed.error, TurnExpired) if paused else failed.error is refused
    held = TurnStatus.PAUSED if paused else TurnStatus.RUNNING
    assert segment.outcome is Outcome.UNCLOSED
    assert (await _open_cursor(store)).status is held


class _AcknowledgementLost(StudioStore):
    """A lease store whose first renewal after a Stop request commits and
    then raises, as a renewal whose acknowledgement is lost does."""

    lose_next = False

    async def renew(self, *args: Any, **kwargs: Any) -> Any:
        renewal = await super().renew(*args, **kwargs)
        if self.lose_next and renewal.stop_id is not None:
            self.lose_next = False
            raise TimeoutError("the renewal's acknowledgement was lost")
        return renewal


async def test_a_stop_whose_renewal_lost_its_acknowledgement_still_reaches_the_node(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    store = TurnStore(pg_engine)
    lease_a = _AcknowledgementLost(pg_engine, **clock.store_kwargs)
    lease_b = await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)
    switching = asyncio.Event()

    async def switch(ctx: NodeCtx, node_input: Any) -> Next:
        async with ctx.stop.open() as stopped:
            switching.set()
            await stopped.wait()
        return Next("offer")

    graph = Graph(
        start="switch", nodes={"switch": Node(run=switch)}, edges={("switch", "offer"): END}
    )
    running = asyncio.create_task(
        _segment(store, lease_a, await _open(store, segment_id, graph), segment_id, graph, [])
    )
    await switching.wait()
    lease_a.lose_next = True
    assert await lease_b.request_cancel(PRINCIPAL, CONVERSATION)
    segment = await asyncio.wait_for(running, 5)

    assert not lease_a.lose_next
    # The scope took the Stop, and the turn ended with no pause to record
    # it, so the turn closes on it.
    assert segment.outcome is Outcome.CANCELLED


class _RenewalsFail(StudioStore):
    """A `StudioStore` whose renewals fail, once `failing` is set, with the
    lease deadline passed, so the heartbeat ends the segment `lost` while
    the stored lease is still the segment's own."""

    failing = False

    async def renew(self, *args: Any, **kwargs: Any) -> Any:
        if not self.failing:
            return await super().renew(*args, **kwargs)
        raise RuntimeError("the database cannot be reached")

    def lease_deadline(self, principal: str, conversation_id: str) -> float:
        return float("-inf") if self.failing else super().lease_deadline(principal, conversation_id)


async def _cut_lease(pg_engine: AsyncEngine, clock: StoreClock, cut: str) -> StudioStore:
    """The lease of the instance a test cuts by `cut`: a Stop posted from
    another instance, or a lost lease (`_RenewalsFail`)."""
    if cut == "stop":
        return await _instance(pg_engine, clock)
    return _RenewalsFail(pg_engine, **clock.store_kwargs)


async def _make_cut(lease: StudioStore, other: StudioStore, cut: str) -> None:
    """Cut the segment running under `lease` (`_cut_lease`)."""
    if isinstance(lease, _RenewalsFail):
        lease.failing = True
        return
    assert cut == "stop"
    assert await other.request_cancel(PRINCIPAL, CONVERSATION)


@pytest.mark.parametrize(
    ("cut", "absorbs_by"),
    [
        ("stop", "returning"),
        ("stop", "raising"),
        ("stop", "after_commit"),
        ("lost", "returning"),
        ("lost", "after_final_commit"),
    ],
)
async def test_a_cut_settles_the_turn_when_its_node_absorbs_the_cancellation(
    pg_engine: AsyncEngine, clock: StoreClock, cut: str, absorbs_by: str
) -> None:
    """The heartbeat cancels the node, which catches the cancellation and
    returns a result, raises an error of its own, or lets its work after
    the commit catch it. The segment settles as when the cancellation
    escapes, and no later node runs. A Stop closes the turn `cancelled` at
    the node it cut and ends the Stop. A lost lease, renewals failing while
    the stored lease is still the segment's, leaves the turn running as
    last written, with nothing the node returned committed. A cut after
    the turn's end committed leaves it `done`, its `NodeDone` reported."""
    store = TurnStore(pg_engine)
    lease_a = await _cut_lease(pg_engine, clock, cut)
    lease_b = await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)
    ran: list[str] = []

    async def absorb() -> None:
        await _make_cut(lease_a, lease_b, cut)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        ctx.state["work"] = "returned"
        if absorbs_by in {"after_commit", "after_final_commit"}:
            ctx.after_commit(absorb)
            return Next("done" if absorbs_by == "after_final_commit" else "next")
        await absorb()
        if absorbs_by == "raising":
            raise RuntimeError("the node's own error after the cancellation")
        return Next("next")

    async def later(ctx: NodeCtx, node_input: Any) -> Next:
        ran.append("later")
        return Next("done")

    graph = Graph(
        start="work",
        nodes={"work": Node(run=work), "later": Node(run=later)},
        edges={("work", "next"): "later", ("work", "done"): END, ("later", "done"): END},
    )
    events: list[TurnEvent] = []
    opened = await _open(store, segment_id, graph)
    segment = await asyncio.wait_for(_segment(store, lease_a, opened, segment_id, graph, events), 5)

    assert ran == []
    assert not [event for event in events if isinstance(event, TurnFailed)]
    stored = await store.turn(opened.id)
    if absorbs_by == "after_final_commit":
        assert (segment.outcome, stored.status) == (Outcome.DONE, TurnStatus.DONE)
        assert events == [NodeDone(node="work", label="done", output=None)]
        return
    if cut == "lost":
        assert segment.outcome is Outcome.LOST
        assert (stored.status, stored.node, stored.version) == (
            TurnStatus.RUNNING,
            "work",
            opened.version,
        )
        assert "work" not in stored.state
        return
    assert segment.outcome is Outcome.CANCELLED
    assert (stored.status, stored.error) == (TurnStatus.CANCELLED, None)
    assert stored.node == ("later" if absorbs_by == "after_commit" else "work")
    assert not await lease_b.stop_live(PRINCIPAL, CONVERSATION)


class _CutAfterCommit(TurnStore):
    """A `TurnStore` whose first `advance`, or first close `done`, commits
    and then cuts the segment through `cut` before it returns, so the cut
    lands while the write's acknowledgement is awaited. With `absorb` the
    write catches the cancellation and returns, leaving the cut pending."""

    def __init__(
        self, engine: AsyncEngine, *, cut: Callable[[], Awaitable[None]], absorb: bool = False
    ) -> None:
        super().__init__(engine)
        self._cut = cut
        self._cuts = 1
        self._absorb = absorb

    async def _cut_once(self) -> None:
        if self._cuts:
            self._cuts -= 1
            await self._cut()
            if self._absorb:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(3600)
            else:
                await asyncio.sleep(3600)

    async def advance(self, *args: Any, **kwargs: Any) -> Cursor:
        written = await super().advance(*args, **kwargs)
        await self._cut_once()
        return written

    async def finish(self, *args: Any, **kwargs: Any) -> Cursor:
        written = await super().finish(*args, **kwargs)
        if kwargs.get("status") is TurnStatus.DONE:
            await self._cut_once()
        return written


@pytest.mark.parametrize("ends", [True, False], ids=["final-node", "middle-node"])
@pytest.mark.parametrize(
    "cut_at", ["acknowledgement", "absorbed_acknowledgement", "after_commit", "after_after_commit"]
)
async def test_a_cut_after_a_boundary_committed_reports_its_node_done_once(
    pg_engine: AsyncEngine, clock: StoreClock, cut_at: str, ends: bool
) -> None:
    """A lost lease cuts the segment once `work`'s move or end committed:
    while the write's acknowledgement is awaited, during `work`'s work
    after the commit, or once that work caught the cancellation and
    returned. `work`'s `NodeDone` is reported once, before the segment
    settles on the turn as stored: `done` for the final node, the move to
    `later` for a middle one, with `later` never run. A write that caught
    the cut and returned still has its work after the commit run (an
    acknowledgement recorded) before the segment settles on the cut."""
    lease_a = _RenewalsFail(pg_engine, **clock.store_kwargs)
    lease_b = await _instance(pg_engine, clock)

    async def cut() -> None:
        await _make_cut(lease_a, lease_b, "lost")

    store: TurnStore = (
        _CutAfterCommit(pg_engine, cut=cut, absorb=cut_at == "absorbed_acknowledgement")
        if cut_at in ("acknowledgement", "absorbed_acknowledgement")
        else TurnStore(pg_engine)
    )
    segment_id = await _claim(lease_a)
    ran: list[str] = []
    acknowledged: list[str] = []

    async def acknowledge() -> None:
        acknowledged.append("work")

    label = "done" if ends else "next"

    async def escaping() -> None:
        await cut()
        await asyncio.sleep(3600)

    async def absorbing() -> None:
        await cut()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        if cut_at == "after_commit":
            ctx.after_commit(escaping)
        elif cut_at == "after_after_commit":
            ctx.after_commit(absorbing)
        elif cut_at == "absorbed_acknowledgement":
            ctx.after_commit(acknowledge)
        return Next(label)

    async def later(ctx: NodeCtx, node_input: Any) -> Next:
        ran.append("later")
        return Next("done")

    graph = Graph(
        start="work",
        nodes={"work": Node(run=work), "later": Node(run=later)},
        edges={("work", "next"): "later", ("work", "done"): END, ("later", "done"): END},
    )
    events: list[TurnEvent] = []
    opened = await _open(store, segment_id, graph)
    segment = await asyncio.wait_for(_segment(store, lease_a, opened, segment_id, graph, events), 5)

    assert events == [NodeDone(node="work", label=label, output=None)]
    assert ran == []
    assert acknowledged == (["work"] if cut_at == "absorbed_acknowledgement" else [])
    stored = await TurnStore(pg_engine).turn(opened.id)
    if ends:
        assert (segment.outcome, stored.status) == (Outcome.DONE, TurnStatus.DONE)
    else:
        assert segment.outcome is Outcome.LOST
        assert (stored.status, stored.node) == (TurnStatus.RUNNING, "later")


class _TakenOverAtTheWrite(TurnStore):
    """A `TurnStore` whose first `advance` writes nothing of its own: segment
    B takes the conversation's lease and makes the same move from the same
    version, and the write then waits until the segment is cut."""

    async def advance(self, cursor: Cursor, **kwargs: Any) -> Cursor:
        async with self._engine.begin() as conn:
            await conn.execute(text("UPDATE conversations SET lease_segment_id = 'segment-b'"))
        await TurnStore(self._engine).advance(cursor, **{**kwargs, "segment_id": "segment-b"})
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def test_a_cut_reports_no_node_done_for_a_move_another_segment_wrote(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """Another segment takes the lease while `work`'s move is being written
    and writes that same move from the same version. The stored cursor is
    one version on, at `later`, but its writer is the other segment, so the
    cut segment ends `lost` and reports no `NodeDone`."""
    store = _TakenOverAtTheWrite(pg_engine)
    lease = await _instance(pg_engine, clock)
    segment_id = await _claim(lease)

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        return Next("next")

    graph = Graph(
        start="work",
        nodes={"work": Node(run=work), "later": _stays("later", ran=[])},
        edges={("work", "next"): "later"},
    )
    events: list[TurnEvent] = []
    opened = await _open(store, segment_id, graph)
    segment = await asyncio.wait_for(_segment(store, lease, opened, segment_id, graph, events), 5)

    stored = await store.turn(opened.id)
    assert (stored.version, stored.node, stored.written_by) == (
        opened.version + 1,
        "later",
        "segment-b",
    )
    assert segment.outcome is Outcome.LOST
    assert events == []


class _ReadBackFails(TurnStore):
    """A `TurnStore` whose `turn` reads fail once `failing` is set, as a
    database the lease was lost to is unreachable for the read back."""

    failing = False

    async def turn(self, turn_id: int) -> Cursor:
        if self.failing:
            raise ConnectionError("the turn could not be read back")
        return await super().turn(turn_id)


async def test_a_lost_lease_during_after_commit_work_leaves_the_segment_on_the_committed_cursor(
    pg_engine: AsyncEngine, clock: StoreClock
) -> None:
    """`work`'s move to `later` commits, and the lease is lost while its
    work after the commit awaits. The read back that would settle the turn
    fails, so the segment is lost holding the cursor the write committed,
    whose records the next turn keeps."""
    store = _ReadBackFails(pg_engine)
    lease_a = _RenewalsFail(pg_engine, **clock.store_kwargs)
    lease_b = await _instance(pg_engine, clock)
    segment_id = await _claim(lease_a)

    async def cut_while_working() -> None:
        store.failing = True
        await _make_cut(lease_a, lease_b, "lost")
        await asyncio.sleep(3600)

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        ctx.after_commit(cut_while_working)
        return Next("next")

    async def later(ctx: NodeCtx, node_input: Any) -> Next:
        return Next("done")

    graph = Graph(
        start="work",
        nodes={"work": Node(run=work), "later": Node(run=later)},
        edges={("work", "next"): "later", ("later", "done"): END},
    )
    opened = await _open(store, segment_id, graph)
    segment = await asyncio.wait_for(_segment(store, lease_a, opened, segment_id, graph, []), 5)

    assert segment.outcome is Outcome.LOST
    committed = await TurnStore(pg_engine).turn(opened.id)
    assert (committed.node, committed.version) == ("later", opened.version + 1)
    assert (segment.cursor.node, segment.cursor.version) == ("later", committed.version)


class _CloseFailsUnderACut(TurnStore):
    """A `TurnStore` whose first close cuts the segment through `cut`,
    catches the cancellation, and raises an error of its own, writing
    nothing."""

    def __init__(self, engine: AsyncEngine, *, cut: Callable[[], Awaitable[None]]) -> None:
        super().__init__(engine)
        self._cut = cut
        self._failed = False

    async def finish(self, *args: Any, **kwargs: Any) -> Cursor:
        if not self._failed:
            self._failed = True
            await self._cut()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(3600)
            raise RuntimeError("the close failed after the cancellation")
        return await super().finish(*args, **kwargs)


@pytest.mark.parametrize("closing", ["on_error", "on_taken_stop"])
async def test_a_close_that_fails_under_a_cut_settles_on_the_cut(
    pg_engine: AsyncEngine, clock: StoreClock, closing: str
) -> None:
    """The turn closes on a node's error, or on the Stop its scope took, and
    the close fails while a lost lease cuts the segment. The segment settles
    on the cut, `lost` with the turn left as last written, and reports no
    `TurnFailed`."""
    lease_a = _RenewalsFail(pg_engine, **clock.store_kwargs)
    lease_b = await _instance(pg_engine, clock)
    store = _CloseFailsUnderACut(pg_engine, cut=lambda: _make_cut(lease_a, lease_b, "lost"))
    segment_id = await _claim(lease_a)

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        if closing == "on_error":
            raise RuntimeError("the node failed")
        # The scope took a Stop and the node leaves on an edge no pause
        # records it on, so the turn closes on it.
        ctx.stop.taken = "stop_taken"
        return Next("done")

    graph = Graph(start="work", nodes={"work": Node(run=work)}, edges={("work", "done"): END})
    events: list[TurnEvent] = []
    opened = await _open(store, segment_id, graph)
    segment = await asyncio.wait_for(_segment(store, lease_a, opened, segment_id, graph, events), 5)

    assert segment.outcome is Outcome.LOST
    assert not [event for event in events if isinstance(event, TurnFailed)]
    stored = await store.turn(opened.id)
    assert (stored.status, stored.version) == (TurnStatus.RUNNING, opened.version)


class _WriteAbsorbsCut(TurnStore):
    """A `TurnStore` whose `advance` or `record_answer` (`absorbing`) cuts
    the segment through `cut`, catches the cancellation, and raises an
    error of its own, writing nothing."""

    def __init__(self, engine: AsyncEngine, *, absorbing: str, cut: Callable[[], Awaitable[None]]):
        super().__init__(engine)
        self._absorbing = absorbing
        self._cut = cut

    async def _absorb(self) -> None:
        await self._cut()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)
        raise RuntimeError("the write failed after the cancellation")

    async def advance(self, *args: Any, **kwargs: Any) -> Cursor:
        if self._absorbing == "write":
            await self._absorb()
        return await super().advance(*args, **kwargs)

    async def record_answer(self, *args: Any, **kwargs: Any) -> Cursor:
        if self._absorbing == "answer":
            await self._absorb()
        return await super().record_answer(*args, **kwargs)


@pytest.mark.parametrize("cut", ["stop", "lost"])
@pytest.mark.parametrize("absorbing", ["write", "answer"])
async def test_a_cut_a_boundary_write_absorbs_settles_the_turn_without_failing_it(
    pg_engine: AsyncEngine, clock: StoreClock, cut: str, absorbing: str
) -> None:
    """The heartbeat cuts the segment inside the move's write after `work`,
    or inside the acceptance of the answer to `ask`'s pause, and the write
    catches the cancellation and raises an error of its own. The turn is
    not closed `failed`: a Stop closes it `cancelled`, and a lost lease
    leaves it as last written."""
    lease_a = await _cut_lease(pg_engine, clock, cut)
    lease_b = await _instance(pg_engine, clock)
    store = _WriteAbsorbsCut(
        pg_engine, absorbing=absorbing, cut=lambda: _make_cut(lease_a, lease_b, cut)
    )
    calls: list[str] = []

    async def work(ctx: NodeCtx, node_input: Any) -> Next:
        return Next("next")

    graph = Graph(
        start="ask" if absorbing == "answer" else "work",
        nodes={"ask": _asking(calls), "work": Node(run=work)},
        edges={("ask", "next"): END, ("work", "next"): "ask"},
    )
    segment_id = await _claim(lease_a)
    events: list[TurnEvent] = []
    answer: Answer | None = None
    opened = await _open(store, segment_id, graph)
    if absorbing == "answer":
        await _segment(store, lease_a, opened, segment_id, graph, events)
        [paused] = events
        assert isinstance(paused, Paused)
        answer = Answer(interrupt_id=paused.interrupt_id, round=1, value={"donor": "DAC"})
        opened = await _open_cursor(store)
        events.clear()
    segment = await asyncio.wait_for(
        _segment(store, lease_a, opened, segment_id, graph, events, answer=answer), 5
    )

    assert not [event for event in events if isinstance(event, TurnFailed)]
    assert calls == (["run"] if absorbing == "answer" else [])
    stored = await store.turn(opened.id)
    if cut == "stop":
        assert segment.outcome is Outcome.CANCELLED
        assert (stored.status, stored.error) == (TurnStatus.CANCELLED, None)
        return
    assert segment.outcome is (Outcome.PAUSED if absorbing == "answer" else Outcome.LOST)
    assert (stored.status, stored.version) == (opened.status, opened.version)


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
