"""A planned turn's run left unfinished, carried on through the Studio routes
over the conversation's stored run ledger (`harness.run_ledger.StudioRunLedger`):
a publication left pending, and a computation a client left mid-run."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import harness.studio as studio_module
import harness.turn.store as turn_store_module
import httpx2
import pytest
from compiler.surface import ProgressCallback, ToolCallResult
from executor.completion import PUBLISH_ATTEMPTS
from executor.ledger import IdempotencyConflictError, RunRecord, RunStatus, idempotency_key
from fastapi import FastAPI
from google.genai import types
from harness.conversation import current_conversation_key
from harness.graph import answer as answer_module
from harness.graph import execute as execute_module
from harness.graph.state import frame_id_for
from harness.run_ledger import StudioRunLedger
from harness.scripted import ScriptedSurface
from harness.studio_state import SEGMENT_TTL_S
from harness.studio_store import SegmentClose, StudioStore
from harness.turn.store import TurnStatus
from planner.workflow import ToolId
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import agent_read
from .scripted_tools import chart, make_charts_call, plan_call
from .store_clock import StoreClock
from .studio_routes import AUTH, orchestrator_app, stream, studio_client, turn_body
from .test_run_completion import _FlakyPublisher
from .turn_driver import CONVERSATION

# The principal `tests.studio_routes.AUTH`'s token names.
PRINCIPAL = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]


def _frame(frames: list[dict[str, Any]], event: str) -> dict[str, Any]:
    return next(frame for frame in frames if frame["event"] == event)


async def test_a_studio_run_left_pending_is_carried_on_from_the_conversation_ledger(
    monkeypatch: pytest.MonkeyPatch, pg_engine: AsyncEngine
) -> None:
    """Through the studio, over its stored run ledger: a run whose
    publication is unfinished at the end of its turn is kept under a key that
    names the conversation, not the turn, and the next turn, in a session of
    its own, completes it and hands the agent its frames and the notes read
    from the plan stored with the ledger."""
    total = "The figures are the sum of 3 indicators."
    monkeypatch.setattr(answer_module, "_stat_var_totals", lambda *a, **k: [total])
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    publisher = _FlakyPublisher()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="It is done."),
        tool_surface=ScriptedSurface(),
        store=store,
        run_ledger=ledger,
        publisher=publisher,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        plan = _frame(first, "plan.proposed")
        await stream(
            client,
            "/api/studio/interrupt",
            {
                "kind": "interrupt.response",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "interrupt_id": plan["interrupt_id"],
                "decision": "approve",
            },
        )
        after_plan = agent_read(model.requests[-1])
        body = json.loads(
            next(line for line in after_plan.split("\n") if line.startswith('{"frames"'))
        )
        assert (body["run"]["status"], body["frames"]) == ("pending", [])

        publisher.fixed = True
        await stream(client, "/api/studio/turn", turn_body("Is it ready?"))

    next_turn = agent_read(model.requests[-1])
    runs = json.loads(next(line for line in next_turn.split("\n") if line.startswith('{"runs"')))
    (resumed,) = runs["runs"]
    assert (resumed["run_id"], resumed["status"]) == (body["run"]["run_id"], "completed")
    assert [frame["frame"] for frame in resumed["frames"]]
    assert resumed["totals"] == [total]
    token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
    try:
        run = await ledger.get(resumed["run_id"])
    finally:
        current_conversation_key.reset(token)
    assert run is not None and run.status is RunStatus.COMPLETED
    assert run.idempotency_key == idempotency_key(
        caller=PRINCIPAL,
        conversation_id=f"{PRINCIPAL}/{CONVERSATION}",
        task=f"plan:{plan['revision']}",
    )


async def test_the_default_ledger_keeps_one_conversations_runs_from_another(
    pg_engine: AsyncEngine,
) -> None:
    publisher = _FlakyPublisher()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        types.Part.from_text(text="Handed to the planner."),
        types.Part.from_text(text="It is pending."),
        types.Part.from_text(text="Another conversation."),
        tool_surface=ScriptedSurface(),
        publisher=publisher,
        engine=pg_engine,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        await stream(
            client,
            "/api/studio/interrupt",
            {
                "kind": "interrupt.response",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "interrupt_id": _frame(first, "plan.proposed")["interrupt_id"],
                "decision": "approve",
            },
        )
        assert '"status": "pending"' in agent_read(model.requests[-1])

        publisher.fixed = True
        other = turn_body("Hello", conversation_id="conv_01other000")
        await stream(client, "/api/studio/turn", other)

    assert '{"runs"' not in agent_read(model.requests[-1])
    assert len(publisher.frames) == PUBLISH_ATTEMPTS


# --- carry-on after a client leaves mid-run -------------------------------------------

# The frame the canned plan's `aggregate` step answers with
# (`harness.scripted_plan.scripted_aggregate_result`).
PLAN_FRAME = frame_id_for({"spec_id": "ops_" + "9" * 32, "content_version": "ocv_" + "9" * 32})


class _HeldAggregate(ScriptedSurface):
    """`ScriptedSurface` whose `aggregate` calls wait until `release` is set,
    `entered` set once the first arrives, so the run is mid-computation, its
    retrievals sealed."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls.append(tool.name)
        if tool.name == "aggregate":
            self.entered.set()
            await self.release.wait()
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


def _charts_the_plan() -> tuple[types.Part, ...]:
    """What the agent does once the plan's run is handed back: one card on
    its frame, then its answer."""
    return (
        make_charts_call(chart(PLAN_FRAME, type_="stat_card", title="Aid to Kenya")),
        types.Part.from_text(text="Charted."),
    )


def _approve_body(interrupt_id: str) -> dict[str, Any]:
    return {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "interrupt_id": interrupt_id,
        "decision": "approve",
    }


async def _leaves_when(
    app: FastAPI, path: str, body: Mapping[str, Any], *, leave: asyncio.Event
) -> list[dict[str, Any]]:
    """POST `body` to `path` as a client that leaves once `leave` is set, and
    return the frames it read before. The server reads `http.disconnect`,
    as uvicorn delivers it, and Starlette cancels the response. The
    in-process client cannot leave mid-stream, so this drives the ASGI app
    itself."""
    raw = json.dumps(body).encode()
    sent = False
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await leave.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
            (b"authorization", AUTH["Authorization"].encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("studio.test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return [
        json.loads(line.removeprefix("data: "))
        for line in b"".join(chunks).decode().splitlines()
        if line.startswith("data: ")
    ]


async def _runs(engine: AsyncEngine, ledger: StudioRunLedger) -> list[RunRecord]:
    """Every run `ledger` keeps for the test's conversation, finished or not."""
    async with engine.connect() as conn:
        ids = (
            await conn.execute(
                text(
                    "SELECT r.run_id FROM runs r JOIN conversations c ON c.id = r.conversation "
                    "WHERE c.principal = :principal AND c.conversation_id = :conversation_id"
                ),
                {"principal": PRINCIPAL, "conversation_id": CONVERSATION},
            )
        ).scalars()
    token = current_conversation_key.set((PRINCIPAL, CONVERSATION))
    try:
        runs = [await ledger.get(run_id) for run_id in ids]
    finally:
        current_conversation_key.reset(token)
    return [run for run in runs if run is not None]


async def _left_mid_run(app: FastAPI, client: httpx2.AsyncClient, surface: _HeldAggregate) -> None:
    """Propose the plan, approve it, and leave while its run's `aggregate`
    call is in flight; the call then answers for whoever carries the run on."""
    first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
    plan = _frame(first, "plan.proposed")
    left = await _leaves_when(
        app, "/api/studio/interrupt", _approve_body(plan["interrupt_id"]), leave=surface.entered
    )
    assert "done" not in [frame["event"] for frame in left]
    surface.release.set()


@pytest.mark.parametrize("next_request", ["turn", "interrupt"])
async def test_a_run_a_client_left_is_carried_on_to_its_charts_by_the_next_request(
    pg_engine: AsyncEngine, next_request: str
) -> None:
    """The client leaves while the approved plan's run computes. The run is
    left unfinished, not failed, and the conversation's turn left running
    at `execute`. The next request, sent at once, carries the same run on
    and the agent charts its frame: a `/turn` abandons the open turn and
    its `seed` hands the run back; a retried `/interrupt` runs `execute`
    again from the cursor."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    surface = _HeldAggregate()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        *_charts_the_plan(),
        tool_surface=surface,
        store=store,
        run_ledger=ledger,
    )
    async with studio_client(app) as client:
        await _left_mid_run(app, client, surface)
        (left,) = await _runs(pg_engine, ledger)
        assert left.status is RunStatus.RUNNING
        cursor = await app.state.studio.turns.open_cursor(PRINCIPAL, CONVERSATION)
        assert cursor is not None
        assert (cursor.status, cursor.node) == (TurnStatus.RUNNING, "execute")

        if next_request == "turn":
            frames = await stream(client, "/api/studio/turn", turn_body("Is it ready?"))
            assert '{"runs"' in agent_read(model.requests[-2])
        else:
            pending = store.current(PRINCIPAL, CONVERSATION).pending
            (interrupt_id,) = pending
            frames = await stream(client, "/api/studio/interrupt", _approve_body(interrupt_id))

    events = [frame["event"] for frame in frames]
    assert "component.created" in events
    assert events[-1] == "done"
    (run,) = await _runs(pg_engine, ledger)
    assert (run.run_id, run.status) == (left.run_id, RunStatus.COMPLETED)
    assert surface.calls.count("aggregate") == 2


async def test_a_run_left_by_an_instance_that_died_is_carried_on_by_another_once_its_lease_lapses(
    pg_engine: AsyncEngine,
) -> None:
    """Instance A's client leaves mid-run and A dies before it lets go of
    the conversation's lease. Instance B's `/turn` is refused while the
    lease lives, and once it lapses B abandons A's turn, carries A's run on
    from the stored ledger and charts its frame."""
    clock = await StoreClock.start(pg_engine)
    store_a = StudioStore(pg_engine, **clock.store_kwargs)
    store_b = StudioStore(pg_engine, **clock.store_kwargs)
    surface_a = _HeldAggregate()
    app_a, _ = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        tool_surface=surface_a,
        store=store_a,
        run_ledger=StudioRunLedger(store_a),
    )
    surface_b = ScriptedSurface()
    app_b, model_b = orchestrator_app(
        *_charts_the_plan(),
        tool_surface=surface_b,
        store=store_b,
        run_ledger=StudioRunLedger(store_b),
    )

    async def dies(*args: Any, **kwargs: Any) -> SegmentClose:
        """A's closing write, which a dead instance never makes."""
        return SegmentClose.FAILED

    async with studio_client(app_a) as client_a:
        first = await stream(client_a, "/api/studio/turn", turn_body("Kenya's debt service?"))
        store_a.close_segment = dies  # type: ignore[method-assign]
        await _leaves_when(
            app_a,
            "/api/studio/interrupt",
            _approve_body(_frame(first, "plan.proposed")["interrupt_id"]),
            leave=surface_a.entered,
        )
    (left,) = await _runs(pg_engine, StudioRunLedger(store_b))

    async with studio_client(app_b) as client_b:
        busy = await client_b.post("/api/studio/turn", json=turn_body("Ready?"), headers=AUTH)
        assert busy.status_code == 409
        await clock.advance(SEGMENT_TTL_S + 1)
        frames = await stream(client_b, "/api/studio/turn", turn_body("Ready?"))

    assert "component.created" in [frame["event"] for frame in frames]
    assert '{"runs"' in agent_read(model_b.requests[0])
    (run,) = await _runs(pg_engine, StudioRunLedger(store_b))
    assert (run.run_id, run.status) == (left.run_id, RunStatus.COMPLETED)


async def test_a_stop_during_a_run_records_it_cancelled_and_nothing_carries_it_on(
    pg_engine: AsyncEngine,
) -> None:
    """A Stop while the approved plan's run computes ends the turn
    `cancelled` and records the run failed `cancelled`, so the next turn
    hands no run back and no step runs again."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    surface = _HeldAggregate()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        tool_surface=surface,
        store=store,
        run_ledger=ledger,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        plan = _frame(first, "plan.proposed")
        running = asyncio.ensure_future(
            stream(client, "/api/studio/interrupt", _approve_body(plan["interrupt_id"]))
        )
        await asyncio.wait_for(surface.entered.wait(), timeout=10)
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        stopped = await asyncio.wait_for(running, timeout=10)
        surface.release.set()
        await stream(client, "/api/studio/turn", turn_body("Hello"))

    assert _frame(stopped, "error")["error"]["error"] == "cancelled"
    (run,) = await _runs(pg_engine, ledger)
    assert run.status is RunStatus.FAILED
    assert run.failure is not None and run.failure.code == "cancelled"
    assert '{"runs"' not in agent_read(model.requests[-1])
    assert surface.calls.count("aggregate") == 1


async def test_a_client_that_leaves_mid_model_call_leaves_the_turn_for_the_next_one(
    pg_engine: AsyncEngine,
) -> None:
    """The client leaves while the agent's model call streams. The turn's
    cursor stays running where the engine left it and the lease is let go,
    so the next `/turn` streams at once, closing the left turn `abandoned`."""
    store = StudioStore(pg_engine)
    calling = asyncio.Event()

    async def hangs(request: Any) -> types.Part:
        calling.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    app, _ = orchestrator_app(hangs, types.Part.from_text(text="Hello."), store=store)
    await _leaves_when(app, "/api/studio/turn", turn_body("Kenya's debt service?"), leave=calling)
    turns = app.state.studio.turns
    left = await turns.open_cursor(PRINCIPAL, CONVERSATION)
    assert left is not None
    assert (left.status, left.node) == (TurnStatus.RUNNING, "orchestrate")

    async with studio_client(app) as client:
        frames = await stream(client, "/api/studio/turn", turn_body("Hello?"))

    assert frames[-1]["event"] == "done"
    assert (await turns.turn(left.id)).status is TurnStatus.ABANDONED


async def test_a_stop_after_the_client_left_ends_the_run_it_left(pg_engine: AsyncEngine) -> None:
    """The client leaves mid-run, then a Stop arrives with no segment live.
    The Stop closes the turn `cancelled` and records its run failed
    `cancelled`, so the next turn hands no run back."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    surface = _HeldAggregate()
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        tool_surface=surface,
        store=store,
        run_ledger=ledger,
    )
    async with studio_client(app) as client:
        await _left_mid_run(app, client, surface)
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        await stream(client, "/api/studio/turn", turn_body("Hello"))

    (run,) = await _runs(pg_engine, ledger)
    assert run.status is RunStatus.FAILED
    assert run.failure is not None and run.failure.code == "cancelled"
    assert '{"runs"' not in agent_read(model.requests[-1])
    assert surface.calls.count("aggregate") == 1
    assert store.current(PRINCIPAL, CONVERSATION).turns[0]["error"] == "cancelled"


async def test_a_stop_that_cannot_end_the_runs_left_answers_retry_and_leaves_the_turn_open(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Stop with no live segment whose run ledger cannot record the left
    run stopped answers a retryable 503 and changes nothing, so the run is
    never carried on as though the user had not stopped it; the Stop
    retried once the ledger is back ends both."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    end_unfinished = turn_store_module.end_unfinished
    down = [True]

    async def unless_down(*args: Any, **kwargs: Any) -> None:
        if down[0]:
            raise ConnectionError("the run store is unreachable")
        await end_unfinished(*args, **kwargs)

    monkeypatch.setattr(turn_store_module, "end_unfinished", unless_down)
    surface = _HeldAggregate()
    app, _ = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        tool_surface=surface,
        store=store,
        run_ledger=ledger,
    )
    cancel_body = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}
    async with studio_client(app) as client:
        await _left_mid_run(app, client, surface)
        refused = await client.post("/api/studio/cancel", json=cancel_body, headers=AUTH)
        left = await app.state.studio.turns.open_cursor(PRINCIPAL, CONVERSATION)
        (unended,) = await _runs(pg_engine, ledger)
        down[0] = False
        retried = await client.post("/api/studio/cancel", json=cancel_body, headers=AUTH)

    assert refused.status_code == 503
    assert refused.json()["retry"] == "retry_after_delay"
    assert left is not None and left.status is TurnStatus.RUNNING
    assert unended.status in (RunStatus.ADMITTED, RunStatus.RUNNING)
    assert retried.status_code == 204
    (run,) = await _runs(pg_engine, ledger)
    assert run.failure is not None and run.failure.code == "cancelled"
    assert await app.state.studio.turns.open_cursor(PRINCIPAL, CONVERSATION) is None
    assert store.current(PRINCIPAL, CONVERSATION).turns[0]["error"] == "cancelled"


# --- the approval admits its run ------------------------------------------------


class _HangsOnce:
    """Stands in for async callable `real`: its first call sets `entered`
    and waits until cancelled, and every later call is `real`'s."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.entered = asyncio.Event()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.entered.is_set():
            self.entered.set()
            await asyncio.Event().wait()
        return await self.real(*args, **kwargs)


def _hang_at(app: FastAPI, monkeypatch: pytest.MonkeyPatch, where: str) -> _HangsOnce:
    """Hang the approval's resume once, at `where`: `record_answer`, the
    turn store's write accepting the answer, the first write after the
    route's save; or `execute`, the node's first await, after the boundary
    write moved the cursor there."""
    if where == "record_answer":
        turns = app.state.studio.turns
        hang = _HangsOnce(turns.record_answer)
        monkeypatch.setattr(turns, "record_answer", hang)
    else:
        hang = _HangsOnce(execute_module.store_run_plan)
        monkeypatch.setattr(execute_module, "store_run_plan", hang)
    return hang


async def _approvals(engine: AsyncEngine) -> list[str]:
    """The interrupt ids the stored conversation has approvals on record for."""
    stored = await StudioStore(engine).load(PRINCIPAL, CONVERSATION)
    assert stored is not None
    return [record.interrupt_id for record in stored.approvals]


@pytest.mark.parametrize("next_request", ["turn", "interrupt"])
@pytest.mark.parametrize("where", ["record_answer", "execute"])
async def test_an_approval_a_client_left_before_execute_admitted_is_carried_on_by_the_next_request(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, where: str, next_request: str
) -> None:
    """The client leaves after the route saved the approval and before
    `execute` admits a run: before the turn store accepts the answer, or
    once the cursor stands at `execute`. The save admitted the run with the
    approval, so the stored approval and the run row read together. The
    next request carries that one run on to its chart with no new plan: a
    `/turn` through its `seed`, a retried `/interrupt` of the same approval
    through `execute`, which admits nothing new. The run never started
    before, so its first computation counts no resume."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    app, model = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        *_charts_the_plan(),
        tool_surface=ScriptedSurface(),
        store=store,
        run_ledger=ledger,
    )
    hang = _hang_at(app, monkeypatch, where)
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        interrupt_id = _frame(first, "plan.proposed")["interrupt_id"]
        await _leaves_when(
            app, "/api/studio/interrupt", _approve_body(interrupt_id), leave=hang.entered
        )
        assert await _approvals(pg_engine) == [interrupt_id]
        (admitted,) = await _runs(pg_engine, ledger)
        assert admitted.status is RunStatus.ADMITTED

        if next_request == "turn":
            frames = await stream(client, "/api/studio/turn", turn_body("Is it ready?"))
            assert '{"runs"' in agent_read(model.requests[-2])
        else:
            frames = await stream(client, "/api/studio/interrupt", _approve_body(interrupt_id))

    events = [frame["event"] for frame in frames]
    assert "plan.proposed" not in events
    assert "component.created" in events
    assert await _approvals(pg_engine) == [interrupt_id]
    (run,) = await _runs(pg_engine, ledger)
    assert (run.run_id, run.status, run.resumes) == (admitted.run_id, RunStatus.COMPLETED, 0)


async def test_an_approval_whose_run_cannot_be_admitted_records_nothing_and_stays_answerable(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database fault while the approval's save admits its run rolls the
    whole save back: no approval and no run are stored, and the same pause
    answered again runs the plan."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    app, _ = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        *_charts_the_plan(),
        tool_surface=ScriptedSurface(),
        store=store,
        run_ledger=ledger,
    )
    admit_on = studio_module.admit_on

    async def admits_then_faults(*args: Any, **kwargs: Any) -> Any:
        await admit_on(*args, **kwargs)
        raise ConnectionError("the database went away")

    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        interrupt_id = _frame(first, "plan.proposed")["interrupt_id"]
        with monkeypatch.context() as patched:
            patched.setattr(studio_module, "admit_on", admits_then_faults)
            with pytest.raises(ConnectionError):
                await client.post(
                    "/api/studio/interrupt", json=_approve_body(interrupt_id), headers=AUTH
                )
        assert await _approvals(pg_engine) == []
        assert await _runs(pg_engine, ledger) == []

        frames = await stream(client, "/api/studio/interrupt", _approve_body(interrupt_id))

    assert "component.created" in [frame["event"] for frame in frames]
    assert await _approvals(pg_engine) == [interrupt_id]
    (run,) = await _runs(pg_engine, ledger)
    assert run.status is RunStatus.COMPLETED


async def test_an_approval_whose_key_names_other_work_is_recorded_without_a_run(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admission the ledger refuses, as when the key names a run of other
    work, is undone alone: the approval's save still commits, with no run
    and no stored plan, for `execute` to surface."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    app, _ = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        tool_surface=ScriptedSurface(),
        store=store,
        run_ledger=ledger,
    )
    admit_on = studio_module.admit_on

    async def admits_then_conflicts(conn: Any, pk: int, candidate: RunRecord, **kwargs: Any) -> Any:
        await admit_on(conn, pk, candidate, **kwargs)
        raise IdempotencyConflictError(candidate, "sha_other")

    monkeypatch.setattr(studio_module, "admit_on", admits_then_conflicts)
    hang = _hang_at(app, monkeypatch, "record_answer")
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        interrupt_id = _frame(first, "plan.proposed")["interrupt_id"]
        await _leaves_when(
            app, "/api/studio/interrupt", _approve_body(interrupt_id), leave=hang.entered
        )

    assert await _approvals(pg_engine) == [interrupt_id]
    assert await _runs(pg_engine, ledger) == []
    async with pg_engine.connect() as conn:
        plans = (await conn.execute(text("SELECT count(*) FROM run_plans"))).scalar_one()
    assert plans == 0


async def test_a_stop_after_an_approval_ends_the_run_it_admitted_and_the_next_turn_plans(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Stop after the approval, while `execute` has not yet started its
    run, closes the turn `cancelled` and records the run the approval
    admitted failed `cancelled`, so the next turn carries nothing on and
    plans again."""
    store = StudioStore(pg_engine)
    ledger = StudioRunLedger(store)
    app, _ = orchestrator_app(
        plan_call("Kenya's debt service since 2015"),
        plan_call("Kenya's debt service since 2016"),
        tool_surface=ScriptedSurface(),
        store=store,
        run_ledger=ledger,
    )
    hang = _hang_at(app, monkeypatch, "execute")
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("Kenya's debt service?"))
        plan = _frame(first, "plan.proposed")
        running = asyncio.ensure_future(
            stream(client, "/api/studio/interrupt", _approve_body(plan["interrupt_id"]))
        )
        await asyncio.wait_for(hang.entered.wait(), timeout=10)
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        await asyncio.wait_for(running, timeout=10)
        frames = await stream(client, "/api/studio/turn", turn_body("And since 2016?"))

    (run,) = await _runs(pg_engine, ledger)
    assert run.status is RunStatus.FAILED
    assert run.failure is not None and run.failure.code == "cancelled"
    assert "plan.proposed" in [frame["event"] for frame in frames]
