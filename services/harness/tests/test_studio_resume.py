"""`POST /api/studio/resume` and `GET /api/studio/state`.

A client that leaves a running turn takes the drive with it: the stream's
owner cancels the segment, the closing write clears the lease, and the turn
stays open at its cursor. These tests leave a turn that way over the ASGI
interface, as a browser closing the page does, then read the conversation's
state and carry the turn on with `/resume`.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx2
import pytest
from compiler.surface import ProgressCallback, ToolCallResult
from fastapi import FastAPI
from google.genai import types
from harness.scripted import ScriptedSurface
from harness.studio import _conversation_read, _read_status, _resume_refusal
from harness.studio_state import SEGMENT_TTL_S, ConversationState, _PendingInterrupt
from harness.studio_store import (
    SegmentClose,
    Standing,
    StoredConversation,
    StudioStore,
    TurnStanding,
    record_status,
    turn_standing,
)
from harness.turn.store import TurnStatus, TurnStore
from harness.wire import TURN_INDEX_HEADER
from planner.workflow import ToolId
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_tools import (
    CannedSearch,
    chart,
    definite_envelope,
    fetch_call,
    make_charts_call,
    plan_call,
    search_call,
)
from .store_clock import StoreClock
from .studio_routes import (
    AUTH,
    frame_for,
    orchestrator_app,
    stream,
    studio_client,
    turn_body,
)
from .turn_driver import CONVERSATION

# The principal `AUTH`'s bearer token resolves to.
PRINCIPAL = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
# Every wait in this file is bounded by this, so a turn that hangs fails.
_DEADLINE_S = 10.0


def _text(text: str) -> types.Part:
    return types.Part.from_text(text=text)


class _HangOnce:
    """A search that hangs on its first call, as a slow upstream does while
    the client leaves, and refuses any later one."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if self.entered.is_set():
            raise AssertionError("the resumed turn searched again")
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _StepHangsOnce(ScriptedSurface):
    """`ScriptedSurface`, but the first plan step's call hangs until the
    client leaves; every later call answers as `ScriptedSurface` does."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if not self.entered.is_set():
            self.entered.set()
            await asyncio.Event().wait()
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


def _frames(chunks: list[bytes]) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for chunk in chunks
        for line in chunk.decode().splitlines()
        if line.startswith("data: ")
    ]


async def _post_then_leave(
    app: FastAPI, path: str, body: Mapping[str, Any], leave: asyncio.Event
) -> list[dict[str, Any]]:
    """POST `body` to `path` over the ASGI interface and read the stream
    until `leave` is set, when the client disconnects. Returns the frames
    read once the response has returned, which it does only after the
    stream's owner has made its closing write."""
    chunks: list[bytes] = []
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await leave.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"studio.test"),
            (b"content-type", b"application/json"),
            (b"authorization", AUTH["Authorization"].encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("studio.test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=_DEADLINE_S)
    return _frames(chunks)


async def _leave_once(
    app: FastAPI, path: str, body: Mapping[str, Any], entered: asyncio.Event
) -> list[dict[str, Any]]:
    """POST `body` and leave once `entered` is set: the client closes the page
    while the turn waits on a call."""
    leave = asyncio.Event()
    posting = asyncio.ensure_future(_post_then_leave(app, path, body, leave))
    await asyncio.wait_for(entered.wait(), timeout=_DEADLINE_S)
    leave.set()
    return await posting


def _resume_body(turn_index: int, conversation_id: str = CONVERSATION) -> dict[str, Any]:
    return {
        "kind": "turn.resume",
        "schema_version": 1,
        "conversation_id": conversation_id,
        "turn_index": turn_index,
    }


async def _state(client: httpx2.AsyncClient, **params: Any) -> httpx2.Response:
    return await client.get(
        "/api/studio/state",
        params={"conversation_id": CONVERSATION, **params},
        headers=AUTH,
    )


def _events(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["event"] for frame in frames]


async def test_a_resume_carries_a_turn_its_client_left_on_to_its_pause(
    pg_engine: AsyncEngine,
) -> None:
    """The client leaves while the turn's first search hangs. The turn stays
    running with no lease, and the state read says so. `/resume` runs the
    node at the cursor again and the turn reaches its plan card, which the
    state read then returns as the open pause, with the frames it was sent
    as."""
    searching = _HangOnce()
    app, _ = orchestrator_app(
        search_call("aid to Kenya"), plan_call("q with forks"), resolver=searching, engine=pg_engine
    )

    left = await _leave_once(app, "/api/studio/turn", turn_body("q with forks"), searching.entered)
    assert "plan.proposed" not in _events(left)

    async with studio_client(app) as client:
        before = await _state(client, since=0)
        assert before.status_code == 200
        assert before.json()["turn"] == {
            "index": 1,
            "status": "running",
            "trace_id": before.json()["turn"]["trace_id"],
            "user_message": "q with forks",
            "lease_live": False,
        }
        assert "pause" not in before.json()

        resumed = await stream(client, "/api/studio/resume", _resume_body(1))
        assert _events(resumed)[-1] == "plan.proposed"
        plan = resumed[-1]

        after = (await _state(client, since=0)).json()

    assert after["turn"]["status"] == "paused"
    assert after["turn"]["trace_id"] == before.json()["turn"]["trace_id"]
    assert after["pause"]["interrupt_id"] == plan["interrupt_id"]
    assert after["pause"]["kind"] == "plan"
    assert after["pause"]["frames"][-1] == plan
    assert after["closed_since"] == []


async def test_a_resume_drops_the_answer_its_departed_client_left_resuming(
    pg_engine: AsyncEngine,
) -> None:
    """The client approves the plan and leaves while its first step runs,
    which keeps the answered entry resuming. `/resume` marks it under its
    own segment, so the turn's close drops it, and the turn's reply reaches
    the state read."""
    surface = _StepHangsOnce()
    app, _ = orchestrator_app(
        plan_call("q with forks"),
        _text("Planned and run."),
        tool_surface=surface,
        engine=pg_engine,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("q with forks"))
    plan = first[-1]
    assert plan["event"] == "plan.proposed"

    await _leave_once(
        app,
        "/api/studio/interrupt",
        {
            "kind": "interrupt.response",
            "schema_version": 1,
            "conversation_id": CONVERSATION,
            "interrupt_id": plan["interrupt_id"],
            "decision": "approve",
        },
        surface.entered,
    )

    async with studio_client(app) as client:
        resumed = await stream(client, "/api/studio/resume", _resume_body(1))
        read = (await _state(client, since=0)).json()

    assert _events(resumed)[-1] == "done"
    assert read["turn"]["status"] == "done"
    assert read["closed_since"][0]["reply"].startswith("Planned and run.")
    stored = await StudioStore(pg_engine).load(PRINCIPAL, CONVERSATION)
    assert stored is not None
    assert stored.pending == {}


async def test_the_state_read_shows_a_live_lease_and_resume_waits_for_it(
    pg_engine: AsyncEngine,
) -> None:
    """While a segment holds the turn, the state read says the lease is live
    and for how long, and `/resume` is refused as busy with a retry time."""
    searching = _HangOnce()
    app, _ = orchestrator_app(search_call("aid to Kenya"), resolver=searching, engine=pg_engine)
    leave = asyncio.Event()
    posting = asyncio.ensure_future(
        _post_then_leave(app, "/api/studio/turn", turn_body("aid to Kenya"), leave)
    )
    try:
        await asyncio.wait_for(searching.entered.wait(), timeout=_DEADLINE_S)
        async with studio_client(app) as client:
            read = (await _state(client)).json()
            busy = await client.post("/api/studio/resume", json=_resume_body(1), headers=AUTH)
    finally:
        leave.set()
        await posting

    assert read["turn"]["status"] == "running"
    assert read["turn"]["lease_live"] is True
    assert read["turn"]["retry_after_seconds"] > 0
    assert busy.status_code == 409
    assert busy.json()["error"] == "transient_unavailable"


async def test_closed_since_returns_the_turns_after_the_one_the_client_holds(
    pg_engine: AsyncEngine,
) -> None:
    """Each closed turn after `since` comes back with its message and reply,
    oldest first, at most three, beside `closed_total`. A page that reads
    again from the last turn it got until it holds `closed_total` gets every
    turn once, in order."""
    replies = ("One.", "Two.", "Three.", "Four.", "Five.", "Six.", "Seven.")
    messages = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")
    app, _ = orchestrator_app(*(_text(reply) for reply in replies), engine=pg_engine)
    async with studio_client(app) as client:
        for message in messages:
            await stream(client, "/api/studio/turn", turn_body(message))
        since_one = (await _state(client, since=1)).json()
        since_six = (await _state(client, since=6)).json()
        unnamed = (await _state(client)).json()
        paged: list[dict[str, Any]] = []
        reads = 0
        while True:
            read = (await _state(client, since=len(paged))).json()
            reads += 1
            paged.extend(read["closed_since"])
            if len(paged) >= read["closed_total"]:
                break

    assert [
        (turn["turn_index"], turn["user_message"], turn["reply"])
        for turn in since_one["closed_since"]
    ] == [(2, "second", "Two."), (3, "third", "Three."), (4, "fourth", "Four.")]
    assert [turn["turn_index"] for turn in since_six["closed_since"]] == [7]
    assert since_one["turn"]["status"] == "done"
    assert since_one["turn"]["index"] == 7
    assert unnamed["closed_since"] == []
    assert unnamed["closed_total"] == since_one["closed_total"] == 7
    assert [(turn["turn_index"], turn["reply"]) for turn in paged] == list(
        zip(range(1, 8), replies, strict=True)
    )
    assert reads == 3


@pytest.mark.parametrize("route", ["state", "resume"])
async def test_an_unknown_conversation_is_not_found(pg_engine: AsyncEngine, route: str) -> None:
    app, _ = orchestrator_app(engine=pg_engine)
    async with studio_client(app) as client:
        if route == "state":
            response = await _state(client)
        else:
            response = await client.post("/api/studio/resume", json=_resume_body(1), headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


@pytest.mark.parametrize(
    ("script", "turn_index"),
    [
        pytest.param((_text("Done."),), 1, id="closed-turn"),
        pytest.param((_text("Done."),), 2, id="no-such-turn"),
        pytest.param((plan_call("q with forks"),), 1, id="paused-turn"),
    ],
)
async def test_a_resume_of_a_turn_that_is_not_open_and_running_is_refused(
    pg_engine: AsyncEngine, script: tuple[types.Part, ...], turn_index: int
) -> None:
    app, _ = orchestrator_app(*script, engine=pg_engine)
    async with studio_client(app) as client:
        await stream(client, "/api/studio/turn", turn_body("q with forks"))
        refused = await client.post(
            "/api/studio/resume", json=_resume_body(turn_index), headers=AUTH
        )
        read = (await _state(client)).json()

    assert refused.status_code == 409
    assert refused.json()["error"] == "edit_conflict"
    assert refused.json()["pointer"] == "/turn_index"
    assert read["turn"]["lease_live"] is False


async def test_a_resume_sends_the_approved_plan_its_departed_client_never_got(
    pg_engine: AsyncEngine,
) -> None:
    """The client approves the plan and leaves once the turn accepted the
    answer, before the `approve` node took it, so no `plan.approved` went
    out. `/resume` runs that node and sends the approved plan once."""
    app, _ = orchestrator_app(
        plan_call("q with forks"),
        _text("Planned and run."),
        tool_surface=ScriptedSurface(),
        engine=pg_engine,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("q with forks"))
    plan = first[-1]
    assert plan["event"] == "plan.proposed"

    turns = app.state.studio.turns
    accepted = asyncio.Event()
    record_answer = turns.record_answer

    async def accept_then_hang(*args: Any, **kwargs: Any) -> Any:
        cursor = await record_answer(*args, **kwargs)
        accepted.set()
        await asyncio.Event().wait()
        return cursor

    turns.record_answer = accept_then_hang
    left = await _leave_once(
        app,
        "/api/studio/interrupt",
        {
            "kind": "interrupt.response",
            "schema_version": 1,
            "conversation_id": CONVERSATION,
            "interrupt_id": plan["interrupt_id"],
            "decision": "approve",
        },
        accepted,
    )
    turns.record_answer = record_answer
    assert "plan.approved" not in _events(left)

    async with studio_client(app) as client:
        resumed = await stream(client, "/api/studio/resume", _resume_body(1))

    assert _events(resumed).count("plan.approved") == 1
    assert next(f for f in resumed if f["event"] == "plan.approved")["revision"] == plan["revision"]
    assert _events(resumed)[-1] == "done"


async def test_a_resume_carries_on_an_answer_its_client_left_before_the_turn_took_it(
    pg_engine: AsyncEngine,
) -> None:
    """The client approves and leaves after `/interrupt` recorded the answer
    and before the turn accepted it, so the cursor still stands paused with
    its only entry resuming. The state read shows the turn running with no
    lease and no pause to answer, and `/resume` carries the answer on as a
    retry of it would."""
    app, _ = orchestrator_app(
        plan_call("q with forks"),
        _text("Planned and run."),
        tool_surface=ScriptedSurface(),
        engine=pg_engine,
    )
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("q with forks"))
    plan = first[-1]
    assert plan["event"] == "plan.proposed"

    turns = app.state.studio.turns
    reached = asyncio.Event()
    record_answer = turns.record_answer

    async def hang_before_accepting(*args: Any, **kwargs: Any) -> Any:
        reached.set()
        await asyncio.Event().wait()
        return await record_answer(*args, **kwargs)

    turns.record_answer = hang_before_accepting
    await _leave_once(
        app,
        "/api/studio/interrupt",
        {
            "kind": "interrupt.response",
            "schema_version": 1,
            "conversation_id": CONVERSATION,
            "interrupt_id": plan["interrupt_id"],
            "decision": "approve",
        },
        reached,
    )
    turns.record_answer = record_answer

    async with studio_client(app) as client:
        read = (await _state(client)).json()
        resumed = await stream(client, "/api/studio/resume", _resume_body(1))

    assert read["turn"]["status"] == "running"
    assert read["turn"]["lease_live"] is False
    assert "pause" not in read
    assert _events(resumed).count("plan.approved") == 1
    assert _events(resumed)[-1] == "done"


def _entry(interrupt_id: str, *, answered: bool = False) -> _PendingInterrupt:
    return _PendingInterrupt(
        adk_id=interrupt_id,
        kind="plan",
        resuming_segment="seg_left" if answered else None,
        answer={"decision": "approve", "payload": None} if answered else None,
    )


# Each state the read can report, and what the page then does: `resume` for
# `running` with no live lease, `wait` for a live lease, `answer` the pause,
# or `append` the closed turn.
_READ_TABLE = [
    pytest.param(TurnStatus.RUNNING, None, (), False, None, "running", "resume", id="left-running"),
    pytest.param(TurnStatus.RUNNING, None, (), False, 30.0, "running", "wait", id="segment-live"),
    pytest.param(
        TurnStatus.PAUSED,
        "int_1",
        (_entry("int_1"),),
        False,
        None,
        "paused",
        "answer",
        id="paused-awaiting",
    ),
    pytest.param(
        TurnStatus.PAUSED,
        "int_1",
        (_entry("int_1", answered=True),),
        False,
        None,
        "running",
        "resume",
        id="paused-answer-left",
    ),
    pytest.param(
        TurnStatus.PAUSED,
        "int_1",
        (),
        False,
        None,
        "running",
        "resume",
        id="paused-entry-never-stored",
    ),
    pytest.param(
        TurnStatus.PAUSED,
        "int_2",
        (_entry("int_1", answered=True), _entry("int_2")),
        False,
        None,
        "paused",
        "answer",
        id="paused-again-after-an-answer-left",
    ),
    pytest.param(
        TurnStatus.PAUSED,
        "int_2",
        (_entry("int_1", answered=True),),
        False,
        None,
        "running",
        "resume",
        id="paused-again-entry-never-stored",
    ),
    pytest.param(
        TurnStatus.DONE, None, (), False, 30.0, "running", "wait", id="closed-record-unwritten"
    ),
    pytest.param(TurnStatus.DONE, None, (), True, None, "done", "append", id="closed-recorded"),
    pytest.param(None, None, (), False, 30.0, "running", "wait", id="edit-card-live"),
    pytest.param(None, None, (), False, None, "abandoned", "append", id="edit-card-left"),
]


@pytest.mark.parametrize(
    ("cursor", "pause_id", "pending", "recorded", "lease_seconds", "status", "action"),
    _READ_TABLE,
)
def test_every_state_the_read_reports_has_one_page_action(
    cursor: TurnStatus | None,
    pause_id: str | None,
    pending: tuple[_PendingInterrupt, ...],
    recorded: bool,
    lease_seconds: float | None,
    status: str,
    action: str,
) -> None:
    """The contract's table of page actions, over every state the read
    reports: a `running` turn with no live lease is resumed, a live lease is
    waited out, a `paused` turn carries the pause to answer, and a closed
    turn is in `closed_since`."""
    state = ConversationState(conversation_id=CONVERSATION, turn_index=1, trace_id="tr_1")
    if recorded:
        state.close_turn(final_text="Done.")
    elif cursor is None and lease_seconds is None:
        state.turn_recorded = False
    state.pending = {entry.adk_id: entry for entry in pending}
    stored = StoredConversation(
        state=state, turn_status=cursor, lease_seconds=lease_seconds, pause_id=pause_id
    )

    read = _conversation_read(stored, since=0)

    assert read.turn is not None
    assert read.turn.status.value == status
    match action:
        case "resume":
            assert not read.turn.lease_live and read.pause is None
        case "wait":
            assert read.turn.lease_live and read.turn.retry_after_seconds
        case "answer":
            assert read.pause is not None and read.pause.interrupt_id == pause_id
        case "append":
            assert not read.turn.lease_live
            assert [turn.turn_index for turn in read.closed_since] == ([1] if recorded else [])


async def test_a_resume_sends_a_pause_whose_entry_was_never_stored(pg_engine: AsyncEngine) -> None:
    """The turn pauses on its plan card and the closing write that stores the
    card's entry saves nothing. Once the lease lapses, the read reports the
    turn running with no pause, and `/resume` sends the pause again from the
    cursor, after which the read shows it to answer."""
    clock = await StoreClock.start(pg_engine)
    store = StudioStore(pg_engine, **clock.store_kwargs)
    app, _ = orchestrator_app(plan_call("q with forks"), store=store)
    close_segment = store.close_segment
    failed = False

    async def fail_once(*args: Any, **kwargs: Any) -> SegmentClose:
        nonlocal failed
        if not failed:
            failed = True
            return SegmentClose.FAILED
        return await close_segment(*args, **kwargs)

    store.close_segment = fail_once  # type: ignore[method-assign]
    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", turn_body("q with forks"))
        assert first[-1]["event"] == "error"
        await clock.advance(SEGMENT_TTL_S + 1)
        before = (await _state(client)).json()
        resumed = await stream(client, "/api/studio/resume", _resume_body(1))
        after = (await _state(client)).json()

    assert before["turn"]["status"] == "running"
    assert before["turn"]["lease_live"] is False
    assert "pause" not in before
    assert _events(resumed)[-1] == "plan.proposed"
    assert after["turn"]["status"] == "paused"
    assert after["pause"]["interrupt_id"] == resumed[-1]["interrupt_id"]


async def test_a_closed_turn_carries_the_cards_it_made(pg_engine: AsyncEngine) -> None:
    """A page that never got a turn's frames draws its cards from the state
    read: each card the turn made comes with its stored type, title, revision
    and fact summary."""
    app, _ = orchestrator_app(
        search_call("life expectancy in Kenya"),
        fetch_call(),
        make_charts_call(chart(frame_for(), title="Life expectancy in Kenya")),
        _text("Here is the card."),
        resolver=CannedSearch(definite_envelope()),
        engine=pg_engine,
    )
    async with studio_client(app) as client:
        frames = await stream(client, "/api/studio/turn", turn_body("life expectancy in Kenya"))
        read = (await _state(client, since=0)).json()

    created = next(frame for frame in frames if frame["event"] == "component.created")
    (block,) = frames[-1]["composition"]["blocks"]
    (closed,) = read["closed_since"]
    assert closed["component_ids"] == [created["component_id"]]
    (card,) = closed["cards"]
    assert card == {
        "component_id": created["component_id"],
        "type": created["type"],
        "title": block["title"],
        "revision": created["revision"],
        "fact_summary": created["fact_summary"],
    }


@pytest.mark.parametrize(
    ("record", "status"),
    [
        pytest.param({"error": None}, TurnStatus.DONE, id="clean"),
        pytest.param({"error": "cancelled"}, TurnStatus.CANCELLED, id="stopped"),
        pytest.param({"error": "abandoned"}, TurnStatus.ABANDONED, id="abandoned"),
        pytest.param({"error": "state_not_saved"}, TurnStatus.FAILED, id="unsaved"),
        pytest.param({"error": "capability_unavailable"}, TurnStatus.FAILED, id="failed"),
    ],
)
def test_a_closed_turn_reads_the_status_its_record_states(
    record: dict[str, Any], status: TurnStatus
) -> None:
    assert record_status(record) is status


# The pause's entry, by how it stands against the cursor's pause `int_1`.
_ENTRIES: dict[str, tuple[_PendingInterrupt, ...]] = {
    "no-entry": (),
    "awaiting": (_entry("int_1"),),
    "answer-left": (_entry("int_1", answered=True),),
    "other-pause": (_entry("int_0", answered=True),),
}

_CLOSED_DONE = TurnStanding(Standing.CLOSED, TurnStatus.DONE)

# Where an unrecorded turn with no live lease stands, by its cursor and the
# pause's entry. A recorded turn with no live lease stands closed as its
# record states, and a live lease holds any turn.
_UNRECORDED_UNHELD: dict[tuple[TurnStatus | None, str], TurnStanding] = {
    **{(None, entry): TurnStanding(Standing.CLOSED, TurnStatus.ABANDONED) for entry in _ENTRIES},
    **{(TurnStatus.RUNNING, entry): TurnStanding(Standing.RESUMABLE) for entry in _ENTRIES},
    (TurnStatus.PAUSED, "no-entry"): TurnStanding(Standing.REPLAY),
    (TurnStatus.PAUSED, "awaiting"): TurnStanding(Standing.AWAITING_ANSWER),
    (TurnStatus.PAUSED, "answer-left"): TurnStanding(Standing.CARRY),
    (TurnStatus.PAUSED, "other-pause"): TurnStanding(Standing.REPLAY),
    **{(TurnStatus.DONE, entry): _CLOSED_DONE for entry in _ENTRIES},
    **{
        (TurnStatus.FAILED, entry): TurnStanding(Standing.CLOSED, TurnStatus.FAILED)
        for entry in _ENTRIES
    },
    **{
        (TurnStatus.CANCELLED, entry): TurnStanding(Standing.CLOSED, TurnStatus.CANCELLED)
        for entry in _ENTRIES
    },
    **{
        (TurnStatus.ABANDONED, entry): TurnStanding(Standing.CLOSED, TurnStatus.ABANDONED)
        for entry in _ENTRIES
    },
}


def _standing_cells() -> list[Any]:
    cells = []
    for cursor, recorded, lease, entry in itertools.product(
        [None, *TurnStatus], [False, True], [False, True], _ENTRIES
    ):
        if lease:
            expected = TurnStanding(Standing.HELD)
        elif recorded:
            # The record states the turn failed, whatever the cursor says.
            expected = TurnStanding(Standing.CLOSED, TurnStatus.FAILED)
        else:
            expected = _UNRECORDED_UNHELD[cursor, entry]
        label = cursor.value if cursor is not None else "no-cursor"
        cells.append(
            pytest.param(
                cursor,
                recorded,
                lease,
                entry,
                expected,
                id=f"{label}-{'recorded' if recorded else 'open'}-"
                f"{'lease' if lease else 'unheld'}-{entry}",
            )
        )
    return cells


_STANDING_CELLS = _standing_cells()


def test_the_standing_table_covers_every_cell() -> None:
    assert len(_STANDING_CELLS) == 7 * 2 * 2 * 4


@pytest.mark.parametrize(("cursor", "recorded", "lease", "entry", "expected"), _STANDING_CELLS)
def test_every_turn_the_store_can_hold_has_one_standing(
    cursor: TurnStatus | None,
    recorded: bool,
    lease: bool,
    entry: str,
    expected: TurnStanding,
) -> None:
    """`turn_standing` over the product of the cursor's status, whether the
    turn is recorded, whether a lease is live, and the pause's entry. The
    state read and `/resume` both map the standing, so the read's status and
    the resume's answer agree on every cell: `running` is resumed or waited
    out, and `paused` or a closed status refuses the resume."""
    state = ConversationState(conversation_id=CONVERSATION, turn_index=1, trace_id="tr_1")
    if recorded:
        state.close_turn(final_text="", error="capability_unavailable")
    state.pending = {pending.adk_id: pending for pending in _ENTRIES[entry]}

    standing = turn_standing(
        cursor_status=cursor,
        pause_id="int_1" if cursor is TurnStatus.PAUSED else None,
        lease_live=lease,
        state=state,
    )

    assert standing == expected
    refused = _resume_refusal(standing, 1) is not None
    assert refused == (
        _read_status(standing).value != "running" or standing.standing is Standing.CARRY
    )


class _BrokenSurface(ScriptedSurface):
    """A tool surface whose every call fails, as a lost analysis mount does."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        raise RuntimeError("analysis mount went away")


def _approve(interrupt_id: str) -> dict[str, Any]:
    return {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "interrupt_id": interrupt_id,
        "decision": "approve",
    }


async def _nothing() -> None:
    return None


type _Setup = Callable[[AsyncEngine], Awaitable[tuple[FastAPI, Callable[[], Awaitable[None]]]]]


async def _closed_done(engine: AsyncEngine) -> tuple[FastAPI, Callable[[], Awaitable[None]]]:
    app, _ = orchestrator_app(_text("Done."), engine=engine)
    async with studio_client(app) as client:
        await stream(client, "/api/studio/turn", turn_body("hi"))
    return app, _nothing


async def _recorded_over_an_open_cursor(
    engine: AsyncEngine,
) -> tuple[FastAPI, Callable[[], Awaitable[None]]]:
    """A row stored before the closing write closed the cursor with the
    record: the turn recorded `failed`, its cursor left running."""
    app, _ = orchestrator_app(
        plan_call("q with forks"), tool_surface=_BrokenSurface(), engine=engine
    )
    async with studio_client(app) as client:
        plan = (await stream(client, "/api/studio/turn", turn_body("q with forks")))[-1]
        failed = await stream(client, "/api/studio/interrupt", _approve(plan["interrupt_id"]))
    assert failed[-1]["event"] == "error"
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE turn_cursors SET status = 'running', error = NULL"))
    return app, _nothing


async def _held(engine: AsyncEngine) -> tuple[FastAPI, Callable[[], Awaitable[None]]]:
    searching = _HangOnce()
    app, _ = orchestrator_app(search_call("aid to Kenya"), resolver=searching, engine=engine)
    leave = asyncio.Event()
    posting = asyncio.ensure_future(
        _post_then_leave(app, "/api/studio/turn", turn_body("aid to Kenya"), leave)
    )
    await asyncio.wait_for(searching.entered.wait(), timeout=_DEADLINE_S)

    async def release() -> None:
        leave.set()
        await posting

    return app, release


async def _left_running(engine: AsyncEngine) -> tuple[FastAPI, Callable[[], Awaitable[None]]]:
    searching = _HangOnce()
    app, _ = orchestrator_app(
        search_call("aid to Kenya"), plan_call("q with forks"), resolver=searching, engine=engine
    )
    await _leave_once(app, "/api/studio/turn", turn_body("q with forks"), searching.entered)
    return app, _nothing


async def _awaiting_answer(engine: AsyncEngine) -> tuple[FastAPI, Callable[[], Awaitable[None]]]:
    app, _ = orchestrator_app(plan_call("q with forks"), engine=engine)
    async with studio_client(app) as client:
        await stream(client, "/api/studio/turn", turn_body("q with forks"))
    return app, _nothing


async def _answer_left(engine: AsyncEngine) -> tuple[FastAPI, Callable[[], Awaitable[None]]]:
    app, _ = orchestrator_app(
        plan_call("q with forks"),
        _text("Planned and run."),
        tool_surface=ScriptedSurface(),
        engine=engine,
    )
    async with studio_client(app) as client:
        plan = (await stream(client, "/api/studio/turn", turn_body("q with forks")))[-1]
    turns = app.state.studio.turns
    reached = asyncio.Event()
    record_answer = turns.record_answer

    async def hang_before_accepting(*args: Any, **kwargs: Any) -> Any:
        reached.set()
        await asyncio.Event().wait()
        return await record_answer(*args, **kwargs)

    turns.record_answer = hang_before_accepting
    await _leave_once(app, "/api/studio/interrupt", _approve(plan["interrupt_id"]), reached)
    turns.record_answer = record_answer
    return app, _nothing


async def _pause_never_stored(engine: AsyncEngine) -> tuple[FastAPI, Callable[[], Awaitable[None]]]:
    clock = await StoreClock.start(engine)
    store = StudioStore(engine, **clock.store_kwargs)
    app, _ = orchestrator_app(plan_call("q with forks"), store=store)
    close_segment = store.close_segment
    failed = False

    async def fail_once(*args: Any, **kwargs: Any) -> SegmentClose:
        nonlocal failed
        if not failed:
            failed = True
            return SegmentClose.FAILED
        return await close_segment(*args, **kwargs)

    store.close_segment = fail_once  # type: ignore[method-assign]
    async with studio_client(app) as client:
        await stream(client, "/api/studio/turn", turn_body("q with forks"))
    await clock.advance(SEGMENT_TTL_S + 1)
    return app, _nothing


@pytest.mark.parametrize(
    ("setup", "standing", "status", "lease_live", "resumed"),
    [
        pytest.param(_closed_done, Standing.CLOSED, "done", False, "edit_conflict", id="closed"),
        pytest.param(
            _recorded_over_an_open_cursor,
            Standing.CLOSED,
            "failed",
            False,
            "edit_conflict",
            id="closed-by-record-over-an-open-cursor",
        ),
        pytest.param(_held, Standing.HELD, "running", True, "transient_unavailable", id="held"),
        pytest.param(
            _left_running, Standing.RESUMABLE, "running", False, "plan.proposed", id="resumable"
        ),
        pytest.param(
            _awaiting_answer,
            Standing.AWAITING_ANSWER,
            "paused",
            False,
            "edit_conflict",
            id="awaiting-answer",
        ),
        pytest.param(_answer_left, Standing.CARRY, "running", False, "done", id="carry"),
        pytest.param(
            _pause_never_stored, Standing.REPLAY, "running", False, "plan.proposed", id="replay"
        ),
    ],
)
async def test_the_state_read_and_resume_agree_on_each_standing(
    pg_engine: AsyncEngine,
    setup: _Setup,
    standing: Standing,
    status: str,
    lease_live: bool,
    resumed: str,
) -> None:
    """Through the real store and routes, a turn of each standing reads the
    status the contract's table gives it, and `/resume` answers what that
    status tells the page to expect: a stream for `running` with no live
    lease, a retry for a live lease, and `edit_conflict` for `paused` or a
    closed turn. A turn recorded with its cursor left open reads closed as
    its record states, and the resume's load closes the cursor so."""
    app, release = await setup(pg_engine)
    try:
        stored = await app.state.studio.store.peek(PRINCIPAL, CONVERSATION)
        assert stored is not None
        found = turn_standing(
            cursor_status=stored.turn_status,
            pause_id=stored.pause_id,
            lease_live=stored.lease_seconds is not None,
            state=stored.state,
        )
        async with studio_client(app) as client:
            read = (await _state(client)).json()
            response = await client.post("/api/studio/resume", json=_resume_body(1), headers=AUTH)
    finally:
        await release()

    assert found.standing is standing
    assert read["turn"]["status"] == status
    assert read["turn"]["lease_live"] is lease_live
    if response.status_code == 200:
        assert _events(_frames([response.content]))[-1] == resumed
    else:
        assert response.status_code == 409
        assert response.json()["error"] == resumed
    if standing is Standing.CLOSED:
        assert await TurnStore(pg_engine).open_cursor(PRINCIPAL, CONVERSATION) is None


async def test_a_closing_write_that_records_a_failed_turn_closes_its_cursor(
    pg_engine: AsyncEngine,
) -> None:
    """A segment records its turn failed with the turn's cursor still open,
    as a node whose own close could not be written leaves it. The closing
    write that commits the record closes the cursor `failed` in the same
    transaction, with the record's code as its error."""
    app, _ = orchestrator_app(plan_call("q with forks"), engine=pg_engine)
    async with studio_client(app) as client:
        await stream(client, "/api/studio/turn", turn_body("q with forks"))
    store = StudioStore(pg_engine)
    conversation = await store.load(PRINCIPAL, CONVERSATION)
    assert conversation is not None
    segment_id = store.claim_segment(PRINCIPAL, CONVERSATION)
    await store.save(PRINCIPAL, CONVERSATION)
    conversation.close_turn(final_text="", error="capability_unavailable")

    closed = await store.close_segment(PRINCIPAL, CONVERSATION, segment_id)

    assert closed is SegmentClose.SAVED
    async with pg_engine.connect() as conn:
        rows = (await conn.execute(text("SELECT status, error FROM turn_cursors"))).all()
    assert [tuple(row) for row in rows] == [("failed", {"code": "capability_unavailable"})]


async def test_every_stream_names_the_turn_it_carries(pg_engine: AsyncEngine) -> None:
    """`/turn`, `/interrupt` and `/resume` each answer with the index of the
    turn the stream carries in the turn index header, the `turn.index` a
    state read gives that turn."""
    searching = _HangOnce()
    app, _ = orchestrator_app(
        _text("One."),
        plan_call("q with forks"),
        _text("Planned and run."),
        search_call("aid to Kenya"),
        plan_call("q with forks"),
        resolver=searching,
        tool_surface=ScriptedSurface(),
        engine=pg_engine,
    )

    async def posted(client: httpx2.AsyncClient, path: str, body: Mapping[str, Any]) -> Any:
        response = await client.post(path, json=body, headers=AUTH)
        assert response.status_code == 200
        return response.headers[TURN_INDEX_HEADER], _frames([response.content])

    async with studio_client(app) as client:
        first, _ = await posted(client, "/api/studio/turn", turn_body("first"))
        second, plan = await posted(client, "/api/studio/turn", turn_body("q with forks"))
        answered, _ = await posted(
            client, "/api/studio/interrupt", _approve(plan[-1]["interrupt_id"])
        )
    await _leave_once(app, "/api/studio/turn", turn_body("q with forks"), searching.entered)
    async with studio_client(app) as client:
        resumed, frames = await posted(client, "/api/studio/resume", _resume_body(3))
        read = (await _state(client)).json()

    assert (first, second, answered, resumed) == ("1", "2", "2", "3")
    assert _events(frames)[-1] == "plan.proposed"
    assert read["turn"]["index"] == 3
