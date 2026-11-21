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
import json
from collections.abc import Mapping
from typing import Any

import httpx2
import pytest
from compiler.surface import ProgressCallback, ToolCallResult
from fastapi import FastAPI
from google.genai import types
from harness.scripted import ScriptedSurface
from harness.studio_store import StudioStore
from planner.workflow import ToolId
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_tools import plan_call, search_call
from .studio_routes import AUTH, orchestrator_app, stream, studio_client, turn_body
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
    oldest first, at most three."""
    app, _ = orchestrator_app(
        _text("One."), _text("Two."), _text("Three."), _text("Four."), engine=pg_engine
    )
    async with studio_client(app) as client:
        for message in ("first", "second", "third", "fourth"):
            await stream(client, "/api/studio/turn", turn_body(message))
        since_one = (await _state(client, since=1)).json()
        since_three = (await _state(client, since=3)).json()
        unnamed = (await _state(client)).json()

    assert [
        (turn["turn_index"], turn["user_message"], turn["reply"])
        for turn in since_one["closed_since"]
    ] == [(2, "second", "Two."), (3, "third", "Three."), (4, "fourth", "Four.")]
    assert [turn["turn_index"] for turn in since_three["closed_since"]] == [4]
    assert since_one["turn"]["status"] == "done"
    assert since_one["turn"]["index"] == 4
    assert unnamed["closed_since"] == []

    async with studio_client(app) as client:
        capped = (await _state(client, since=0)).json()
    assert [turn["turn_index"] for turn in capped["closed_since"]] == [1, 2, 3]


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
