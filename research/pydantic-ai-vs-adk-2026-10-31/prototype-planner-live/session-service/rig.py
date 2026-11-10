"""One Studio turn on either planner over a given ADK session service.

`build` wires the routes as `harness.server` wires them: a `StudioStore` over the session
service (the conversation's ledger session lives beside the per-turn sessions), a
`StudioRunLedger` over the store, and the real root workflow. The orchestrator's model is
scripted to a planner-entry task's recorded calls and the planner's model is a scripted
google-genai client, as in `node/studio_arms.py`, which this reuses for the task recording and
the translation stand-in.

`run_actions` posts a list of requests (a turn, an answer, an amendment, an approval) and
returns what each answered. It runs in this process or, through `segment_child.py`, in a
process of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg
import scripted as sc
import validate
from entry import (
    RecordedSurface,
    TranslationCalls,
    _no_anchor_read,
    load_recording,
)
from google.adk.sessions import BaseSessionService
from harness.graph.state import AGENT_NAME as GRAPH_NAME
from harness.planner_seam import PlannerSeam
from harness.run_ledger import StudioRunLedger
from harness.settings import Settings
from harness.studio_state import StudioStore
from harness.turn_events import PLANNER_ASK
from pai_node import pai_planner
from scenes import INTERRUPT, TURN, interrupt_body
from sqlalchemy import event as sa_event
from streaming import AUTH
from studio_arms import TASKS, Arm, StudioResolver, orchestrator_script
from submission.service import COMBINATION_OPTIONS, COMBINE_OPTION
from tests.orchestrator_fakes import ScriptLlm, orchestrator_app, studio_client, turn_body
from translator.tool import build_translate

CONVERSATION = "conv_01test0000"
PLAIN = TASKS["aggregate-average-oda-kenya"]
WITH_ASK = TASKS["sum-net-flows-aid-debt-ghana"]

Call = tuple[str, dict[str, Any]]


def propose(task: Any) -> Call:
    return ("propose_plan", dict(task.reference_payload))


def ask(**arguments: Any) -> Call:
    return (PLANNER_ASK, arguments)


GET_SKILL: Call = ("get_skill", {"skill_id": "oda-dac2a"})
PATCH: Call = ("propose_plan", {"amends_version": 1})
ANSWERS = {
    "combine": {"q1": {"kind": "option", "option_id": COMBINE_OPTION}},
    "text": {"q1": {"kind": "text", "text": "Since 2020"}},
}
# Each script is a task and the planner model's answers, one list of calls per request.
SCRIPTS: dict[str, tuple[Any, list[list[Call]]]] = {
    # a skill read, an ask, its answer, a proposal
    "skill-ask": (
        PLAIN,
        [[GET_SKILL], [ask(message="Which years do you mean?")], [propose(PLAIN)]],
    ),
    # an ask about combining, its answer, a proposal that needs the answer
    "combine-ask": (
        WITH_ASK,
        [
            [propose(WITH_ASK)],
            [ask(message="Do these add up?", options=[dict(o) for o in COMBINATION_OPTIONS])],
            [propose(WITH_ASK)],
        ],
    ),
    # four responses in one run: three skill reads, then a proposal
    "four-responses": (PLAIN, [[GET_SKILL], [GET_SKILL], [GET_SKILL], [propose(PLAIN)]]),
    # a skill read, then a proposal
    "skill-propose": (PLAIN, [[GET_SKILL], [propose(PLAIN)]]),
    # turn one asks and is answered, then turn two proposes and is amended
    "two-turns": (
        PLAIN,
        [
            [ask(message="Which years do you mean?")],
            [propose(PLAIN)],
            [propose(PLAIN)],
            [PATCH],
        ],
    ),
}


@dataclass
class WriteLog:
    """The INSERT, UPDATE and DELETE statements a service's engine ran, by table and by which
    ADK session they address (`turn` or `ledger`), with whether the planner was running."""

    writes: list[tuple[str, str, str, bool]] = field(default_factory=list)
    in_planner: bool = False

    def attach(self, service: Any) -> None:
        sa_event.listen(service.db_engine.sync_engine, "before_cursor_execute", self._record)

    def _record(self, conn: Any, cursor: Any, statement: str, parameters: Any, *_: Any) -> None:
        found = re.match(r"\s*(INSERT INTO|UPDATE|DELETE FROM)\s+(\w+)", statement)
        if not found:
            return
        values = list(parameters.values() if isinstance(parameters, Mapping) else parameters)
        if GRAPH_NAME in values:
            scope = "turn"
        elif f"{GRAPH_NAME}.ledger" in values:
            scope = "ledger"
        else:
            scope = "other"
        self.writes.append((found[1].split()[0], found[2], scope, self.in_planner))

    def count(self, *, table: str, scope: str, planner: bool | None = None) -> int:
        return sum(
            1
            for _verb, t, s, p in self.writes
            if t == table and s == scope and (planner is None or p == planner)
        )


@contextlib.contextmanager
def planner_window(log: WriteLog) -> Iterator[None]:
    """Mark the writes made while `PlannerSeam.__call__` runs, on either arm (the Pydantic AI
    seam calls it as well)."""
    original = PlannerSeam.__call__

    async def bracketed(self: PlannerSeam, *args: Any, **kwargs: Any) -> Any:
        log.in_planner = True
        try:
            return await original(self, *args, **kwargs)
        finally:
            log.in_planner = False

    PlannerSeam.__call__ = bracketed  # type: ignore[method-assign]
    try:
        yield
    finally:
        PlannerSeam.__call__ = original  # type: ignore[method-assign]


@dataclass
class Rig:
    app: Any
    orchestrator: ScriptLlm
    client: sc.Scripted
    store: StudioStore
    service: BaseSessionService
    translations: list[int]


async def build(
    arm: Arm,
    task: Any,
    steps: Sequence[Sequence[Call]],
    *,
    service: BaseSessionService,
    clock: Callable[[], float] = time.time,
    pai: dict[str, Any] | None = None,
    block: Path | None = None,
    turns: int = 1,
) -> Rig:
    """The Studio app for `task` on `arm`, its sessions and ledger on `service`. With `block`,
    the first translation inside `propose_plan` writes the file and never returns."""
    recording = load_recording(task.id)
    surface = RecordedSurface(recording.get("tools") or {})
    translation = TranslationCalls(
        recording.get("translation") or {}, schemas=await validate.tool_schemas()
    )
    inner = build_translate(
        surface=translation,
        resolve_entities=translation.resolve_entities,
        anchor_read=_no_anchor_read,
        describe_nodes=translation.describe_nodes,
    )
    started: list[int] = []

    async def translate(planner_call: Any) -> Any:
        started.append(1)
        if block is not None:
            block.write_text(str(os.getpid()))
            await asyncio.Event().wait()
        return await inner(planner_call)

    client = sc.scripted([list(step) for step in steps])
    seam = PlannerSeam(settings=Settings(), surface=surface, translate=translate)
    store = StudioStore(service, app_name=GRAPH_NAME, clock=clock)
    planner: Any = seam
    if arm == "adk":
        seam._agent.model.client = client.client
    else:
        planner = pai_planner(
            seam, client=client.client, surface=surface, translate=translate, **(pai or {})
        )
    app, model = orchestrator_app(
        *orchestrator_script(task) * turns,
        resolver=StudioResolver(recording["searches"]),
        planner=planner,
        session_service=service,
        store=store,
        run_ledger=StudioRunLedger(store),
    )
    return Rig(app, model, client, store, service, started)


async def post(client: Any, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """POST `body`; the status, and the frames of an event stream or the body of an error."""
    async with client.stream("POST", path, json=body, headers=AUTH) as response:
        status = response.status_code
        if not response.headers["content-type"].startswith("text/event-stream"):
            return {"status": status, "frames": [], "error": json.loads(await response.aread())}
        frames = [
            json.loads(line.removeprefix("data: "))
            async for line in response.aiter_lines()
            if line.startswith("data: ")
        ]
    return {"status": status, "frames": frames, "error": None}


def last_frame(frames: Sequence[Mapping[str, Any]], event: str) -> dict[str, Any]:
    (found,) = [f for f in frames if f["event"] == event][-1:]
    return dict(found)


async def run_actions(
    rig: Rig, task: Any, actions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Post each action and return what it answered. An action is `{"kind": "turn"}` (with an
    optional `message`), `{"kind": "answer", "answers": name}`, `{"kind": "amend", "words":
    text}` or `{"kind": "approve"}`. An action with no `interrupt_id` answers the pause the
    action before it ended on."""
    results: list[dict[str, Any]] = []
    async with studio_client(rig.app) as client:
        for action in actions:
            kind = action["kind"]
            if kind == "turn":
                body = turn_body(action.get("message") or task.question)
                results.append(await post(client, TURN, body))
                continue
            interrupt_id = action.get("interrupt_id")
            if interrupt_id is None:
                previous = results[-1]["frames"]
                pause = last_frame(
                    previous, "elicitation.request" if kind == "answer" else "plan.proposed"
                )
                interrupt_id = pause["interrupt_id"]
            if kind == "answer":
                body = interrupt_body(interrupt_id, "answer", answers=ANSWERS[action["answers"]])
            elif kind == "amend":
                body = interrupt_body(interrupt_id, "amend", payload=action["words"])
            else:
                body = interrupt_body(interrupt_id, "approve")
            results.append(await post(client, INTERRUPT, body))
    return results


async def inspect(url: str) -> dict[str, Any]:
    """What the database holds, read with SQL of its own (not through ADK): per session, the
    event rows by author and the progress reports among them, the event bytes (as JSON text, and as stored, which is the compressed
    size when Postgres compressed the value), the state bytes and its keys."""
    dsn = url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        by_author = await conn.fetch(
            "select session_id, event_data->>'author' as author, count(*) as n "
            "from events group by 1, 2 order by 1, 2"
        )
        progress = await conn.fetch(
            "select session_id, count(*) as n from events "
            "where event_data->'custom_metadata'->'progress' is not null group by 1"
        )
        event_bytes = await conn.fetch(
            "select app_name, session_id, count(*) as n, "
            "sum(octet_length(event_data::text)) as json_bytes, "
            "sum(pg_column_size(event_data)) as stored_bytes "
            "from events group by 1, 2 order by 1, 2"
        )
        states = await conn.fetch(
            "select app_name, id, octet_length(state::text) as json_bytes, "
            "pg_column_size(state) as stored_bytes, state::text as text "
            "from sessions order by 1, 2"
        )
    finally:
        await conn.close()
    ledger = next((r for r in states if r["id"] == CONVERSATION), None)
    return {
        "ledger": json.loads(ledger["text"]) if ledger else None,
        "authors": {f"{r['session_id']}/{r['author']}": r["n"] for r in by_author},
        "progress": {r["session_id"]: r["n"] for r in progress},
        "events": {
            f"{r['app_name']}/{r['session_id']}": {
                "rows": r["n"],
                "json_bytes": r["json_bytes"],
                "stored_bytes": r["stored_bytes"],
            }
            for r in event_bytes
        },
        "states": {
            f"{r['app_name']}/{r['id']}": {
                "json_bytes": r["json_bytes"],
                "stored_bytes": r["stored_bytes"],
                "keys": sorted(json.loads(r["text"])),
            }
            for r in states
        },
    }


async def planner_trace(url: str, session_id: str) -> list[list[str]]:
    """The events of one ADK session in the order they were stored, as `[author, kind, name]`
    for each function call and function response in them (`kind` is `call` or `response`)."""
    conn = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        rows = await conn.fetch(
            "select event_data::text as text from events where session_id = $1 "
            "order by timestamp, id",
            session_id,
        )
    finally:
        await conn.close()
    trace: list[list[str]] = []
    for row in rows:
        data = json.loads(row["text"])
        for part in (data.get("content") or {}).get("parts") or []:
            for kind, key in (("call", "function_call"), ("response", "function_response")):
                if part.get(key):
                    trace.append([data["author"], kind, part[key]["name"]])
    return trace
