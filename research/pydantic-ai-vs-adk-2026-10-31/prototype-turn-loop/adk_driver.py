"""One request against the ADK workflow, as a child process: one JSON line per
outcome. Commands as in `own_driver.py`.

With `--guards` the driver also does what dev's Studio routes do around ADK,
on a per-conversation ledger session in the same `DatabaseSessionService`
(`harness.studio_state`): it records each pause the stream raises as pending,
claims an answer by removing it under ADK's stale-session compare-and-swap
(the `/interrupt` 409 and `plan_revision_stale`), and, while a turn streams,
polls the ledger's cancel flag every `--heartbeat-s` seconds (dev's
`HEARTBEAT_INTERVAL_S`, 5) and closes the stream when it is set.
"""

import argparse
import asyncio
import contextlib
import json
import sys
import time
from contextlib import aclosing
from pathlib import Path
from typing import Any

from adk_flow import APP_NAME, build_runner
from google.adk import Event
from google.adk.errors import StaleSessionError
from google.adk.events import EventActions
from google.adk.sessions import DatabaseSessionService, Session
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types

USER = "u1"
LEDGER_APP = f"{APP_NAME}.ledger"


def emit(line: dict) -> None:
    sys.stdout.write(json.dumps({"t": time.time(), **line}) + "\n")
    sys.stdout.flush()


class BarrierSessionService(DatabaseSessionService):
    """Holds the process's first `get_session` after its read until `go` exists,
    so two processes can be made to read the same session revision."""

    def __init__(self, db_url: str, *, barrier: Path | None) -> None:
        super().__init__(db_url=db_url)
        self._barrier = barrier

    async def get_session(self, **kwargs: Any) -> Session | None:
        session = await super().get_session(**kwargs)
        if self._barrier is not None:
            barrier, self._barrier = self._barrier, None
            emit({"type": "read"})
            while not (barrier / "go").exists():
                await asyncio.sleep(0.005)
        return session


class Refused(Exception):  # noqa: N818 -- named for the wire 409
    pass


async def ledger(service: DatabaseSessionService, conversation: str) -> Session:
    session = await service.get_session(app_name=LEDGER_APP, user_id=USER, session_id=conversation)
    if session is None:
        session = await service.create_session(
            app_name=LEDGER_APP, user_id=USER, session_id=conversation
        )
    return session


async def ledger_write(service: DatabaseSessionService, session: Session, delta: dict) -> None:
    event = Event(author="studio", actions=EventActions(state_delta=delta))
    await service.append_event(session, event)


async def claim(
    service: DatabaseSessionService, conversation: str, interrupt: str, round_: int
) -> None:
    """Dev's `/interrupt` claim: the pause must be pending at this round, and
    removing it must win the ledger's compare-and-swap."""
    session = await ledger(service, conversation)
    pending = dict(session.state.get("pending", {}))
    if pending.get(interrupt) != round_:
        raise Refused("no open pause")
    del pending[interrupt]
    try:
        await ledger_write(service, session, {"pending": pending})
    except StaleSessionError as error:
        raise Refused("stale ledger") from error


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "answer", "cancel"])
    parser.add_argument("--db", required=True)
    parser.add_argument("--turn", required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--guards", action="store_true")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--decision", default="approve")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--step-s", type=float, default=0.05)
    parser.add_argument("--heartbeat-s", type=float, default=5.0)
    parser.add_argument("--barrier", type=Path)
    args = parser.parse_args()

    service = BarrierSessionService(args.db, barrier=args.barrier)
    runner = build_runner(service, version=args.version, guards=args.guards, log=args.log)
    conversation = args.turn.split("#")[0]

    if args.command == "cancel":
        session = await ledger(service, conversation)
        at = time.time()
        await ledger_write(service, session, {"cancel": True})
        emit({"type": "cancel", "at": at, "committed": time.time()})
        return
    if args.command == "start":
        state = {"steps": args.steps, "step_s": args.step_s}
        await service.create_session(
            app_name=APP_NAME, user_id=USER, session_id=args.turn, state=state
        )
        message = types.Content(role="user", parts=[types.Part(text="go")])
    else:
        if args.guards:
            try:
                await claim(service, conversation, "approve", args.round)
            except Refused as refused:
                emit({"type": "conflict", "reason": str(refused)})
                return
        response = {"decision": args.decision, "round": args.round}
        part = create_request_input_response("approve", response)
        message = types.Content(role="user", parts=[part])
    emit({"type": "claimed"})

    stream_task = asyncio.current_task()
    assert stream_task is not None
    stopped = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(args.heartbeat_s)
            session = await ledger(service, conversation)
            if session.state.get("cancel"):
                await ledger_write(service, session, {"cancel": False})
                stopped.set()
                stream_task.cancel()
                return

    beat = asyncio.create_task(heartbeat()) if args.guards else None
    outcome: dict[str, Any] = {"type": "done"}
    try:
        async with aclosing(
            runner.run_async(user_id=USER, session_id=args.turn, new_message=message)
        ) as stream:
            async for event in stream:
                meta = event.custom_metadata or {}
                if "step" in meta:
                    emit({"type": "step", "n": meta["step"]})
                elif "step_start" in meta:
                    emit({"type": "step_start", "n": meta["step_start"]})
                elif event.long_running_tool_ids:
                    (call,) = event.get_function_calls()
                    interrupt = (call.args or {})["interruptId"]
                    session = await service.get_session(
                        app_name=APP_NAME, user_id=USER, session_id=args.turn
                    )
                    round_ = session.state.get("round") if session else None
                    outcome = {"type": "paused", "interrupt": interrupt, "round": round_}
                    if args.guards:
                        led = await ledger(service, conversation)
                        pending = {**led.state.get("pending", {}), interrupt: round_}
                        await ledger_write(service, led, {"pending": pending})
                elif event.error_code:
                    outcome = {
                        "type": "error",
                        "code": event.error_code,
                        "message": event.error_message,
                    }
    except asyncio.CancelledError:
        if not stopped.is_set():
            raise
        stream_task.uncancel()
        outcome = {"type": "stopped", "stopped_at": time.time()}
    except StaleSessionError:
        outcome = {"type": "conflict", "reason": "stale session"}
    finally:
        if beat is not None:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
    emit(outcome)


asyncio.run(main())
