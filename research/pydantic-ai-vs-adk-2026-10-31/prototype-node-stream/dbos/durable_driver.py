"""Run steps of one turn against a session database and a DBOS system database,
one JSON line per event.

Steps: `start` sends the user's message, `answer` answers the planner's ask and
`approve` approves the plan. `--hold-call N` parks `propose_plan` call N after its
first progress event, so a parent can kill the process there. `--mode stage_a` runs
stage A's `Mode.LIVE` node without DBOS.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dbos import DBOS
from durable_graph import DMode
from durable_graph import build_runner as build_durable_runner
from durable_planner import DurableDeps, launch
from graph import APPROVE_INTERRUPT, Mode, ask_interrupt_id
from graph import build_runner as build_stage_a_runner
from planner_agent import build_planner
from turn import (
    USER,
    answer,
    events,
    interrupt_of,
    new_session,
    progress_of,
    session_service,
    user_text,
)

SCOPE = "Ghana only"
STAGE_A = "stage_a"


def emit(line: dict) -> None:
    sys.stdout.write(json.dumps(line) + "\n")
    sys.stdout.flush()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--dbos-db", type=Path, required=True)
    parser.add_argument("--executor-id", required=True)
    parser.add_argument("--session", default="s1")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--hold-call", type=int)
    parser.add_argument("--mode", choices=[*DMode, STAGE_A], default=DMode.DURABLE)
    parser.add_argument("steps", nargs="+", choices=["start", "answer", "approve"])
    args = parser.parse_args()

    deps = DurableDeps(log_file=args.log_file)
    if args.hold_call is not None:
        deps.gate, deps.gate_call, deps.gate_timeout = asyncio.Event(), args.hold_call, 600
    if args.mode == STAGE_A:
        runner = build_stage_a_runner(
            session_service(args.db), agent=build_planner(deps), deps=deps, mode=Mode.LIVE
        )
    else:
        runner = build_durable_runner(session_service(args.db), deps=deps, mode=DMode(args.mode))
        launch(args.dbos_db, args.executor_id)
    for step in args.steps:
        if step == "start":
            await new_session(runner, args.session)
            message = user_text("go")
        elif step == "answer":
            message = answer(ask_interrupt_id("ask-1"), {"answer": SCOPE})
        else:
            message = answer(APPROVE_INTERRUPT, {"approved": True})
        emit({"type": "step", "step": step})
        async for event in events(runner, args.session, message):
            if (progress := progress_of(event)) is not None:
                emit({"type": "progress", **progress})
            elif (interrupt := interrupt_of(event)) is not None:
                emit({"type": "interrupt", "id": interrupt["interruptId"]})
            elif event.error_code:
                emit({"type": "error", "code": event.error_code})
            elif event.output is not None:
                emit({"type": "output", "node": event.node_info.path, "output": event.output})
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=USER, session_id=args.session
    )
    assert session is not None
    emit({"type": "end", "events": len(session.events)})
    if args.mode != STAGE_A:
        DBOS.destroy()


asyncio.run(main())
