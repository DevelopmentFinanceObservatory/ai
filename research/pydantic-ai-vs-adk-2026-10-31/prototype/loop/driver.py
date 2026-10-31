"""Run one step of a turn against a store file; print each frame as a JSON line.

    python driver.py --db turns.db start --cid c1 --text "ODA to Ghana"
    python driver.py --db turns.db resume --cid c1 --interrupt c1:1 --answer '"Ghana only"'
    python driver.py --db turns.db resume --cid c1 --interrupt c1:2 --answer '{"kind":"approve","revision":1}'
    python driver.py --db turns.db recover --cid c1

Each line is flushed as it is written, so a parent process can read a frame and kill
this one. Exit code 3 means the loop refused the step.
"""

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from agents import Deps, orchestrator, planner
from loop import Loop, RefusedError
from store import Store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--progress", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument(
        "--hold-at",
        type=int,
        help="block propose_plan call N forever after its first progress event",
    )
    parser.add_argument("--drop-agent", action="append", default=[])
    parser.add_argument("--tool-log")
    parser.add_argument(
        "--linger", type=float, default=0, help="seconds to stay alive after the last frame"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--cid", required=True)
    start.add_argument("--text", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--cid", required=True)
    resume.add_argument("--interrupt", required=True)
    resume.add_argument("--answer", required=True, help="JSON")
    recover = commands.add_parser("recover")
    recover.add_argument("--cid", required=True)
    return parser.parse_args(argv)


async def main(args: argparse.Namespace) -> int:
    deps = Deps(
        progress=args.progress,
        delay=args.delay,
        gate=asyncio.Event() if args.hold_at else None,
        gate_call=args.hold_at,
        tool_log=Path(args.tool_log) if args.tool_log else None,
    )
    store = Store(args.db, drop_agents=frozenset(args.drop_agent))
    loop = Loop(store, orchestrator, planner, deps)
    match args.command:
        case "start":
            frames = loop.start_turn(args.cid, args.text)
        case "resume":
            frames = loop.resume(args.cid, args.interrupt, json.loads(args.answer))
        case "recover":
            frames = loop.recover(args.cid)
    try:
        async for frame in frames:
            print(json.dumps(frame, sort_keys=True), flush=True)
    except RefusedError as refusal:
        print(json.dumps({"type": "refused", "reason": str(refusal)}), flush=True)
        return 3
    time.sleep(args.linger)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
