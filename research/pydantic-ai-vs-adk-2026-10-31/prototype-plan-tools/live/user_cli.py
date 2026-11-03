"""The simulated user, for a tester playing the client model.

    user open --task <task> --run <run>     print what the user asks first
    user say --run <run> '<your message>'   send the user your message; print their reply
    user end --run <run> --ending <ending>  close the run: answered, stopped or cap

Every turn is appended to `runs/<run>.jsonl` beside the run's tool calls. The
user's script stays with the simulated user.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import simuser
from rig import RunLog

ENDINGS = ("answered", "stopped", "cap")


def turns(run: str) -> tuple[str, list[tuple[str, str]], int]:
    """The run's task, its turns so far and its tool-call count."""
    path = RunLog(run).path
    if not path.exists():
        raise SystemExit(f"no run {run}; start it with `user open`")
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    task = next(line["task"] for line in lines if line.get("type") == "start")
    said = [(line["who"], line["text"]) for line in lines if line.get("type") == "say"]
    calls = sum(line.get("type") == "call" for line in lines)
    return task, said, calls


def main() -> int:
    parser = argparse.ArgumentParser(prog="user", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    opened = sub.add_parser("open")
    opened.add_argument("--task", required=True)
    opened.add_argument("--run", required=True)
    say = sub.add_parser("say")
    say.add_argument("--run", required=True)
    say.add_argument("message")
    end = sub.add_parser("end")
    end.add_argument("--run", required=True)
    end.add_argument("--ending", required=True, choices=ENDINGS)
    args = parser.parse_args()
    log = RunLog(args.run)
    if args.command == "open":
        if log.path.exists():
            raise SystemExit(f"run {args.run} exists; pick a new run name")
        question = simuser.opening(args.task)
        log.write({"type": "start", "task": args.task, "rep": 1, "client_model": "tester"})
        log.write({"type": "say", "who": "User", "text": question})
        print(question)
        return 0
    task, said, calls = turns(args.run)
    if args.command == "end":
        log.write({"type": "end", "ending": args.ending, "tool_calls": calls})
        print(json.dumps({"run": args.run, "ending": args.ending, "tool_calls": calls}))
        return 0
    said.append(("Assistant", args.message))
    log.write({"type": "say", "who": "Assistant", "text": args.message})
    answered = asyncio.run(simuser.reply(task, said))
    if answered.done:
        log.write({"type": "say", "who": "User", "text": "", "done": True})
        print("[the user has nothing more to say: end the run]")
        return 0
    log.write({"type": "say", "who": "User", "text": answered.reply})
    print(answered.reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
