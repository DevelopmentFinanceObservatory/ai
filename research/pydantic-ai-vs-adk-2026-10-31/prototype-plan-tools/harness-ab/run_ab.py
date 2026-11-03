"""Run arms A, B-inproc and B-http (or `--arms`) live on the sample tasks, one entry at a time.

Entries go repetition by repetition, then task, with the three arms back to back
in an order that rotates from one task to the next, so drift in the model's
latency falls on every arm. Each record is appended to `<out>/records.jsonl`
as it ends; rerunning with the same `--out` skips the entries already recorded
and retries the ones that errored.

    python run_ab.py --reps 3 --out runs/live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ab import ARMS, Arm, Current, build_server, plan_client, record, run_b
from entry import load_recording, load_tasks, quiet_logs, run_entry
from harness.prefix_cache import PrefixCache
from harness.settings import Settings
from planner.settings import AgentSettings
from run import revision
from tasks import SAMPLE

HERE = Path(__file__).resolve().parent


def _done(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["arm"], r["task"], r["rep"]) for r in rows if r["status"] == "ok"}


async def main(args: argparse.Namespace) -> int:
    quiet_logs()
    tasks = {task.id: task for task in load_tasks() if task.id in SAMPLE}
    order = [task_id for task_id in SAMPLE if not args.only or task_id in args.only.split(",")]
    recordings = {task_id: load_recording(task_id) for task_id in order}
    model = AgentSettings().planner_model_name
    at = revision()
    prefix_cache = PrefixCache() if Settings().studio_prefix_cache else None
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    current = Current()
    server = build_server(current)
    async with plan_client(server, "inproc") as inproc, plan_client(server, "http") as http:
        arms = {
            "A": Arm("A"),
            "B-inproc": Arm("B-inproc", client=inproc, server=server),
            "B-http": Arm("B-http", client=http, server=server),
            "B-served": Arm("B-served", client=inproc, server=server, served_declaration=True),
        }
        names = args.arms.split(",")
        for attempt in range(1, args.passes + 1):
            done = _done(path)
            todo = [
                (arm, task_id, rep)
                for rep in range(1, args.reps + 1)
                for index, task_id in enumerate(order)
                for arm in (names[(index + rep + k) % len(names)] for k in range(len(names)))
                if (arm, task_id, rep) not in done
            ]
            print(f"pass {attempt}: {len(todo)} entries on {model}", file=sys.stderr)
            if not todo:
                break
            for count, (name, task_id, rep) in enumerate(todo, start=1):
                arm = arms[name]
                started = len(server.server_times)
                if arm.client is None:
                    run = await run_entry(
                        tasks[task_id],
                        recordings[task_id],
                        prefix_cache=prefix_cache,
                        timeout_s=args.run_timeout,
                    )
                else:
                    run = await run_b(
                        tasks[task_id],
                        recordings[task_id],
                        client=arm.client,
                        current=current,
                        prefix_cache=prefix_cache,
                        timeout_s=args.run_timeout,
                        served_declaration=arm.served_declaration,
                    )
                row = record(
                    tasks[task_id], run, arm=arm, rep=rep, model=model, revision=at, started=started
                )
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"[{count}/{len(todo)}] {name} {task_id} r{rep} {row['stop_reason']} "
                    f"calls={len(row['calls'])} {row['wall_s']:.1f}s"
                    + (f" ERROR {row['error'].splitlines()[0][:160]}" if row["error"] else ""),
                    file=sys.stderr,
                )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    latest = {(r["arm"], r["task"], r["rep"]): r for r in rows}
    return sum(r["status"] == "error" for r in latest.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--arms", default="A,B-inproc,B-http", help=f"of {', '.join(ARMS)}")
    parser.add_argument("--only", help="comma-separated task ids; default the ten samples")
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "live")
    parser.add_argument("--run-timeout", type=float, default=300.0)
    parser.add_argument("--passes", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main(parse_args())) else 0)
