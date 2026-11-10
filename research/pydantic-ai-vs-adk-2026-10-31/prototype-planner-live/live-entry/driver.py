"""Run dev's ADK planner entry and the Pydantic AI entry on live Gemini, paired.

Each (task, repetition) runs both arms once. The arm that goes first alternates from one
pair to the next, the two entries of a pair start together, and entries share one
`PrefixCache` and a fixed number of slots. A record is appended to `records.jsonl` as soon
as its entry ends. A rerun with the same `--out` skips every (arm, task, repetition) that
has a record, so a run stops and resumes in chunks; `--stop-after-s` starts no new entry
after that many seconds. A failed entry keeps its record, since a failure is a result.

A rate-limit error (HTTP 429) is not a result: the entry waits, backs off and runs again,
up to `--rate-limit-retries` times, and the record counts the retries. Calls the planner's
own retry layer repeats after a 429 show in the record's `tap` as errored calls.

    uv run --env-file <env> --with 'pydantic-ai-slim[google]==2.66.0' --package harness \\
        python research/pai-planner-live/live-entry/driver.py --out <dir> --stop-after-s 480
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "services" / "harness").is_dir())
for path in (
    ROOT / "evals" / "planner_entry",
    ROOT / "research" / "pai-planner-live" / "agent",
    HERE,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run as eval_run  # noqa: E402
import tap  # noqa: E402
from entry import (  # noqa: E402
    RECORD_SCHEMA,
    Task,
    entry_record,
    inputs_digest,
    load_recording,
    load_tasks,
    quiet_logs,
    run_entry,
)
from harness.prefix_cache import PrefixCache  # noqa: E402
from harness.settings import Settings  # noqa: E402
from pai_entry import run_pai_entry  # noqa: E402
from planner.settings import AgentSettings  # noqa: E402

ARMS = ("adk", "pai")
RATE_LIMIT_MARKS = ("429", "RESOURCE_EXHAUSTED")

Key = tuple[str, str, int]


def pair_order(task_index: int, rep: int) -> tuple[str, str]:
    """The arms of one (task, repetition) in start order; it alternates pair by pair."""
    return ARMS if (task_index + rep) % 2 == 0 else ARMS[::-1]


def pending_keys(tasks: Sequence[Task], reps: int, *, done: set[Key]) -> list[Key]:
    """Repetition, then task, the two arms of a pair side by side in alternating order."""
    return [
        (arm, task.id, rep)
        for rep in range(1, reps + 1)
        for index, task in enumerate(tasks)
        for arm in pair_order(index, rep)
        if (arm, task.id, rep) not in done
    ]


def is_rate_limited(record: dict[str, Any]) -> bool:
    """Whether the entry ended in a rate-limit error."""
    return bool(record["error"]) and any(mark in record["error"] for mark in RATE_LIMIT_MARKS)


def error_record(
    task: Task, recording: dict[str, Any], *, message: str, wall_s: float
) -> dict[str, Any]:
    """The record of an entry that failed outside the entry's own error handling."""
    return {
        "schema": RECORD_SCHEMA,
        "inputs": inputs_digest(task, recording),
        "status": "error",
        "error": message[:3000],
        "stop_reason": "error",
        "accepted": False,
        "plan_kinds": None,
        "plan": None,
        "tool_calls": [],
        "proposals": [],
        "first_proposal": None,
        "calls": [],
        "budget_closed": None,
        "surface_misses": [],
        "wall_s": round(wall_s, 3),
    }


async def run_arm(
    arm: str,
    task: Task,
    recording: dict[str, Any],
    *,
    prefix_cache: PrefixCache | None,
    timeout_s: float,
) -> dict[str, Any]:
    """One attempt at one entry: the entry's record, and the model calls the tap saw."""
    started = time.perf_counter()
    stamp = {"rep": 0, "model": "", "revision": "", "experiment": ""}
    with tap.collecting() as calls:
        try:
            # The hard bound sits beyond the entry's own, which scores a timeout itself.
            async with asyncio.timeout(timeout_s + 60):
                if arm == "adk":
                    run = await run_entry(
                        task, recording, prefix_cache=prefix_cache, timeout_s=timeout_s
                    )
                else:
                    run = (
                        await run_pai_entry(
                            task, recording, prefix_cache=prefix_cache, timeout_s=timeout_s
                        )
                    ).run
            record = entry_record(task=task, run=run, arm=arm, **stamp)
        except Exception as exc:
            record = error_record(
                task,
                recording,
                message=f"{type(exc).__name__}: {exc}",
                wall_s=time.perf_counter() - started,
            )
    return {**record, "tap": calls}


async def run_with_backoff(
    arm: str,
    task: Task,
    recording: dict[str, Any],
    *,
    prefix_cache: PrefixCache | None,
    timeout_s: float,
    retries: int,
) -> dict[str, Any]:
    """`run_arm` until it ends in anything but a rate-limit error, backing off between
    attempts. The record carries the retries made and what each discarded attempt cost."""
    discarded: list[dict[str, Any]] = []
    while True:
        record = await run_arm(arm, task, recording, prefix_cache=prefix_cache, timeout_s=timeout_s)
        if not is_rate_limited(record) or len(discarded) >= retries:
            return {**record, "retries": len(discarded), "discarded": discarded}
        discarded.append(
            {"wall_s": record["wall_s"], "error": record["error"].splitlines()[0][:200]}
        )
        await asyncio.sleep(20 * 2 ** len(discarded) * random.uniform(0.75, 1.25))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def run_all(args: argparse.Namespace, tasks: list[Task]) -> int:
    """Run every (arm, task, repetition) without a record. Returns how many were left."""
    records_path: Path = args.out / "records.jsonl"
    recordings = {task.id: load_recording(task.id) for task in tasks}
    by_id = {task.id: task for task in tasks}
    model = AgentSettings().planner_model_name
    revision = eval_run.revision()
    experiments = {
        arm: eval_run.experiment(arm, model=model, revision=revision, run_timeout=args.run_timeout)
        for arm in ARMS
    }
    existing = read_jsonl(records_path)
    if foreign := {r["arm"] for r in existing if r["experiment"] != experiments[r["arm"]]}:
        raise SystemExit(
            f"{args.out} holds runs of {sorted(foreign)} under another model, code or "
            "setting; start a new --out"
        )
    done = {(r["arm"], r["task"], r["rep"]) for r in existing}
    pending = pending_keys(tasks, args.reps, done=done)
    print(f"{len(done)} done, {len(pending)} to do on {model} at {revision}", file=sys.stderr)
    prefix_cache = PrefixCache() if Settings().studio_prefix_cache else None
    gate = asyncio.Semaphore(args.concurrency)
    deadline = None if args.stop_after_s is None else time.monotonic() + args.stop_after_s
    finished = 0

    async def one(key: Key) -> None:
        nonlocal finished
        arm, task_id, rep = key
        async with gate:
            if deadline is not None and time.monotonic() > deadline:
                return
            record = await run_with_backoff(
                arm,
                by_id[task_id],
                recordings[task_id],
                prefix_cache=prefix_cache,
                timeout_s=args.run_timeout,
                retries=args.rate_limit_retries,
            )
        record |= {
            "arm": arm,
            "rep": rep,
            "task": task_id,
            "entry_id": uuid.uuid4().hex[:8],
            "model": model,
            "revision": revision,
            "experiment": experiments[arm],
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        finished += 1
        print(
            f"[{finished}/{len(pending)}] {arm} {task_id} rep{rep} {record['stop_reason']} "
            f"calls={len(record['tap'])} {record['wall_s']:.1f}s retries={record['retries']}"
            + (f" ERROR {record['error'].splitlines()[0][:160]}" if record["error"] else ""),
            file=sys.stderr,
        )

    await asyncio.gather(*(one(key) for key in pending))
    return len(pending) - finished


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="run directory; reuse to resume")
    parser.add_argument("--only", help="comma-separated task ids; default every task")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per task and arm")
    parser.add_argument("--concurrency", type=int, default=4, help="entries in flight")
    parser.add_argument("--run-timeout", type=float, default=300.0, help="seconds per entry")
    parser.add_argument("--rate-limit-retries", type=int, default=4, help="reruns after a 429")
    parser.add_argument("--stop-after-s", type=float, help="start no new entry after this long")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    quiet_logs()
    tasks = load_tasks()
    if args.only:
        wanted = args.only.split(",")
        if missing := set(wanted) - {task.id for task in tasks}:
            raise SystemExit(f"unknown task ids {sorted(missing)}")
        tasks = [task for task in tasks if task.id in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    with tap.installed():
        left = asyncio.run(run_all(args, tasks))
    if left:
        print(f"{left} entries not started; rerun with --out {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
