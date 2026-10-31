"""Run the served ADK planner and its Pydantic AI port on the planner entry tasks, paired.

Each (task, repetition) runs every selected arm, once, live on Gemini. The three arms:

- `baseline`: the ADK entry as `evals/planner_entry/run.py` runs it.
- `pai`: the port in mode `like-adk` (`pai_entry.pai_record`).
- `pai-validated`: the port in mode `validated`.

Entries run four at a time across all arms and share one `PrefixCache`. The arms of one
(task, repetition) start in an order shuffled with a seed fixed by the task and the
repetition, so a resume keeps the order. `tap` writes every model call to `tap.jsonl` in
the run directory, labelled with the entry that made it; the records go to `records.jsonl`
in the layout `run.py` writes, so `summarise.py` reads them. Pass the same `--out` to
resume: finished entries are skipped and errored ones retried.

    uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness python \\
        research/pai-prototype/gemini/driver.py --out /tmp/run --chunk 1/8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "planner_entry"
sys.path.insert(0, str(EVAL_DIR))

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
from planner.settings import AgentSettings  # noqa: E402
from summarise import latest, read_records  # noqa: E402

ARMS = ("baseline", "pai", "pai-validated")
PAI_MODES = {"pai": "like-adk", "pai-validated": "validated"}
# What `summarise.py` reads of a record; a PAI record missing one is an integration error.
REQUIRED = frozenset(
    {
        "schema",
        "arm",
        "task",
        "rep",
        "model",
        "revision",
        "experiment",
        "inputs",
        "status",
        "error",
        "stop_reason",
        "plan_kinds",
        "proposals",
        "first_proposal",
        "calls",
        "budget_closed",
        "surface_misses",
        "wall_s",
    }
)

Triple = tuple[str, str, int]
ArmRunner = Callable[[Task, dict[str, Any], str, int], Awaitable[dict[str, Any]]]


def load_pai_record() -> Callable[..., Awaitable[dict[str, Any]]]:
    """B1's `pai_entry.pai_record`, or `NotImplementedError` while its module is absent."""
    try:
        from pai_entry import pai_record  # noqa: PLC0415 -- absent until the port is built
    except ModuleNotFoundError as exc:
        if exc.name != "pai_entry":
            raise
        raise NotImplementedError(
            "pai_entry is not present, so the Pydantic AI arms cannot run"
        ) from exc
    return pai_record


def shuffled_triples(
    tasks: Sequence[Task], arms: Sequence[str], reps: int, *, seed: int
) -> list[Triple]:
    """Repetition, then task, with each (task, repetition)'s arms in a seeded order, so a
    partial run stays balanced and a resume repeats the order."""
    triples = []
    for rep in range(1, reps + 1):
        for task in tasks:
            order = list(arms)
            random.Random(f"{seed}:{task.id}:{rep}").shuffle(order)
            triples += [(arm, task.id, rep) for arm in order]
    return triples


def select_tasks(tasks: list[Task], *, only: str | None, chunk: str | None) -> list[Task]:
    """`only`'s task ids, then the `K/N` part of the id-sorted list that `chunk` names."""
    if only:
        wanted = only.split(",")
        if missing := set(wanted) - {task.id for task in tasks}:
            raise SystemExit(f"unknown task ids {sorted(missing)}")
        tasks = [task for task in tasks if task.id in wanted]
    if chunk:
        index, count = (int(part) for part in chunk.split("/"))
        if not 1 <= index <= count:
            raise SystemExit(f"--chunk {chunk}: K must be from 1 to N")
        tasks = sorted(tasks, key=lambda task: task.id)
        size = -(-len(tasks) // count)
        tasks = tasks[(index - 1) * size : index * size]
    return tasks


def error_record(
    task: Task, recording: dict[str, Any], *, message: str, wall_s: float
) -> dict[str, Any]:
    """The smallest record `summarise.py` takes, for an entry that failed outside its arm."""
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
    pai_record: Callable[..., Awaitable[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """One entry of `arm`: the record, without the identity the driver stamps on it."""
    started = time.perf_counter()
    try:
        # The hard bound sits beyond the entry's own, which scores a timeout itself.
        async with asyncio.timeout(timeout_s + 60):
            if arm == "baseline":
                run = await run_entry(
                    task, recording, prefix_cache=prefix_cache, timeout_s=timeout_s
                )
                return entry_record(
                    task=task, run=run, arm=arm, rep=0, model="", revision="", experiment=""
                )
            assert pai_record is not None
            return await pai_record(
                task,
                recording,
                arm=arm,
                mode=PAI_MODES[arm],
                prefix_cache=prefix_cache,
                run_timeout_s=timeout_s,
            )
    except Exception as exc:
        return error_record(
            task,
            recording,
            message=f"{type(exc).__name__}: {exc}",
            wall_s=time.perf_counter() - started,
        )


def stamp(
    record: dict[str, Any],
    *,
    arm: str,
    rep: int,
    entry_id: str,
    model: str,
    revision: str,
    experiment: str,
) -> dict[str, Any]:
    """`record` with the identity fields the driver owns; refused if the arm left out a
    field `summarise.py` reads."""
    record = {**record, "arm": arm, "rep": rep, "entry_id": entry_id}
    record |= {"model": model, "revision": revision, "experiment": experiment}
    if missing := REQUIRED - record.keys():
        return {
            **record,
            "status": "error",
            "error": f"the {arm} record lacks {sorted(missing)}",
            "surface_misses": record.get("surface_misses") or [],
            "calls": record.get("calls") or [],
        }
    return record


async def run_all(args: argparse.Namespace, tasks: list[Task], arms: list[str]) -> int:
    """Run every arm, task and repetition not yet recorded, for up to `--passes` passes,
    each retrying the runs that errored. Returns how many still end in an error."""
    records_path: Path = args.out / "records.jsonl"
    recordings = {task.id: load_recording(task.id) for task in tasks}
    by_id = {task.id: task for task in tasks}
    model = AgentSettings().planner_model_name
    at = eval_run.revision()
    experiments = {
        arm: eval_run.experiment(arm, model=model, revision=at, run_timeout=args.run_timeout)
        for arm in arms
    }
    existing = read_records([records_path]) if records_path.exists() else []
    if foreign := sorted(
        {
            r["arm"]
            for r in existing
            if r["arm"] in experiments and r["experiment"] != experiments[r["arm"]]
        }
    ):
        raise SystemExit(
            f"{args.out} holds runs of {', '.join(foreign)} under another model, code or "
            "setting; start a new --out"
        )
    pai_record = load_pai_record() if any(arm in PAI_MODES for arm in arms) else None
    prefix_cache = PrefixCache() if Settings().studio_prefix_cache else None
    gate = asyncio.Semaphore(args.concurrency)
    deadline = None if args.stop_after_s is None else time.monotonic() + args.stop_after_s
    skipped = 0

    async def one(triple: Triple, count: int, total: int) -> None:
        nonlocal skipped
        arm, task_id, rep = triple
        async with gate:
            if deadline is not None and time.monotonic() > deadline:
                skipped += 1
                return
            entry_id = f"{arm}:{task_id}:{rep}:{uuid.uuid4().hex[:8]}"
            with tap.labelled(arm=arm, task=task_id, rep=rep, entry_id=entry_id):
                record = await run_arm(
                    arm,
                    by_id[task_id],
                    recordings[task_id],
                    prefix_cache=prefix_cache,
                    timeout_s=args.run_timeout,
                    pai_record=pai_record,
                )
        record = stamp(
            record,
            arm=arm,
            rep=rep,
            entry_id=entry_id,
            model=model,
            revision=at,
            experiment=experiments[arm],
        )
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{count}/{total}] {arm} {task_id} rep{rep} {record['stop_reason']} "
            f"calls={len(record['calls'])} {record['wall_s']:.1f}s"
            + (f" ERROR {record['error'].splitlines()[0][:160]}" if record["error"] else ""),
            file=sys.stderr,
        )

    for attempt in range(1, args.passes + 1):
        existing = read_records([records_path]) if records_path.exists() else []
        done = {(r["arm"], r["task"], r["rep"]) for r in existing if r["status"] == "ok"}
        pending = [
            triple
            for triple in shuffled_triples(tasks, arms, args.reps, seed=args.seed)
            if triple not in done
        ]
        print(f"pass {attempt} on {model} at {at}: {len(pending)} runs to do", file=sys.stderr)
        if not pending:
            break
        await asyncio.gather(
            *(one(triple, i, len(pending)) for i, triple in enumerate(pending, start=1))
        )
        if skipped:
            print(f"{skipped} runs not started; rerun with --out {args.out}", file=sys.stderr)
            break
    wanted = set(shuffled_triples(tasks, arms, args.reps, seed=args.seed))
    chosen = latest(read_records([records_path])) if records_path.exists() else {}
    return sum(record["status"] == "error" for key, record in chosen.items() if key in wanted)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="run directory; reuse to resume")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated; default all")
    parser.add_argument("--only", help="comma-separated task ids; default every task")
    parser.add_argument("--chunk", help="K/N: the Kth of N slices of the id-sorted tasks")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per task and arm")
    parser.add_argument("--concurrency", type=int, default=4, help="entries in flight, all arms")
    parser.add_argument("--run-timeout", type=float, default=300.0, help="seconds per entry")
    parser.add_argument("--passes", type=int, default=2, help="passes; each retries errors")
    parser.add_argument("--seed", type=int, default=20261031, help="seed for the arm order")
    parser.add_argument(
        "--stop-after-s", type=float, help="start no new entry after this many seconds"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    arms = args.arms.split(",")
    if unknown := set(arms) - set(ARMS):
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {list(ARMS)}")
    if "baseline" not in arms:
        print("no baseline arm: analyse.py needs one in the same --out", file=sys.stderr)
    quiet_logs()
    tasks = select_tasks(load_tasks(), only=args.only, chunk=args.chunk)
    args.out.mkdir(parents=True, exist_ok=True)
    with tap.installed(args.out / "tap.jsonl"):
        errors = asyncio.run(run_all(args, tasks, arms))
    if errors:
        print(f"{errors} runs ended in an error; rerun with --out {args.out}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
