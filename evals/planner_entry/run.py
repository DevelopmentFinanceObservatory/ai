"""Run a named arm against the baseline on every task, and write the records and a summary.

Each (arm, task, repetition) is one planner entry on a fresh session (`entry.run_entry`)
against the live planner model. Runs go repetition by repetition, then task, with the
two arms side by side, so an interrupted run leaves a balanced sample and both arms
meet the same model latency. A rerun with the same `--out` skips every run already
recorded and retries the errors, so a run cut short resumes; it is refused when the
directory holds runs of the same arm under another experiment (`experiment`).

    just eval-planner-entry baseline-replica
    just eval-planner-entry baseline-replica --reps 1 --only share-health-oda-kenya

`--baseline-records` pairs the arm against a stored run instead of running the
baseline now: to measure a change made in production code, run `baseline` alone on
the base revision, then `baseline-replica` on the changed one against those records.
`--baseline-langfuse` pairs against a run pushed to Langfuse, and `--push` pushes this
run once it ends (`langfuse_store`).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import langfuse_store
from arms import ARMS
from entry import (
    EVAL_DIR,
    Task,
    entry_record,
    load_recording,
    load_tasks,
    quiet_logs,
    run_entry,
)
from harness.prefix_cache import PrefixCache
from harness.settings import Settings
from langfuse import Langfuse
from planner.settings import AgentSettings
from summarise import latest, read_records, write_records, write_summary

RESULTS_DIR = EVAL_DIR / ".results"

Triple = tuple[str, str, int]


def revision() -> str:
    """The checkout's commit, marked `+dirty.<digest>` when the code a run executes
    differs from it: the harness and the planner, the shared packages, and this eval.
    The digest covers the differences, so two uncommitted states read apart."""
    root = EVAL_DIR.parents[1]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout

    commit = git("rev-parse", "--short", "HEAD").strip() or "unknown"
    paths = ("services/harness/src", "packages", str(EVAL_DIR.relative_to(root)))
    diff = git("diff", "HEAD", "--", *paths)
    untracked = sorted(git("ls-files", "--others", "--exclude-standard", "--", *paths).split())
    if not diff and not untracked:
        return commit
    digest = hashlib.sha256(diff.encode())
    for name in untracked:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return f"{commit}+dirty.{digest.hexdigest()[:8]}"


def experiment(arm: str, *, model: str, revision: str, run_timeout: float) -> str:
    """A digest of everything a run of `arm` is treated with besides its task: the arm,
    the model, the code at `revision`, the turn budget each request runs under
    (`harness.turn_budget.TurnBudget`), the prefix cache, and the eval's own time
    limit. Records of one arm in one run share it."""
    settings = Settings()
    treatment = {
        "arm": arm,
        "model": model,
        "revision": revision,
        "run_timeout_s": run_timeout,
        "turn_model_time_s": settings.turn_model_time_s,
        "turn_model_calls_max": settings.turn_model_calls_max,
        "turn_wall_s": settings.turn_wall_s,
        "model_first_chunk_s": settings.model_first_chunk_s,
        "prefix_cache": settings.studio_prefix_cache,
    }
    return hashlib.sha256(json.dumps(treatment, sort_keys=True).encode()).hexdigest()[:16]


def plan_triples(
    tasks: Sequence[Task], arms: Sequence[str], reps: int, *, done: set[Triple]
) -> list[Triple]:
    """Repetition, then task, then arm, so a partial run stays balanced and paired."""
    return [
        (arm, task.id, rep)
        for rep in range(1, reps + 1)
        for task in tasks
        for arm in arms
        if (arm, task.id, rep) not in done
    ]


async def run_all(args: argparse.Namespace, tasks: list[Task], arms: list[str]) -> int:
    """Run every arm, task and repetition not yet recorded, for up to `--passes` passes,
    each retrying the runs that errored. Returns how many still end in an error."""
    records_path: Path = args.out / "records.jsonl"
    recordings = {task.id: load_recording(task.id) for task in tasks}
    by_id = {task.id: task for task in tasks}
    model = AgentSettings().planner_model_name
    at = revision()
    experiments = {
        arm: experiment(arm, model=model, revision=at, run_timeout=args.run_timeout) for arm in arms
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
            f"{args.out} holds runs of {', '.join(foreign)} under another model, code, "
            "setting or arm; start a new --out"
        )
    prefix_cache = PrefixCache() if Settings().studio_prefix_cache else None
    gate = asyncio.Semaphore(args.concurrency)

    async def one(triple: Triple, count: int, total: int) -> None:
        arm, task_id, rep = triple
        async with gate:
            run = await run_entry(
                by_id[task_id],
                recordings[task_id],
                prefix_cache=prefix_cache,
                edits=ARMS[arm].edits,
                timeout_s=args.run_timeout,
            )
        record = entry_record(
            task=by_id[task_id],
            run=run,
            arm=arm,
            rep=rep,
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
        pending = plan_triples(tasks, arms, args.reps, done=done)
        print(f"pass {attempt} on {model} at {at}: {len(pending)} runs to do", file=sys.stderr)
        if not pending:
            break
        await asyncio.gather(
            *(one(triple, i, len(pending)) for i, triple in enumerate(pending, start=1))
        )
    wanted = set(plan_triples(tasks, arms, args.reps, done=set()))
    chosen = latest(read_records([records_path])) if records_path.exists() else {}
    return sum(record["status"] == "error" for key, record in chosen.items() if key in wanted)


def stored_baseline(args: argparse.Namespace, client: Langfuse | None) -> list[dict[str, Any]]:
    """The stored baseline runs to pair the arm against, none when the baseline runs now:
    those in `--baseline-records`, or in the Langfuse experiment `--baseline-langfuse`
    names, which are kept beside the run as `baseline-records.jsonl`."""
    if args.baseline_records is not None:
        source = str(args.baseline_records)
        records = read_records([args.baseline_records])
    elif args.baseline_langfuse is not None:
        assert client is not None
        source = f"Langfuse's {args.baseline_langfuse}"
        records = langfuse_store.pull(client, args.baseline_langfuse)
        write_records(args.out / "baseline-records.jsonl", records)
    else:
        return []
    if not (kept := [r for r in records if r["arm"] == args.baseline]):
        raise SystemExit(f"{source} holds no {args.baseline} runs")
    return kept


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(ARMS), help="the arm to set against the baseline")
    parser.add_argument("--baseline", default="baseline", choices=sorted(ARMS))
    stored = parser.add_mutually_exclusive_group()
    stored.add_argument(
        "--baseline-records",
        type=Path,
        help="pair against this stored run's baseline records instead of running it",
    )
    stored.add_argument(
        "--baseline-langfuse",
        metavar="NAME",
        help="pair against the baseline runs pushed to Langfuse as this experiment",
    )
    parser.add_argument("--push", action="store_true", help="push the run to Langfuse at the end")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per task and arm")
    parser.add_argument("--concurrency", type=int, default=4, help="entries in flight at once")
    parser.add_argument("--only", help="comma-separated task ids; default every task")
    parser.add_argument("--out", type=Path, help="run directory; reuse one to resume it")
    parser.add_argument("--run-timeout", type=float, default=300.0, help="seconds per entry")
    parser.add_argument("--passes", type=int, default=3, help="passes; each retries errors")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    quiet_logs()
    every_task = load_tasks()
    tasks = every_task
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [task for task in tasks if task.id in wanted]
        if missing := wanted - {task.id for task in tasks}:
            raise SystemExit(f"unknown task ids {sorted(missing)}")
    stored = args.baseline_records is not None or args.baseline_langfuse is not None
    if stored and args.arm == args.baseline:
        raise SystemExit(
            f"--baseline-records and --baseline-langfuse pair the arm against stored "
            f"{args.baseline} runs, so the arm needs another name: run baseline-replica"
        )
    arms = [args.arm]
    if args.arm != args.baseline and not stored:
        arms.append(args.baseline)
    if args.out is None:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        args.out = RESULTS_DIR / f"{stamp}-{args.arm}"
    args.out.mkdir(parents=True, exist_ok=True)
    client = langfuse_store.connect() if args.push or args.baseline_langfuse else None
    baseline = stored_baseline(args, client)
    errors = asyncio.run(run_all(args, tasks, arms))
    records = read_records([args.out / "records.jsonl"]) + baseline
    summary = write_summary(args.out, records, tasks, arm=args.arm, baseline=args.baseline)
    print(summary)
    if client is not None and args.push:
        for name in langfuse_store.push(client, args.out, tasks=every_task):
            print(f"pushed {name}", file=sys.stderr)
    if errors:
        print(f"{errors} runs ended in an error; rerun with --out {args.out}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
