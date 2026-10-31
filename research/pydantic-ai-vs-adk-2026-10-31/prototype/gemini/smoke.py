"""Live smoke of the Pydantic AI planner entry: a few tasks, one repetition, each mode.

Runs `pai_entry.pai_record` on live Gemini with one shared `PrefixCache`, writes the
records to `smoke-records.jsonl` beside this file, and has `summarise.py` read them and
build a summary pairing the `like-adk` arm against the `validated` one, so a record it
would refuse fails here.

    uv run --env-file .env --with 'pydantic-ai-slim[google]==2.64.0' --package harness \\
        python research/pai-prototype/gemini/smoke.py [--only ID,...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "evals" / "planner_entry")]

import pai_entry  # noqa: E402
import summarise  # noqa: E402
from entry import load_recording, load_tasks, quiet_logs  # noqa: E402
from harness.prefix_cache import PrefixCache  # noqa: E402

SMOKE_TASKS = ("aggregate-average-oda-kenya", "sum-net-flows-aid-debt-ghana")
ARMS = {"pai-like-adk": "like-adk", "pai-validated": "validated"}


async def run(task_ids: Sequence[str], out: Path) -> None:
    tasks = {task.id: task for task in load_tasks()}
    cache = PrefixCache()
    records = []
    for task_id in task_ids:
        task = tasks[task_id]
        for arm, mode in ARMS.items():
            record = await pai_entry.pai_record(
                task,
                load_recording(task_id),
                arm=arm,
                mode=mode,
                prefix_cache=cache,  # type: ignore[arg-type]
            )
            records.append(record)
            calls = record["calls"]
            print(
                f"{arm:<14} {task_id:<40} {record['stop_reason']:<12} calls={len(calls)} "
                f"cached={[c['cached_tokens'] for c in calls]} wall={record['wall_s']:.1f}s"
                + (f" ERROR {record['error'].splitlines()[0]}" if record["error"] else ""),
                file=sys.stderr,
            )
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--out", type=Path, default=HERE / "smoke")
    args = parser.parse_args(argv)
    quiet_logs()
    task_ids = args.only.split(",") if args.only else list(SMOKE_TASKS)
    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "records.jsonl"
    asyncio.run(run(task_ids, records_path))
    records = summarise.read_records([records_path])
    print(
        summarise.write_summary(
            args.out, records, load_tasks(), arm="pai-like-adk", baseline="pai-validated"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
