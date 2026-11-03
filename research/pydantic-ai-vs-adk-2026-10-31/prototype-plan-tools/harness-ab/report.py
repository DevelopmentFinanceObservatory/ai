"""Score a live run of `run_ab.py` against the decision rule and print its tables.

    python report.py runs/live          # writes runs/live/report.md

Every run is scored by the planner-entry eval's own reduction (`summarise.reduce_run`).
Right means the task's kinds and years (`kind_pass` and `years_expected`), or, on a
task whose right ending is a refusal and an ask, that ending (`expected_ending`).
Wall time from the brief to the first accepted plan is read over accepted runs.
Each B/A ratio carries a 95% interval from a bootstrap over tasks, the runs of a
drawn task kept together in both arms.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from entry import load_tasks
from summarise import Run, reduce_run

TASKS = {task.id: task for task in load_tasks()}
DRAWS = 4000

type Stat = Callable[[Sequence[Run]], float]


def right(run: Run) -> bool:
    if run.refusal_task:
        return run.expected_ending
    return run.kind_pass and run.years_expected


def p90(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))]


STATS: dict[str, tuple[Stat, float]] = {
    "mean input tokens": (lambda rs: statistics.fmean(r.input_tokens for r in rs), 1.02),
    "mean output + thinking tokens": (
        lambda rs: statistics.fmean(r.output_tokens + r.thinking_tokens for r in rs),
        1.02,
    ),
    "median wall s to accepted plan": (
        lambda rs: statistics.median(r.wall_s for r in rs if r.accepted),
        1.05,
    ),
    "p90 wall s to accepted plan": (lambda rs: p90([r.wall_s for r in rs if r.accepted]), 1.10),
}


def load(directory: Path) -> dict[str, list[Run]]:
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    latest = {(r["arm"], r["task"], r["rep"]): r for r in rows if line_ok(r)}
    by_arm: dict[str, list[Run]] = defaultdict(list)
    for (arm, task, _), record in sorted(latest.items()):
        by_arm[arm].append(reduce_run(record, TASKS[task]))
    return by_arm


def line_ok(record: dict[str, Any]) -> bool:
    return record["status"] == "ok" and not record.get("surface_misses")


def ratio_interval(a: Sequence[Run], b: Sequence[Run], stat: Stat) -> tuple[float, float]:
    tasks = sorted({run.task for run in a})
    a_by = {t: [r for r in a if r.task == t] for t in tasks}
    b_by = {t: [r for r in b if r.task == t] for t in tasks}
    rng = random.Random(7)
    ratios = []
    for _ in range(DRAWS):
        drawn = [rng.choice(tasks) for _ in tasks]
        try:
            ratios.append(
                stat([r for t in drawn for r in b_by[t]])
                / stat([r for t in drawn for r in a_by[t]])
            )
        except (statistics.StatisticsError, ZeroDivisionError, IndexError):
            continue
    ratios.sort()
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]


def spread(runs: Sequence[Run], value: Callable[[Run], float]) -> str:
    values = [value(r) for r in runs]
    return f"{statistics.fmean(values):,.0f} (sd {statistics.stdev(values):,.0f})"


def report(directory: Path) -> str:
    by_arm = load(directory)
    a = by_arm["A"]
    arms = ["A", *sorted(name for name in by_arm if name != "A")]
    lines = ["## Decision rule, B against A", ""]
    lines.append("| Criterion | A | B arm | B | B/A | 95% interval | Limit | Verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, (stat, limit) in STATS.items():
        for arm in arms[1:]:
            b = by_arm[arm]
            ratio = stat(b) / stat(a)
            low, high = ratio_interval(a, b, stat)
            verdict = "pass" if ratio <= limit else "fail"
            lines.append(
                f"| {name} | {stat(a):,.1f} | {arm} | {stat(b):,.1f} | {ratio:.3f} | "
                f"{low:.3f} to {high:.3f} | {limit:.2f} | {verdict} |"
            )
    for name, value in (("accepted", lambda r: r.accepted), ("right", right)):
        for arm in arms[1:]:
            got_a = sum(map(value, a))
            got_b = sum(map(value, by_arm[arm]))
            verdict = "pass" if got_b >= got_a else "fail"
            lines.append(
                f"| {name} | {got_a}/{len(a)} | {arm} | {got_b}/{len(by_arm[arm])} | | | "
                f">= A | {verdict} |"
            )
    lines += ["", "## Per arm", ""]
    lines.append(
        "| Arm | Runs | Accepted | Right | Kind pass | Years | Model calls | Input tokens | "
        "Cached | Output + thinking | Wall s median | Wall s p90 | Budget closed |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in arms:
        runs = by_arm[arm]
        walls = [r.wall_s for r in runs]
        lines.append(
            f"| {arm} | {len(runs)} | {sum(r.accepted for r in runs)} | "
            f"{sum(map(right, runs))} | {sum(r.kind_pass for r in runs)} | "
            f"{sum(r.years_expected for r in runs)} | "
            f"{statistics.fmean(r.calls for r in runs):.2f} | "
            f"{spread(runs, lambda r: r.input_tokens)} | "
            f"{spread(runs, lambda r: r.cached_tokens)} | "
            f"{spread(runs, lambda r: r.output_tokens + r.thinking_tokens)} | "
            f"{statistics.median(walls):.2f} | {p90(walls):.2f} | "
            f"{sum(r.budget_closed is not None for r in runs)} |"
        )
    lines += ["", "## Per task", ""]
    lines.append(
        "| Task | Arm | Endings | Right | Model calls | Input tokens | Output + thinking | Wall s |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for task in sorted({r.task for r in a}, key=list(TASKS).index):
        for arm in arms:
            runs = [r for r in by_arm[arm] if r.task == task]
            lines.append(
                f"| `{task}` | {arm} | {' '.join(r.stop for r in runs)} | "
                f"{sum(map(right, runs))}/{len(runs)} | "
                f"{' '.join(str(r.calls) for r in runs)} | "
                f"{statistics.fmean(r.input_tokens for r in runs):,.0f} | "
                f"{statistics.fmean(r.output_tokens + r.thinking_tokens for r in runs):,.0f} | "
                f"{' '.join(f'{r.wall_s:.1f}' for r in runs)} |"
            )
    lines += ["", *tool_overhead(directory)]
    return "\n".join(lines) + "\n"


def tool_overhead(directory: Path) -> list[str]:
    """Per tool and arm: the median time from the model's call to its response, and in
    arm B each MCP call's client time less plan-mcp's own time on it."""
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    tool_s: dict[tuple[str, str], list[float]] = defaultdict(list)
    overhead: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not line_ok(row):
            continue
        for item in row.get("tool_times") or []:
            tool_s[(row["arm"], item["tool"])].append(item["s"])
        client = row.get("mcp_calls") or []
        server = row.get("server_times") or []
        for c, s in zip(client, server, strict=False):
            if c["tool"] == s["tool"]:
                overhead[(row["arm"], c["tool"])].append(c["client_s"] - s["server_s"])
    lines = ["## Tool calls", ""]
    lines.append("| Arm | Tool | Calls | Median s, call to response | Median MCP overhead ms |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key in sorted(set(tool_s) | set(overhead)):
        times = tool_s.get(key) or []
        held = overhead.get(key) or []
        called = f"{statistics.median(times):.4f}" if times else "harness call"
        extra = f"{statistics.median(held) * 1000:.2f}" if held else ""
        lines.append(f"| {key[0]} | {key[1]} | {len(times) or len(held)} | {called} | {extra} |")
    return lines


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    text = report(directory)
    (directory / "report.md").write_text(text)
    print(text)
