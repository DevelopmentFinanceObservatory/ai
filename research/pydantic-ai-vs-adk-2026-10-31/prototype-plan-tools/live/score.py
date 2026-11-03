"""Score outside-client runs from their logs, and the planner baseline from its records.

A client run (`runs/<run>.jsonl`, written by `dfo` or `client.py`) is scored on:

- rounds: `propose_plan` calls up to and including the first accepted one, and
  the refusal codes met on the way (`outcome: refused`, or `translation:<kind>`
  for an `ok` answer a translation referral kept from being ready);
- accepted within caps: an accepted proposal within 40 tool calls and 12
  proposals;
- right: the landed plan, the version `run_plan` ran or else the last accepted
  one, as plan-mcp stored it (`runs/plans.jsonl`), uses its task's kinds and does
  what the task expects with its years, by the planner-entry eval's own rules
  (`entry.Task.selects_kinds`, `entry.period_reductions`);
- run finished: a `run_plan` answer with status `finished`;
- how each answer and approval was given: `record_user_answer` answers recorded
  `by` the user (asked by plan-mcp through elicitation) or the client (on the
  model's word; every first-version answer), and each version `run_plan`
  started `approved_by` the user or the client, and each the user declined;
- tool calls, and tokens where the client logged usage (`end` line).

A baseline record (`evals/planner_entry/run.py` output) is scored the same way:
rounds are its proposals up to the accepted one, right is its `plan_kinds`, or its
last proposal's kinds when none was accepted, and its `period_reduction`, by the
same rules. Its entry ends at the first accepted plan, so it has no run.

    python score.py --runs 'gemini-*' --baseline baseline/run/records.jsonl --out results.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "evals" / "planner_entry"))

from entry import (  # noqa: E402
    Task,
    load_tasks,
    period_reductions,
    plan_period_reduction,
)
from planner.plan import Plan  # noqa: E402
from pydantic import ValidationError  # noqa: E402

RUNS = HERE / "runs"
PLANS = RUNS / "plans.jsonl"
MAX_TOOL_CALLS = 40
MAX_PROPOSALS = 12
TASKS: dict[str, Task] = {task.id: task for task in load_tasks()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answer(call: Mapping[str, Any]) -> dict[str, Any]:
    """The tool's answer as the server sent it: `structured`, else the text read as JSON."""
    if isinstance(call.get("structured"), dict):
        return call["structured"]
    try:
        found = json.loads(call["seen"])
    except (json.JSONDecodeError, TypeError):
        return {}
    return found if isinstance(found, dict) else {}


def accepted(response: Mapping[str, Any]) -> bool:
    """An accepted `propose_plan` answer: `outcome: ok` with the summary of a ready plan."""
    return response.get("outcome") == "ok" and "summary" in response


def task_of(run: str, lines: Sequence[Mapping[str, Any]]) -> str:
    for line in lines:
        if line.get("type") == "start":
            return line["task"]
    for task_id in sorted(TASKS, key=len, reverse=True):
        if task_id in run:
            return task_id
    raise ValueError(f"no task for run {run}")


def stored_plans(path: Path = PLANS) -> dict[tuple[str, int], dict[str, Any]]:
    """The plans plan-mcp accepted (`serve_live.record_accepted_plans`), by session and version."""
    if not path.exists():
        return {}
    return {(line["plan_session"], line["version"]): line for line in read_jsonl(path)}


def years(plan: Mapping[str, Any]) -> str | None:
    """What a stored plan does with its years, read as the planner-entry eval reads it
    (`entry.period_reductions`), or `None` when it no longer validates."""
    try:
        return plan_period_reduction(period_reductions(Plan.model_validate(plan))).value
    except ValidationError:
        return None


def refusal_code(response: Mapping[str, Any]) -> str:
    """A refused proposal's code, or the translation's kind for a referral: a
    `referred` answer, or a first-version `ok` answer with no `summary`."""
    if response.get("outcome") == "referred":
        return f"translation:{response.get('code')}"
    if response.get("outcome") == "ok":
        translation = response.get("translation") or {}
        return f"translation:{translation.get('kind', translation.get('outcome'))}"
    return str(response.get("code"))


def score_run(path: Path, plans: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    lines = read_jsonl(path)
    run = path.stem
    task = TASKS[task_of(run, lines)]
    calls = [line for line in lines if line.get("type") == "call"]
    end = next((line for line in lines if line.get("type") == "end"), {})
    rounds = None
    codes: list[str] = []
    proposals = 0
    session: str | None = None
    landed: int | None = None
    ran: int | None = None
    within_caps = False
    asked_user = 0
    answers_by = {"user": 0, "client": 0}
    approvals_by: dict[str, set[int]] = {"user": set(), "client": set(), "declined": set()}
    for number, call in enumerate(calls, start=1):
        response = answer(call)
        if call["tool"] == "start_plan" and response.get("plan_session"):
            session = response["plan_session"]
        if call["tool"] == "record_user_answer" and response.get("outcome") == "ok":
            asked_user += 1
            answers_by[str(response.get("by") or "client")] += 1
        if call["tool"] == "run_plan" and response.get("run_id"):
            # A first-version run took the client's word and states no `approved_by`.
            approvals_by[response.get("approved_by") or "client"].add(response.get("version"))
        if call["tool"] == "run_plan" and (response.get("detail") or {}).get("by") == "user":
            approvals_by["declined"].add(call["args"].get("version"))
        if call["tool"] == "run_plan" and response.get("status") == "finished":
            ran = response.get("version")
            session = call["args"].get("plan_session", session)
        if call["tool"] != "propose_plan":
            continue
        proposals += 1
        if accepted(response):
            landed = response.get("version")
            session = call["args"].get("plan_session", session)
            if rounds is None:
                rounds = proposals
                within_caps = number <= MAX_TOOL_CALLS and proposals <= MAX_PROPOSALS
        elif rounds is None:
            codes.append(refusal_code(response))
    version = ran if ran is not None else landed
    stored = plans.get((session, version)) if session and version is not None else None
    kinds = stored["kinds"] if stored else None
    held_years = years(stored["plan"]) if stored else None
    usage = end.get("usage") or {}
    return {
        "run": run,
        "task": task.id,
        "client": calls[0]["client"] if calls else None,
        "rounds": rounds,
        "proposals": proposals,
        "codes": codes,
        "asked_user": asked_user,
        "answers_by": answers_by,
        "approvals_by": {by: len(versions) for by, versions in approvals_by.items()},
        "elicits": any(call.get("elicits") for call in calls),
        "accepted_within_caps": within_caps,
        "kinds": kinds,
        "kind_pass": kinds is not None and task.selects_kinds(kinds),
        "years": held_years,
        "years_pass": held_years == task.period_reduction.value,
        "right": (
            kinds is not None
            and task.selects_kinds(kinds)
            and held_years == task.period_reduction.value
        ),
        "run_finished": ran is not None,
        "tool_calls": len(calls),
        "input_tokens": usage.get("input_tokens"),
        "cached_tokens": usage.get("cache_read_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "thinking_tokens": usage.get("thinking_tokens"),
        "ending": end.get("ending"),
        "wall_s": end.get("wall_s"),
    }


def score_baseline(record: Mapping[str, Any]) -> dict[str, Any]:
    task = TASKS[record["task"]]
    proposals = record.get("proposals") or []
    rounds = next((n for n, p in enumerate(proposals, start=1) if p.get("outcome") == "ok"), None)
    upto = proposals if rounds is None else proposals[: rounds - 1]
    kinds = record.get("plan_kinds") or (proposals[-1]["kinds"] if proposals else None)
    usage = record.get("calls") or []

    def total(key: str) -> int | None:
        values = [u.get(key) for u in usage]
        return None if not values or None in values else sum(values)

    return {
        "run": f"baseline-{record['task']}-r{record['rep']}",
        "task": task.id,
        "client": "dfo-planner",
        "rounds": rounds,
        "proposals": len(proposals),
        "codes": [str(p.get("code")) for p in upto if p.get("outcome") != "ok"],
        "asked_user": int(record.get("stop_reason") == "asked_user"),
        "answers_by": None,
        "approvals_by": None,
        "elicits": None,
        "accepted_within_caps": bool(record.get("accepted")),
        "kinds": kinds,
        "kind_pass": kinds is not None and task.selects_kinds(kinds),
        "years": record.get("period_reduction"),
        "years_pass": record.get("period_reduction") == task.period_reduction.value,
        "right": (
            kinds is not None
            and task.selects_kinds(kinds)
            and record.get("period_reduction") == task.period_reduction.value
        ),
        "run_finished": None,
        "tool_calls": len(record.get("tool_calls") or []),
        "input_tokens": total("prompt_tokens"),
        "cached_tokens": total("cached_tokens"),
        "output_tokens": total("output_tokens"),
        "thinking_tokens": total("thinking_tokens"),
        "ending": record.get("stop_reason"),
        "wall_s": record.get("wall_s"),
    }


def _mean(values: Iterable[float | None], places: int = 0) -> str:
    kept = [v for v in values if v is not None]
    return f"{statistics.mean(kept):,.{places}f}" if kept else "-"


def right(rows: Sequence[Mapping[str, Any]]) -> str:
    n = len(rows)
    kinds = sum(r["kind_pass"] for r in rows)
    held = sum(r["years_pass"] for r in rows)
    return f"{sum(r['right'] for r in rows)}/{n} ({kinds}, {held})"


def given(rows: Sequence[Mapping[str, Any]]) -> str:
    """How rows' answers and approvals were given: user/client answers, then
    user/client approvals with the user's refusals in brackets."""
    if any(r["answers_by"] is None for r in rows):
        return "- | -"
    answers = [sum(r["answers_by"][by] for r in rows) for by in ("user", "client")]
    approvals = [sum(r["approvals_by"][by] for r in rows) for by in ("user", "client", "declined")]
    return f"{answers[0]}/{answers[1]} | {approvals[0]}/{approvals[1]} ({approvals[2]})"


def table(scored: Sequence[Mapping[str, Any]]) -> str:
    """One row per task: runs accepted, mean rounds, codes met, right, finished, how
    answers and approvals were given, calls, tokens."""
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        by_task[row["task"]].append(row)
    lines = [
        "| Task | Accepted | Rounds | Refusal codes before acceptance | Right (kinds, years) | Run finished "
        "| Answers by user/client | Approvals by user/client (declined) "
        "| Tool calls | Input tokens | Output + thinking |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task_id, rows in by_task.items():
        n = len(rows)
        codes = sorted({code for row in rows for code in row["codes"]})
        finished = [row["run_finished"] for row in rows]
        lines.append(
            f"| `{task_id}` | {sum(r['accepted_within_caps'] for r in rows)}/{n} "
            f"| {' '.join(str(r['rounds'] or '-') for r in rows)} "
            f"| {', '.join(f'`{c}`' for c in codes) or 'none'} "
            f"| {right(rows)} "
            f"| {'-' if None in finished else f'{sum(finished)}/{n}'} "
            f"| {given(rows)} "
            f"| {_mean(r['tool_calls'] for r in rows)} "
            f"| {_mean(r['input_tokens'] for r in rows)} "
            f"| {_mean((r['output_tokens'] or 0) + (r['thinking_tokens'] or 0) if r['output_tokens'] is not None else None for r in rows)} |"
        )
    n = len(scored)
    finished = [r["run_finished"] for r in scored]
    lines.append(
        f"| all | {sum(r['accepted_within_caps'] for r in scored)}/{n} "
        f"| mean {_mean([r['rounds'] for r in scored], 1)} | "
        f"| {right(scored)} "
        f"| {'-' if None in finished else f'{sum(finished)}/{n}'} "
        f"| {given(scored)} "
        f"| {_mean(r['tool_calls'] for r in scored)} "
        f"| {_mean(r['input_tokens'] for r in scored)} "
        f"| {_mean((r['output_tokens'] or 0) + (r['thinking_tokens'] or 0) if r['output_tokens'] is not None else None for r in scored)} |"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", action="append", default=[], help="glob under runs/")
    parser.add_argument("--baseline", type=Path, help="a planner-entry records.jsonl")
    parser.add_argument("--out", type=Path, help="write the tables and rows here")
    args = parser.parse_args()
    sections: list[str] = []
    rows: list[dict[str, Any]] = []
    plans = stored_plans()
    for pattern in args.runs:
        paths = sorted(p for p in RUNS.glob(f"{pattern}.jsonl") if p != PLANS)
        if not paths:
            raise SystemExit(f"no runs match {pattern}")
        scored = [score_run(path, plans) for path in paths]
        rows += scored
        sections.append(f"## {pattern} ({len(scored)} runs)\n\n{table(scored)}")
    if args.baseline is not None:
        records = [r for r in read_jsonl(args.baseline) if r.get("status") == "ok"]
        scored = [score_baseline(r) for r in records]
        rows += scored
        sections.append(f"## baseline ({len(scored)} runs)\n\n{table(scored)}")
    text = "\n\n".join(sections)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
        args.out.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"scored {len(rows)} runs", file=sys.stderr)


if __name__ == "__main__":
    main()
