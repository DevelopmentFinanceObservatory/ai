"""Summarise a run: each arm's outcomes and costs, and the arm paired against the baseline.

One record per (arm, task, repetition) counts: the last that is not an error, else the
last. A run that errored, or that asked data-mcp for something its recording does not
hold, ran in a world the eval cannot vouch for: it is left out of every figure, counted
apart, and named under the switch triggers. A task whose runs showed the planner
different inputs (`entry.inputs_digest`), such as a brief edited or a recording made
again between a stored baseline and a new run, is left out of the summary and listed,
and an arm whose records come from more than one experiment (`run.experiment`) is
refused. Outcomes are scored on the tasks with no `first_proposal_refusal`; a refusal
task is scored on whether it ended as it should, in that refusal and then an ask of the
user. An accepted plan is also read for what it does with the years of its data
(`entry.record_period_reduction`) and held to its task's `period_reduction`.

Pairing is by task. Each task's rate over its repetitions is set against the
baseline's, and a task counts as better, worse or the same. The p-value is a two-sided
exact sign test over the tasks that differ; the interval on a mean difference is a 95%
bootstrap over tasks, since the repetitions of one task are not independent.

    uv run --package harness python evals/planner_entry/summarise.py \\
        --records RUN/records.jsonl --arm baseline-replica --out RUN
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from entry import (
    KINDS,
    PERIOD_REFUSAL,
    RECORD_SCHEMA,
    PeriodReduction,
    Task,
    load_tasks,
    record_period_reduction,
)

BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 20261011

# The conditions for adding a refusal-driven extension tier to the planner's declaration.
TRIGGER_FIRST_PROPOSAL_POINTS = 0.05
TRIGGER_KIND_PASS_TASKS = 2
TRIGGER_P90_SECONDS = 10.0

# Why a run is left out of every figure.
ERRORED = "errored"
MISSED = "missed the recording"


def read_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    lines = (line for path in paths for line in path.read_text(encoding="utf-8").splitlines())
    return check_schema([json.loads(line) for line in lines if line.strip()])


def check_schema(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`records`, refused when any is of another schema than `RECORD_SCHEMA`."""
    if stale := sorted({str(r.get("schema")) for r in records} - {str(RECORD_SCHEMA)}):
        raise ValueError(
            f"records of schema {', '.join(stale)} are not schema {RECORD_SCHEMA}; run them again"
        )
    return records


def write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write `records` to `path` as `read_records` reads them, one JSON object a line."""
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def left_out_reason(record: Mapping[str, Any]) -> str | None:
    """Why the summary leaves `record` out of every figure, or `None` when it counts."""
    if record["status"] == "error":
        return ERRORED
    if record["surface_misses"]:
        return MISSED
    return None


def latest(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """One record per (arm, task, rep): the last that is not an error, else the last."""
    chosen: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in records:
        key = (record["arm"], record["task"], record["rep"])
        held = chosen.get(key)
        if held is None or held["status"] == "error" or record["status"] != "error":
            chosen[key] = record
    return chosen


# --- One run, reduced ----------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    arm: str
    task: str
    refusal_task: bool
    # Whether the accepted plan does with its years what the task's `period_reduction`
    # says; false when no plan was read for it.
    years_expected: bool
    stop: str
    first_ok: bool
    expected_ending: bool
    kind_pass: bool
    first_code: str | None
    unit_stated: int
    datasets: int
    calls: int
    # Whether every model call the run started finished, so its tokens are known.
    usage_known: bool
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    thinking_tokens: int
    wall_s: float
    # The code of the turn-budget bound that closed one of the run's requests, if one did.
    budget_closed: str | None
    # The `propose_plan` calls the run made.
    proposals: int
    # Whether `propose_plan` refused any of them with `PERIOD_REFUSAL`.
    period_refused: bool
    # What the accepted plan does with its years; `None` when no plan was accepted or
    # the record's plan could not be read.
    period_reduction: PeriodReduction | None
    # Whether the task marks an ask `across_years`.
    asks_across_years: bool

    @property
    def years_read(self) -> bool:
        """Whether the run's accepted plan was read for what it does with its years."""
        return self.period_reduction is not None


def _plan_kinds(record: Mapping[str, Any]) -> set[str]:
    """The kinds scored: the accepted plan's, else the last proposal's."""
    if record["plan_kinds"]:
        return set(record["plan_kinds"])
    proposals = record["proposals"]
    return set(proposals[-1]["kinds"]) if proposals else set()


def _first_datasets(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    proposals = record["proposals"]
    if not proposals:
        return []
    requirements = proposals[0]["args"].get("evidence_requirements") or []
    return [r for r in requirements if isinstance(r, dict) and r.get("kind") == "dataset"]


# Pydantic's wording for a value the declared schema rules out: a missing or extra field,
# a wrong type, a tag no kind has. Any other `invalid_plan` is a rule the plan breaks.
_SHAPE = re.compile(
    r"Field required|Input should|Tuple should|Extra inputs|Input tag|should be a valid|"
    r"String should|List should"
)


def refusal_class(proposal: Mapping[str, Any]) -> str:
    """A refused proposal's code, with `invalid_plan` split into `shape` and `rule`."""
    code = proposal.get("code") or str(proposal.get("outcome"))
    if code != "invalid_plan":
        return code
    return (
        "invalid_plan shape"
        if _SHAPE.search(proposal.get("message") or "")
        else "invalid_plan rule"
    )


def reduce_run(record: Mapping[str, Any], task: Task) -> Run:
    kinds = _plan_kinds(record)
    first = record["first_proposal"] or {}
    datasets = [] if task.unit_asked else _first_datasets(record)
    calls = record["calls"]
    finished = [call for call in calls if call["prompt_tokens"] is not None]
    reduction = record_period_reduction(record)
    return Run(
        arm=record["arm"],
        task=record["task"],
        refusal_task=task.first_proposal_refusal is not None,
        years_expected=reduction == task.period_reduction,
        stop=record["stop_reason"],
        first_ok=first.get("outcome") == "ok",
        expected_ending=(
            task.first_proposal_refusal is not None
            and first.get("code") == task.first_proposal_refusal
            and record["stop_reason"] == "asked_user"
        ),
        kind_pass=task.selects_kinds(kinds),
        first_code=None
        if not record["proposals"] or first.get("outcome") == "ok"
        else refusal_class(record["proposals"][0]),
        unit_stated=sum(bool(r.get("unit")) for r in datasets),
        datasets=len(datasets),
        calls=len(calls),
        usage_known=len(finished) == len(calls),
        input_tokens=sum(c["prompt_tokens"] for c in finished),
        cached_tokens=sum(c["cached_tokens"] for c in finished),
        output_tokens=sum(c["output_tokens"] for c in finished),
        thinking_tokens=sum(c["thinking_tokens"] for c in finished),
        wall_s=record["wall_s"],
        budget_closed=record["budget_closed"],
        proposals=len(record["proposals"]),
        period_refused=any(p["code"] == PERIOD_REFUSAL for p in record["proposals"]),
        period_reduction=reduction,
        asks_across_years=task.asks_across_years,
    )


# --- Statistics ----------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    below, above = math.floor(position), math.ceil(position)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def bootstrap(differences: Sequence[float]) -> tuple[float, float]:
    """95% percentile interval of the mean of `differences`, resampling them."""
    generator = random.Random(BOOTSTRAP_SEED)
    means = [
        statistics.fmean(generator.choices(differences, k=len(differences)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact binomial p-value for `better` of `better + worse` tasks."""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2**n
    return min(1.0, 2 * tail)


def per_task(runs: Sequence[Run], metric: Callable[[Run], float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for run in runs:
        grouped.setdefault(run.task, []).append(metric(run))
    return {task: statistics.fmean(values) for task, values in grouped.items()}


@dataclass(frozen=True)
class Paired:
    tasks: int
    better: int
    worse: int
    same: int
    mean: float
    low: float
    high: float
    p: float


def paired(
    arm: Sequence[Run], baseline: Sequence[Run], metric: Callable[[Run], float]
) -> Paired | None:
    ours, theirs = per_task(arm, metric), per_task(baseline, metric)
    shared = sorted(set(ours) & set(theirs))
    if not shared:
        return None
    differences = [ours[task] - theirs[task] for task in shared]
    better = sum(d > 1e-9 for d in differences)
    worse = sum(d < -1e-9 for d in differences)
    low, high = bootstrap(differences)
    return Paired(
        tasks=len(shared),
        better=better,
        worse=worse,
        same=len(shared) - better - worse,
        mean=statistics.fmean(differences),
        low=low,
        high=high,
        p=sign_test(better, worse),
    )


# --- Report --------------------------------------------------------------------------


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _share(successes: int, n: int) -> str:
    return f"{successes}/{n} ({successes / n:.0%})" if n else "n/a"


def _fmean(values: Iterable[float]) -> float:
    listed = list(values)
    return statistics.fmean(listed) if listed else math.nan


def kind_passing_tasks(runs: Sequence[Run], tasks: Collection[str] | None = None) -> int:
    """The tasks, of `tasks` when given, on which more than half of the runs pass kind
    selection: two of three repetitions."""
    rates = per_task([r for r in runs if not r.refusal_task], lambda r: float(r.kind_pass))
    return sum(rate > 0.5 for task, rate in rates.items() if tasks is None or task in tasks)


def arm_numbers(runs: Sequence[Run]) -> dict[str, Any]:
    scored = [r for r in runs if not r.refusal_task]
    refusal = [r for r in runs if r.refusal_task]
    known = [r for r in runs if r.usage_known]
    walls = [r.wall_s for r in runs]
    kind_tasks = per_task(scored, lambda r: float(r.kind_pass))
    return {
        "runs": len(runs),
        "scored_runs": len(scored),
        "first_ok": sum(r.first_ok for r in scored),
        "kind_pass": sum(r.kind_pass for r in scored),
        "kind_pass_tasks": kind_passing_tasks(runs),
        "kind_tasks": len(kind_tasks),
        "refusal_runs": len(refusal),
        "expected_ending": sum(r.expected_ending for r in refusal),
        "unit_stated": sum(r.unit_stated for r in runs),
        "datasets": sum(r.datasets for r in runs),
        "mean_calls": _fmean(r.calls for r in runs),
        "usage_unknown": len(runs) - len(known),
        "mean_input_tokens": _fmean(r.input_tokens for r in known),
        "mean_cached_tokens": _fmean(r.cached_tokens for r in known),
        "mean_output_tokens": _fmean(r.output_tokens for r in known),
        "mean_thinking_tokens": _fmean(r.thinking_tokens for r in known),
        "mean_wall_s": _fmean(walls),
        "p90_wall_s": percentile(walls, 0.9) if walls else math.nan,
        "max_wall_s": max(walls) if walls else math.nan,
        "budget_closed": sum(r.budget_closed is not None for r in runs),
        "budget_closures": dict(
            Counter(r.budget_closed for r in runs if r.budget_closed is not None).most_common()
        ),
        "stops": dict(Counter(r.stop for r in runs).most_common()),
        "first_refusals": dict(
            Counter(r.first_code or "no proposal" for r in scored if not r.first_ok).most_common()
        ),
    }


def rule_tasks(runs: Iterable[Run]) -> set[str]:
    """The tasks the period rule touched: those with a run in which `propose_plan`
    refused a proposal with `PERIOD_REFUSAL`, or accepted a period reduction for asks
    marked across years."""
    return {
        run.task
        for run in runs
        if run.period_refused or run.period_reduction is PeriodReduction.ACROSS_YEARS
    }


def years_numbers(runs: Sequence[Run], touched: Collection[str]) -> dict[str, Any]:
    """What an arm's plans did with their years. `touched` are the tasks the period rule
    touched in any arm (`rule_tasks`), so every arm's rounds are over the same tasks."""
    read = [r for r in runs if r.years_read]
    unmarked = [r for r in read if not r.asks_across_years]
    on_touched = [r for r in runs if r.task in touched]
    accepted = [r for r in on_touched if r.stop == "accepted"]
    return {
        "years_read": len(read),
        "years_expected": sum(r.years_expected for r in read),
        "unmarked_accepted": len(unmarked),
        "unmarked_reducing": sum(r.period_reduction is not PeriodReduction.NONE for r in unmarked),
        "period_first_refusals": sum(r.first_code == PERIOD_REFUSAL for r in runs),
        "rule_tasks": len({r.task for r in on_touched}),
        "rule_runs": len(on_touched),
        "rule_accepted": len(accepted),
        "rule_mean_rounds": _fmean(r.proposals for r in accepted),
    }


def years_table(numbers: Mapping[str, Mapping[str, Any]]) -> str:
    header = [
        "Arm",
        "Years as the task expects",
        "No ask across years: accepted plans holding a period reduction",
        f"First proposals refused `{PERIOD_REFUSAL}`",
        "Rounds to acceptance where the rule applied",
    ]
    rows = []
    for arm, n in numbers.items():
        rounds = (
            f"{n['rule_mean_rounds']:.2f} over {n['rule_accepted']} of {n['rule_runs']} runs "
            f"on {n['rule_tasks']} tasks"
            if n["rule_accepted"]
            else f"n/a over 0 of {n['rule_runs']} runs on {n['rule_tasks']} tasks"
        )
        rows.append(
            [
                arm,
                _share(n["years_expected"], n["years_read"]),
                _share(n["unmarked_reducing"], n["unmarked_accepted"]),
                _share(n["period_first_refusals"], n["runs"]),
                rounds,
            ]
        )
    return table(header, rows)


def arm_table(
    numbers: Mapping[str, Mapping[str, Any]],
    attempts: Mapping[str, int],
    left_out: Mapping[str, Mapping[str, Counter[str]]],
) -> str:
    header = [
        "Arm",
        "Runs scored (attempts)",
        "Errored",
        "Missed the recording",
        "First proposal accepted",
        "Kind pass, runs",
        "Kind pass, tasks",
        "Refusal tasks ending right",
        "Unasked units",
        "Model calls",
        "Input tokens (cached)",
        "Output tokens",
        "Thinking tokens",
        "Wall s: mean, p90, max",
        "Closed by the turn budget",
    ]
    rows = []
    for arm, n in numbers.items():
        unknown = f"; {n['usage_unknown']} runs unknown" if n["usage_unknown"] else ""
        rows.append(
            [
                arm,
                f"{n['runs']} ({attempts.get(arm, 0)})",
                str(left_out[arm][ERRORED].total()),
                str(left_out[arm][MISSED].total()),
                _share(n["first_ok"], n["scored_runs"]),
                _share(n["kind_pass"], n["scored_runs"]),
                _share(n["kind_pass_tasks"], n["kind_tasks"]),
                _share(n["expected_ending"], n["refusal_runs"]),
                f"{n['unit_stated']} of {n['datasets']}",
                f"{n['mean_calls']:.2f}",
                f"{n['mean_input_tokens']:,.0f} ({n['mean_cached_tokens']:,.0f}){unknown}",
                f"{n['mean_output_tokens']:,.0f}",
                f"{n['mean_thinking_tokens']:,.0f}",
                f"{n['mean_wall_s']:.1f}, {n['p90_wall_s']:.1f}, {n['max_wall_s']:.1f}",
                str(n["budget_closed"]),
            ]
        )
    return table(header, rows)


OUTCOMES: dict[str, Callable[[Run], float]] = {
    "First proposal accepted": lambda r: float(r.first_ok),
    "Kind pass": lambda r: float(r.kind_pass),
}
# Each cost, and whether it is read only from runs whose every model call finished.
COSTS: dict[str, tuple[Callable[[Run], float], bool]] = {
    "Model calls": (lambda r: float(r.calls), False),
    "Input tokens": (lambda r: float(r.input_tokens), True),
    "Output tokens": (lambda r: float(r.output_tokens), True),
    "Thinking tokens": (lambda r: float(r.thinking_tokens), True),
    "Wall s": (lambda r: r.wall_s, False),
}


def triggers(
    arm: Sequence[Run], baseline: Sequence[Run], first: Paired | None
) -> list[tuple[str, bool, str]]:
    """The switch triggers, each with whether it fired and the numbers.

    Kind pass, p90 and runs the turn budget closed are compared over the tasks both
    arms ran, so a task with no finished run in one arm moves none of them. The budget
    trigger compares the arms; a run the budget closed in either is reported apart, as
    an alarm, since a model call that stalls fires it in identical arms.
    """
    shared = {r.task for r in arm} & {r.task for r in baseline}
    rows = []
    if first is not None:
        fired = first.mean <= -TRIGGER_FIRST_PROPOSAL_POINTS and first.high < 0
        rows.append(
            (
                f"First-proposal acceptance down {TRIGGER_FIRST_PROPOSAL_POINTS:.0%} or more, "
                "interval below zero",
                fired,
                f"{first.mean:+.1%} [{first.low:+.1%}, {first.high:+.1%}]",
            )
        )
    scored = {r.task for r in arm if not r.refusal_task} & {
        r.task for r in baseline if not r.refusal_task
    }
    ours, theirs = kind_passing_tasks(arm, scored), kind_passing_tasks(baseline, scored)
    rows.append(
        (
            f"Kind pass down {TRIGGER_KIND_PASS_TASKS} or more tasks",
            theirs - ours >= TRIGGER_KIND_PASS_TASKS,
            f"{ours} against {theirs} of {len(scored)} shared tasks",
        )
    )
    on_shared = {
        name: [r for r in runs if r.task in shared]
        for name, runs in (("arm", arm), ("baseline", baseline))
    }
    p90 = {name: percentile([r.wall_s for r in runs], 0.9) for name, runs in on_shared.items()}
    rows.append(
        (
            f"p90 wall time up {TRIGGER_P90_SECONDS:.0f} s or more",
            p90["arm"] - p90["baseline"] >= TRIGGER_P90_SECONDS,
            f"{p90['arm']:.1f} s against {p90['baseline']:.1f} s",
        )
    )
    closed = {
        name: sum(r.budget_closed is not None for r in runs) for name, runs in on_shared.items()
    }
    rows.append(
        (
            "More runs closed by the turn budget than the baseline",
            closed["arm"] > closed["baseline"],
            f"{closed['arm']} of {len(on_shared['arm'])} runs against "
            f"{closed['baseline']} of {len(on_shared['baseline'])}",
        )
    )
    return rows


def _closures_text(*numbers: Mapping[str, Any]) -> str:
    """The bounds that closed runs, with how many each closed, over `numbers`' arms."""
    closures: Counter[str] = Counter()
    for n in numbers:
        closures.update(n["budget_closures"])
    return ", ".join(f"{code} {count}" for code, count in closures.most_common())


def task_table(by_arm: Mapping[str, Sequence[Run]], tasks: Sequence[Task]) -> str:
    """Per task: first proposals accepted and kind passes, over its repetitions, and the
    accepted plans that do with their years what the task expects, over those read."""
    arms = list(by_arm)
    header = ["Task", "Kind", *(f"{arm}: first ok, kind pass, years" for arm in arms)]
    rows = []
    for task in tasks:
        cells = []
        for arm in arms:
            runs = [r for r in by_arm[arm] if r.task == task.id]
            if not runs:
                cells.append("-")
            elif task.first_proposal_refusal:
                cells.append(f"ending {sum(r.expected_ending for r in runs)}/{len(runs)}")
            else:
                read = [r for r in runs if r.years_read]
                cells.append(
                    f"{sum(r.first_ok for r in runs)}/{len(runs)}, "
                    f"{sum(r.kind_pass for r in runs)}/{len(runs)}, "
                    + (f"{sum(r.years_expected for r in read)}/{len(read)}" if read else "-")
                )
        if any(cell != "-" for cell in cells):
            rows.append([task.id, task.kind, *cells])
    return table(header, rows)


def _left_out_text(left_out: Mapping[str, Mapping[str, Counter[str]]]) -> str:
    """Which tasks' runs each arm left out, and why, as one sentence."""
    parts = [
        f"{name}, {reason} on {', '.join(f'`{task}`' for task in sorted(tasks))}"
        for name, reasons in left_out.items()
        for reason, tasks in reasons.items()
        if tasks
    ]
    errored = any(reasons[ERRORED] for reasons in left_out.values())
    return (
        "These leave out the runs that errored or missed their recording: "
        + "; ".join(parts)
        + (". Rerun with the same `--out` to retry the errors." if errored else ".")
    )


def build_report(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Task],
    *,
    arm: str,
    baseline: str,
) -> tuple[str, dict[str, Any]]:
    by_id = {task.id: task for task in tasks}
    names = (baseline, arm) if arm != baseline else (arm,)
    kept = [r for r in records if r["task"] in by_id and r["arm"] in names]
    for name in names:
        if len(experiments := {r["experiment"] for r in kept if r["arm"] == name}) > 1:
            raise ValueError(
                f"{name}'s records come from {len(experiments)} experiments; summarise each "
                "run directory on its own"
            )
    attempts = Counter(r["arm"] for r in kept)
    chosen = latest(kept)
    shown: dict[str, set[str]] = {}
    for (_name, task_id, _rep), record in chosen.items():
        shown.setdefault(task_id, set()).add(record["inputs"])
    changed = sorted(task_id for task_id, digests in shown.items() if len(digests) > 1)
    chosen = {key: record for key, record in chosen.items() if key[1] not in changed}
    by_arm: dict[str, list[Run]] = {name: [] for name in names}
    left_out: dict[str, dict[str, Counter[str]]] = {
        name: {ERRORED: Counter(), MISSED: Counter()} for name in names
    }
    misses: dict[str, Counter[str]] = {name: Counter() for name in names}
    for (name, task_id, _rep), record in sorted(chosen.items()):
        reason = left_out_reason(record)
        if reason is None:
            by_arm[name].append(reduce_run(record, by_id[task_id]))
            continue
        left_out[name][reason][task_id] += 1
        if reason == MISSED:
            misses[name].update(record["surface_misses"])
    touched = rule_tasks(run for runs in by_arm.values() for run in runs)
    numbers = {
        name: arm_numbers(runs) | years_numbers(runs, touched) for name, runs in by_arm.items()
    }
    models = sorted({r["model"] for r in chosen.values()})
    revisions = {
        name: sorted({r["revision"] for r in chosen.values() if r["arm"] == name})
        for name in by_arm
    }
    ran = sorted({task_id for (_a, task_id, _r) in chosen})
    left = any(reasons[reason] for reasons in left_out.values() for reason in reasons)
    sections = [
        f"# Planner entry eval: {arm} against {baseline}"
        if arm != baseline
        else f"# Planner entry eval: {arm}",
        f"{len(chosen)} runs over {len(ran)} tasks on {', '.join(models)}; "
        + "; ".join(f"{name} at {', '.join(revs)}" for name, revs in revisions.items())
        + ". Rates count runs; tasks with a first-proposal refusal are scored only on "
        "whether they ended in that refusal and an ask. Runs that errored or missed their "
        "recording are left out of every figure; attempts count every entry started, "
        "retries included. Unasked units count dataset requirements stating a unit in first "
        "proposals of tasks whose question and brief name none. Tokens are means over runs "
        "whose every model call finished, input including cached.",
        "## Per arm",
        arm_table(numbers, attempts, left_out),
    ]
    if changed:
        sections.append(
            f"Left out, since their runs showed the planner different inputs: "
            f"{', '.join(f'`{task_id}`' for task_id in changed)}."
        )
    machine: dict[str, Any] = {
        "arms": numbers,
        "attempts": dict(attempts),
        "left_out": {
            name: {reason: dict(tasks) for reason, tasks in reasons.items()}
            for name, reasons in left_out.items()
        },
        "changed_inputs": changed,
    }
    shared = {r.task for r in by_arm[arm]} & {r.task for r in by_arm[baseline]}
    if arm != baseline and not shared:
        sections.append("The arms share no finished task, so nothing is paired.")
    if arm != baseline and shared:
        scored = {name: [r for r in runs if not r.refusal_task] for name, runs in by_arm.items()}
        outcome_rows, machine_pairs = [], {}
        first = None
        for name, metric in OUTCOMES.items():
            result = paired(scored[arm], scored[baseline], metric)
            if result is None:
                continue
            if name == "First proposal accepted":
                first = result
            machine_pairs[name] = result.__dict__
            outcome_rows.append(
                [
                    name,
                    str(result.tasks),
                    f"{result.better} / {result.worse} / {result.same}",
                    f"{result.mean:+.1%} [{result.low:+.1%}, {result.high:+.1%}]",
                    f"{result.p:.2f}",
                ]
            )
        cost_rows = []
        for name, (metric, needs_usage) in COSTS.items():
            ours, theirs = (
                [r for r in by_arm[n] if r.usage_known or not needs_usage] for n in (arm, baseline)
            )
            result = paired(ours, theirs, metric)
            if result is None:
                continue
            machine_pairs[name] = result.__dict__
            cost_rows.append(
                [
                    name,
                    str(result.tasks),
                    f"{result.mean:+,.1f} [{result.low:+,.1f}, {result.high:+,.1f}]",
                ]
            )
        fired = triggers(by_arm[arm], by_arm[baseline], first)
        alarm = {name: numbers[name]["budget_closed"] for name in names}
        machine["paired"] = machine_pairs
        machine["triggers"] = [
            {"trigger": name, "fired": hit, "numbers": text} for name, hit, text in fired
        ]
        machine["budget_alarm"] = alarm
        sections += [
            "## Paired against the baseline",
            "Per task, the arm's rate over its repetitions minus the baseline's.",
            table(
                [
                    "Outcome",
                    "Tasks",
                    "Better / worse / same",
                    "Mean difference [95%]",
                    "Sign test p",
                ],
                outcome_rows,
            ),
            table(["Cost per run", "Tasks", "Mean difference [95%]"], cost_rows),
            "## Switch triggers",
            *([_left_out_text(left_out)] if left else []),
            table(
                ["Trigger", "Fired", "Numbers"],
                [[name, "yes" if hit else "no", text] for name, hit, text in fired],
            ),
            *(
                [
                    f"Budget alarm: the turn budget closed {alarm[arm]} of {arm}'s runs "
                    f"and {alarm[baseline]} of {baseline}'s, by "
                    + _closures_text(numbers[arm], numbers[baseline])
                    + "."
                ]
                if any(alarm.values())
                else []
            ),
        ]
    stops = sorted({stop for n in numbers.values() for stop in n["stops"]})
    sections += [
        "## How entries ended",
        table(
            ["Arm", *stops],
            [
                [name, *(str(n["stops"].get(stop, 0)) for stop in stops)]
                for name, n in numbers.items()
            ],
        ),
        "## Why first proposals were refused",
        table(
            ["Arm", "Codes"],
            [
                [
                    name,
                    ", ".join(f"{code} {count}" for code, count in n["first_refusals"].items())
                    or "none",
                ]
                for name, n in numbers.items()
            ],
        ),
    ]
    sections += [
        "## Years",
        "What accepted plans do with the years of their data. A plan does as its task expects "
        "when its period reduction is the task's `period_reduction`. The second column is over "
        "the accepted plans of tasks that mark no ask `across_years`; `propose_plan` refuses a "
        "period reduction there, so only a run of a revision without the rule counts any. The "
        "rule applied to a task when, in any arm, it refused a proposal or accepted a period "
        "reduction for asks marked across years; rounds are `propose_plan` calls up to the "
        "accepted one.",
        years_table(numbers),
    ]
    if any(misses.values()):
        sections += [
            "## Recording misses",
            "Calls the recording held no answer for, which the planner was told it could not "
            "reach. Production would have answered them, so these runs are left out.",
            table(
                ["Arm", "Call", "Times"],
                [
                    [name, f"`{key}`", str(count)]
                    for name, counted in misses.items()
                    for key, count in counted.most_common()
                ],
            ),
        ]
    sections += [
        "## Per task",
        task_table(by_arm, [task for task in tasks if task.id in ran]),
    ]
    by_kind = Counter(by_id[t].kind for t in ran)
    machine["tasks_per_kind"] = {kind: by_kind[kind] for kind in KINDS}
    return "\n\n".join(sections) + "\n", machine


def write_summary(
    out: Path,
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[Task],
    *,
    arm: str,
    baseline: str,
) -> Path:
    report, machine = build_report(records, tasks, arm=arm, baseline=baseline)
    (out / "summary.md").write_text(report, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(machine, indent=1), encoding="utf-8")
    return out / "summary.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="JSONL files")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--out", type=Path, required=True, help="directory for the summary")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    path = write_summary(
        args.out, read_records(args.records), load_tasks(), arm=args.arm, baseline=args.baseline
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
