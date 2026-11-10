"""Recompute the tables of RESULTS.md from `runs/full/records.jsonl`.

    python research/pai-planner-live/live-turns/analysis.py            # print the tables
    python research/pai-planner-live/live-turns/analysis.py --check    # compare with RESULTS.md

The tables sit in RESULTS.md between `<!-- tables -->` and `<!-- /tables -->`; `--check`
exits 1 when that block differs from what the records give.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "runs" / "full" / "records.jsonl"
RESULTS = HERE / "RESULTS.md"
Q18 = HERE / "runs" / "q18.jsonl"
OPEN, CLOSE = "<!-- tables -->", "<!-- /tables -->"
Q18_OPEN, Q18_CLOSE = "<!-- q18 -->", "<!-- /q18 -->"
SEAM_ENDINGS = {
    "plan_not_ready": "gave up (run ended with no plan declared ready)",
    "budget_exhausted": "round budget spent",
}
ARMS = ("adk", "pai")
PLANNER_LABELS = {"Working on a plan", "Revising the plan"}
BUDGET_DETAIL = "This request used all the work allowed"
Record = dict[str, Any]


def load(path: Path = RECORDS) -> list[Record]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sorted(rows, key=lambda r: (r["question_index"], r["arm"]))


def agent_group(name: str) -> str:
    if name in {"dfo_planner", "planner (pydantic ai)"}:
        return "planner"
    return "orchestrator" if name == "dfo_orchestrator" else "other"


def planner_frames(record: Record) -> list[Record]:
    return [f for f in record["frames"] if f.get("label") in PLANNER_LABELS]


def planner_ran(record: Record) -> bool:
    return bool(planner_frames(record))


def count(record: Record, event: str) -> int:
    return sum(f["event"] == event for f in record["frames"])


def stages(record: Record) -> dict[str, bool]:
    """The stages a turn reached."""
    return {
        "planner ran": planner_ran(record),
        "asked a question": bool(record["asks"]),
        "  asked before the planner started": any(
            ask["at"] < (planner_start(record) or float("inf")) for ask in record["asks"]
        ),
        "  asked after the planner started": any(
            ask["at"] >= (planner_start(record) or float("inf")) for ask in record["asks"]
        ),
        "plan proposed": count(record, "plan.proposed") > 0,
        "plan executed": any(
            f.get("label") == "Starting the plan" and f.get("status") == "complete"
            for f in record["frames"]
        ),
        "answered with data": outcome(record) == "answered with data",
    }


def failed_steps(record: Record) -> list[Record]:
    return [f for f in record["frames"] if f.get("status") == "failed"]


def failure_class(frame: Record) -> str:
    """A failed step's class: the failed step's label for a planner cut, otherwise the
    first sentence of the wire detail, which is fixed text per failure code."""
    detail = frame.get("detail") or ""
    if frame.get("label") == "No plan could be made":
        return "planner budget cut" if detail.startswith(BUDGET_DETAIL) else "no plan made"
    return detail.split(":")[0].rstrip(".")[:90] or "failed step without detail"


def record_failure_classes(record: Record) -> set[str]:
    classes = {failure_class(f) for f in failed_steps(record)}
    if record["ending"] != "done":
        classes.add(f"turn ended: {record['ending']}")
    classes |= {f"error frame: {f['error']}" for f in record["frames"] if f["event"] == "error"}
    if record.get("exception"):
        classes.add("driver exception")
    return classes


def outcome(record: Record) -> str:
    """The turn's final outcome."""
    if record["ending"] != "done":
        return "error"
    if any(f["event"] == "error" for f in record["frames"]):
        return "error"
    if any(failure_class(f) == "planner budget cut" for f in failed_steps(record)):
        return "budget cut"
    if count(record, "component.created") or any(
        f.get("label") == "Starting the plan" and f.get("status") == "complete"
        for f in record["frames"]
    ):
        return "answered with data"
    return "could not answer"


def planner_start(record: Record) -> float | None:
    frames = planner_frames(record)
    return frames[0]["at"] if frames else None


def first_progress_s(record: Record) -> float | None:
    """Seconds from the planner's first status frame to its first progress frame."""
    start = planner_start(record)
    if start is None:
        return None
    for f in record["frames"]:
        if str(f.get("detail") or "").startswith("called ") and f["at"] >= start:
            return round(f["at"] - start, 3)
    return None


def planner_s(record: Record) -> float | None:
    """Seconds from the planner's first status frame to the proposal, or to the planner's
    last closing frame when none was proposed."""
    start = planner_start(record)
    if start is None:
        return None
    proposal = next((f["at"] for f in record["frames"] if f["event"] == "plan.proposed"), None)
    if proposal is not None:
        return round(proposal - start, 3)
    closes = [f["at"] for f in planner_frames(record) if f.get("status") == "complete"]
    return round(max(closes) - start, 3) if closes else None


def tokens(record: Record) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for call in record["model_calls"]:
        group = out[agent_group(call["agent"])]
        group["calls"] += 1
        for key in ("prompt_tokens", "cached_tokens", "output_tokens", "thinking_tokens"):
            group[key] += call[key] or 0
    return out


def by_arm(records: list[Record]) -> dict[str, list[Record]]:
    return {arm: [r for r in records if r["arm"] == arm] for arm in ARMS}


def spread(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.median(values):.1f} ({min(values):.1f} to {max(values):.1f})"


def table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def tables(records: list[Record]) -> str:
    arms = by_arm(records)
    n = {arm: len(rs) for arm, rs in arms.items()}
    out = [f"Turns recorded: {n['adk']} on adk, {n['pai']} on pai.", ""]

    names = list(stages(records[0]))
    rows = [[name, *(sum(stages(r)[name] for r in arms[arm]) for arm in ARMS)] for name in names]
    out += ["### Stages reached", "", table(["stage", *ARMS], rows), ""]

    outcomes = ["answered with data", "could not answer", "budget cut", "error"]
    rows = [[o, *(sum(outcome(r) == o for r in arms[arm]) for arm in ARMS)] for o in outcomes]
    out += ["### Final outcome", "", table(["outcome", *ARMS], rows), ""]

    classes: dict[str, dict[str, list[Record]]] = defaultdict(lambda: {a: [] for a in ARMS})
    for r in records:
        for cls in record_failure_classes(r):
            classes[cls][r["arm"]].append(r)
    rows = []
    for cls, per_arm in sorted(
        classes.items(), key=lambda kv: (-sum(map(len, kv[1].values())), kv[0])
    ):
        example = next(r for a in ARMS for r in per_arm[a])
        rows.append(
            [
                cls,
                len(per_arm["adk"]),
                len(per_arm["pai"]),
                f"{example['arm']} q{example['question_index']}",
            ]
        )
    out += [
        "### Failure classes (turns affected)",
        "",
        table(["class", "adk", "pai", "example"], rows),
        "",
    ]

    only_pai = [
        f"q{r['question_index']}: {cls}"
        for cls, per_arm in sorted(classes.items())
        if not per_arm["adk"]
        for r in per_arm["pai"]
    ]
    out += ["Classes seen only on pai: " + ("; ".join(sorted(only_pai)) or "none"), ""]

    def proposed_planner_s(record: Record) -> float | None:
        return planner_s(record) if count(record, "plan.proposed") else None

    rows = []
    for label, fn in (
        ("planner start to first progress frame (s)", first_progress_s),
        ("planner time (s)", planner_s),
        ("planner time, turns that proposed a plan (s)", proposed_planner_s),
        ("total working time (s)", lambda r: r["wall_s"]),
    ):
        cells = []
        for arm in ARMS:
            values = [v for r in arms[arm] if (v := fn(r)) is not None]
            cells.append(f"{spread(values)}, n={len(values)}")
        rows.append([label, *cells])
    out += ["### Time, median (range)", "", table(["measure", *ARMS], rows), ""]

    rows = []
    for group in ("orchestrator", "planner", "other"):
        for key in ("calls", "prompt_tokens", "cached_tokens", "output_tokens", "thinking_tokens"):
            cells = [sum(tokens(r).get(group, {}).get(key, 0) for r in arms[arm]) for arm in ARMS]
            rows.append([group, key, *cells])
    out += [
        "### Model calls and tokens, summed over all turns",
        "",
        table(["agent", "measure", *ARMS], rows),
        "",
    ]

    rows = [
        [
            f"q{r['question_index']}",
            r["arm"],
            outcome(r),
            r["wall_s"],
            first_progress_s(r) if first_progress_s(r) is not None else "",
            planner_s(r) if planner_s(r) is not None else "",
            "; ".join(sorted(record_failure_classes(r))),
        ]
        for r in records
    ]
    out += [
        "### Every turn",
        "",
        table(
            [
                "question",
                "arm",
                "outcome",
                "working s",
                "first progress s",
                "planner s",
                "failure classes",
            ],
            rows,
        ),
    ]
    return "\n".join(out)


def run_ending(run: Record) -> str:
    """How one planner run ended, as the seam returned it."""
    if run["submitted"]:
        return "plan"
    if run["refused"] is None:
        return "asked"
    code = run["refused"]["code"] if isinstance(run["refused"], dict) else "unreadable"
    if code.startswith("turn_"):
        return f"budget closed ({code})"
    return SEAM_ENDINGS.get(code, f"refused ({code})")


def answer_class(call: Record) -> str:
    """A `propose_plan` answer as the model read it: its outcome and refusal code."""
    response = call["response"]
    if not isinstance(response, dict):
        return "unreadable"
    code = response.get("code")
    return f"{response.get('outcome')}" + (f" / {code}" if code else "")


def count_rows(counts: dict[str, dict[str, int]], head: str) -> str:
    rows = [
        [k, *(counts[a].get(k, 0) for a in ARMS)] for k in sorted(counts["adk"] | counts["pai"])
    ]
    return table([head, *ARMS], rows)


def q18_tables(records: list[Record]) -> str:
    arms = {arm: [r for r in records if r["arm"] == arm] for arm in ARMS}
    out = [f"Turns recorded: {len(arms['adk'])} on adk, {len(arms['pai'])} on pai.", ""]
    first: dict[str, dict[str, int]] = {a: defaultdict(int) for a in ARMS}
    every: dict[str, dict[str, int]] = {a: defaultdict(int) for a in ARMS}
    answers: dict[str, dict[str, int]] = {a: defaultdict(int) for a in ARMS}
    refused_turns = dict.fromkeys(ARMS, 0)
    rounds: dict[str, list[int]] = {a: [] for a in ARMS}
    for r in records:
        runs = r["planner_runs"]
        first[r["arm"]][run_ending(runs[0]) if runs else "planner did not run"] += 1
        seen_refusal = False
        for run in runs:
            every[r["arm"]][run_ending(run)] += 1
            calls = [c for c in run["calls"] if c["tool"] == "propose_plan"]
            rounds[r["arm"]].append(max(len(calls) - 1, 0))
            for call in calls:
                answers[r["arm"]][answer_class(call)] += 1
                seen_refusal |= answer_class(call) != "ok"
        refused_turns[r["arm"]] += seen_refusal
    out += [
        "First planner run of each turn, by how the seam ended it",
        "",
        count_rows(first, "ending"),
        "",
    ]
    out += ["Every planner run", "", count_rows(every, "ending"), ""]
    out += [
        "`propose_plan` answers as the model read them",
        "",
        count_rows(answers, "outcome / code"),
        "",
    ]
    rows = [
        ["turns with at least one refused proposal", *(refused_turns[a] for a in ARMS)],
        [
            "correction rounds per planner run, median (range)",
            *(spread([float(v) for v in rounds[a]]) for a in ARMS),
        ],
    ]
    out += [table(["measure", *ARMS], rows)]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--records", type=Path, default=RECORDS)
    args = parser.parse_args()
    q18 = [json.loads(line) for line in Q18.read_text(encoding="utf-8").splitlines() if line]
    q18.sort(key=lambda r: (r["rep"], r["arm"]))
    text = tables(load(args.records))
    q18_text = q18_tables(q18)
    if not args.check:
        print(text, "", "## q18", "", q18_text, sep="\n")
        return 0
    body = RESULTS.read_text(encoding="utf-8")
    pairs = ((OPEN, CLOSE, text), (Q18_OPEN, Q18_CLOSE, q18_text))
    if any(
        b.split(o, 1)[1].split(c, 1)[0].strip() != t.strip() for o, c, t in pairs for b in [body]
    ):
        print("RESULTS.md differs from the records", file=sys.stderr)
        return 1
    print("RESULTS.md matches the records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
