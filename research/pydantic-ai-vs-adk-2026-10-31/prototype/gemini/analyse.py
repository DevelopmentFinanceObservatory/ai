"""Read a driver run's tap file and records, and write RESULTS.md.

One row per pass line per Pydantic AI arm, set beside the baseline's numbers:

- W2: a request that carries `cached_content` carries no system instruction, tools or
  tool config.
- W3: a replayed function call whose original response part had a thought signature still
  has one, and no thought summary is replayed.
- W4: the cached share of prompt tokens on an entry's second and later calls is within 5
  points of the baseline's, or above it.
- W5: the median seconds to a stream's first chunk is at most 1.10 times the baseline's.
- Parity: each task's first request holds the same contents as the baseline's.
- O1: the planner entry eval's own summary, paired against the baseline, fires no switch
  trigger and no task falls from passing to failing in a majority of its runs.
- W1 comes from `parity.json`, which the port's builder writes.

    uv run --with 'pydantic-ai-slim[google]==2.64.0' --package harness python \\
        research/pai-prototype/gemini/analyse.py --out <run directory>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parents[3] / "evals" / "planner_entry"
sys.path.insert(0, str(EVAL_DIR))

from entry import Task, load_tasks  # noqa: E402
from summarise import (  # noqa: E402
    build_report,
    latest,
    left_out_reason,
    percentile,
    read_records,
    reduce_run,
    table,
)
from tap import canonical, sha  # noqa: E402

BASELINE = "baseline"
CACHED_SHARE_POINTS = 0.05
FIRST_CHUNK_RATIO = 1.10

Call = dict[str, Any]
# An arm's entries, each its model calls in the order they were made.
Entries = list[list[Call]]


@dataclass(frozen=True)
class Verdict:
    line: str
    arm: str
    ok: bool | None
    numbers: str
    baseline: str

    @property
    def word(self) -> str:
        return {True: "pass", False: "FAIL", None: "inconclusive"}[self.ok]


# --- Reading ---------------------------------------------------------------------------


def read_tap(path: Path) -> list[Call]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def chosen_entries(records: Iterable[Mapping[str, Any]], tap: Sequence[Call]) -> dict[str, Entries]:
    """Per arm, the model calls of the entry each (arm, task, rep) counts for: the record
    `summarise.latest` keeps, so a retried error's earlier calls are left out."""
    calls: dict[str, list[Call]] = defaultdict(list)
    for call in tap:
        if call["kind"] == "generate":
            calls[call["entry_id"]].append(call)
    entries: dict[str, Entries] = defaultdict(list)
    for (arm, _task, _rep), record in sorted(latest(records).items()):
        entries[arm].append(sorted(calls.get(record["entry_id"], []), key=lambda c: c["seq"]))
    return dict(entries)


def _calls(entries: Entries) -> Iterable[Call]:
    return (call for entry in entries for call in entry)


# --- The checks ------------------------------------------------------------------------


def w2(entries: Entries) -> dict[str, int]:
    """Requests carrying a cache, and those that also carry what the cache holds."""
    cached = [c for c in _calls(entries) if c["cached_content"]]
    violations = [
        c for c in cached if c["has_system_instruction"] or c["has_tools"] or c["has_tool_config"]
    ]
    return {
        "requests": sum(1 for _ in _calls(entries)),
        "cached": len(cached),
        "violations": len(violations),
    }


def w3(entries: Entries) -> dict[str, int]:
    """Function calls replayed in later requests of an entry, set against the response
    that first carried each. A call is matched by name and arguments, in order."""
    counts: Counter[str] = Counter()
    for entry in entries:
        originals: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for call in entry:
            seen: Counter[tuple[str, str]] = Counter()
            for replay in call["request_function_calls"]:
                key = (replay["name"], replay["args_sha"])
                index, seen[key] = seen[key], seen[key] + 1
                counts["replayed"] += 1
                if index >= len(originals[key]):
                    counts["unmatched"] += 1
                    continue
                original = originals[key][index]
                counts["signed"] += original["has_sig"]
                counts["lost"] += original["has_sig"] and not replay["has_sig"]
                counts["changed"] += (
                    original["has_sig"]
                    and replay["has_sig"]
                    and original["sig_sha"] != replay["sig_sha"]
                )
                counts["added"] += replay["has_sig"] and not original["has_sig"]
            counts["thought_parts"] += call["request_thought_parts"]
            for original in call["response_function_calls"]:
                originals[(original["name"], original["args_sha"])].append(original)
    return {
        key: counts[key]
        for key in ("replayed", "signed", "lost", "changed", "added", "unmatched", "thought_parts")
    }


def w4(entries: Entries) -> dict[str, float | int]:
    """Cached tokens over prompt tokens, on the second and later calls of each entry."""
    later = [c for entry in entries for c in entry[1:] if c["usage"] and c["usage"]["prompt"]]
    prompt = sum(c["usage"]["prompt"] for c in later)
    cached = sum(c["usage"]["cached"] or 0 for c in later)
    return {
        "calls": len(later),
        "prompt": prompt,
        "cached": cached,
        "share": cached / prompt if prompt else float("nan"),
    }


def w5(entries: Entries) -> dict[str, float | int]:
    """Seconds to the first chunk of each streamed call: median, p90 and the median over
    each entry's first call alone."""
    firsts = [c["first_chunk_s"] for c in _calls(entries) if c["stream"] and c["first_chunk_s"]]
    opening = [
        entry[0]["first_chunk_s"]
        for entry in entries
        if entry and entry[0]["stream"] and entry[0]["first_chunk_s"]
    ]
    nan = float("nan")
    return {
        "streams": len(firsts),
        "p50": percentile(firsts, 0.5) if firsts else nan,
        "p90": percentile(firsts, 0.9) if firsts else nan,
        "first_call_n": len(opening),
        "first_call_p50": percentile(opening, 0.5) if opening else nan,
    }


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """The JSON path where `a` and `b` first differ, or None when they are equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                return f"{path}/{key}"
            if (found := first_difference(a[key], b[key], f"{path}/{key}")) is not None:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        for index, (x, y) in enumerate(zip(a, b, strict=False)):
            if (found := first_difference(x, y, f"{path}/{index}")) is not None:
                return found
        return None if len(a) == len(b) else f"{path}/<length {len(a)} against {len(b)}>"
    return None if a == b else path or "/"


def first_requests(
    records: Iterable[Mapping[str, Any]], tap: Sequence[Call]
) -> dict[tuple[str, str], list[Any]]:
    """Per (arm, task), the distinct contents of its runs' first request, one per
    distinct canonical form."""
    first_of: dict[str, Call] = {}
    for call in sorted((c for c in tap if c["kind"] == "generate"), key=lambda c: c["seq"]):
        first_of.setdefault(call["entry_id"], call)
    firsts: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for (arm, task, _rep), record in latest(records).items():
        if (call := first_of.get(record["entry_id"])) is not None:
            firsts[arm, task][sha(canonical(call["contents"]))] = call["contents"]
    return {key: list(found.values()) for key, found in firsts.items()}


def merged(contents: Any) -> Any:
    """`contents` with each run of adjacent same-role contents joined into one, so two
    requests that group the same parts differently compare equal."""
    if not isinstance(contents, list) or not all(isinstance(c, dict) for c in contents):
        return contents
    joined: list[dict[str, Any]] = []
    for content in contents:
        if joined and joined[-1].get("role") == content.get("role"):
            joined[-1] = {**joined[-1], "parts": [*joined[-1]["parts"], *content["parts"]]}
        else:
            joined.append(content)
    return joined


def parity(firsts: Mapping[tuple[str, str], Sequence[Any]], arm: str) -> dict[str, dict[str, str]]:
    """Per task, whether `arm`'s first request holds the baseline's contents: `identical`,
    `regrouped` (the same parts under different content boundaries, which is not a pass),
    `differs` (with where), `unstable` (an arm's runs disagree among themselves) or
    `missing`."""
    result = {}
    for task in sorted({task for _arm, task in firsts}):
        ours, theirs = firsts.get((arm, task), []), firsts.get((BASELINE, task), [])
        if not ours or not theirs:
            result[task] = {"status": "missing", "where": ""}
        elif len(ours) > 1 or len(theirs) > 1:
            result[task] = {"status": "unstable", "where": f"{len(ours)} and {len(theirs)} forms"}
        elif (where := first_difference(ours[0], theirs[0])) is None:
            result[task] = {"status": "identical", "where": ""}
        elif first_difference(merged(ours[0]), merged(theirs[0])) is None:
            result[task] = {"status": "regrouped", "where": where}
        else:
            result[task] = {"status": "differs", "where": where}
    return result


def task_passes(
    records: Iterable[Mapping[str, Any]], tasks: Sequence[Task], arm: str
) -> dict[str, bool]:
    """Per task, whether more than half of `arm`'s counted runs pass: kind selection, or
    the expected ending for a refusal task. Errored runs and runs that missed their
    recording are left out, as in the summary."""
    by_id = {task.id: task for task in tasks}
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for (name, task, _rep), record in latest(records).items():
        if name != arm or task not in by_id or left_out_reason(record):
            continue
        run = reduce_run(record, by_id[task])
        outcomes[task].append(run.expected_ending if run.refusal_task else run.kind_pass)
    return {task: sum(found) / len(found) > 0.5 for task, found in outcomes.items()}


def falls(records: Sequence[Mapping[str, Any]], tasks: Sequence[Task], arm: str) -> list[str]:
    """Tasks the baseline passes in a majority of runs and `arm` does not."""
    base, ours = task_passes(records, tasks, BASELINE), task_passes(records, tasks, arm)
    return sorted(
        task for task, passed in base.items() if passed and task in ours and not ours[task]
    )


# --- The report ------------------------------------------------------------------------


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def _s(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f} s"


def _w2_verdict(arm: str, ours: Mapping[str, int], base: Mapping[str, int]) -> Verdict:
    ok = ours["violations"] == 0 and ours["cached"] > 0
    return Verdict(
        "W2 cache requests carry no prefix",
        arm,
        ok if ours["cached"] else None,
        f"{ours['violations']} violations in {ours['cached']} cached of {ours['requests']} requests",
        f"{base['violations']} in {base['cached']} of {base['requests']}",
    )


def _w3_verdict(arm: str, ours: Mapping[str, int], base: Mapping[str, int]) -> Verdict:
    ok = ours["lost"] == 0 and ours["thought_parts"] == 0
    return Verdict(
        "W3 signatures kept, thought text not replayed",
        arm,
        (ok if ours["signed"] else None),
        f"{ours['lost']} signatures lost of {ours['signed']} signed originals replayed "
        f"({ours['changed']} changed, {ours['added']} added, {ours['unmatched']} unmatched), "
        f"{ours['thought_parts']} thought parts replayed",
        f"{base['lost']} of {base['signed']} ({base['changed']} changed, {base['added']} added, "
        f"{base['unmatched']} unmatched), {base['thought_parts']} thought parts",
    )


def _w4_verdict(arm: str, ours: Mapping[str, Any], base: Mapping[str, Any]) -> Verdict:
    gap = ours["share"] - base["share"]
    return Verdict(
        "W4 cached share, later calls",
        arm,
        None if math.isnan(gap) else gap >= -CACHED_SHARE_POINTS,
        f"{_pct(ours['share'])} of {ours['prompt']:,} prompt tokens over {ours['calls']} calls"
        + ("" if arm == BASELINE else f" ({gap:+.1%} against the baseline)"),
        f"{_pct(base['share'])} of {base['prompt']:,} over {base['calls']} calls",
    )


def _w5_verdict(arm: str, ours: Mapping[str, Any], base: Mapping[str, Any]) -> Verdict:
    ratio = ours["p50"] / base["p50"] if base["p50"] else float("nan")
    return Verdict(
        "W5 seconds to first chunk",
        arm,
        None if math.isnan(ratio) else ratio <= FIRST_CHUNK_RATIO,
        f"p50 {_s(ours['p50'])}"
        + ("" if arm == BASELINE else f" ({ratio:.2f} times)")
        + f", p90 {_s(ours['p90'])}, first-call p50 {_s(ours['first_call_p50'])}, "
        f"{ours['streams']} streams",
        f"p50 {_s(base['p50'])}, p90 {_s(base['p90'])}, first-call p50 "
        f"{_s(base['first_call_p50'])}, {base['streams']} streams",
    )


def _parity_verdict(arm: str, found: Mapping[str, Mapping[str, str]]) -> Verdict:
    counts = Counter(v["status"] for v in found.values())
    return Verdict(
        "Contents parity, first request",
        arm,
        counts["identical"] == len(found) if found else None,
        ", ".join(f"{n} {status}" for status, n in counts.most_common())
        + f" of {len(found)} tasks",
        "n/a",
    )


def _share(passed: Mapping[str, bool]) -> str:
    return f"{sum(passed.values())}/{len(passed)}"


def _o1(
    records: Sequence[Mapping[str, Any]], tasks: Sequence[Task], arm: str
) -> tuple[Verdict, str]:
    report, machine = build_report(records, tasks, arm=arm, baseline=BASELINE)
    fired = [t["trigger"] for t in machine.get("triggers", []) if t["fired"]]
    fell = falls(records, tasks, arm)
    numbers = machine["arms"]
    paired = machine.get("paired", {})
    first, kind = paired.get("First proposal accepted"), paired.get("Kind pass")

    def rate(name: str) -> str:
        n = numbers[name]
        return (
            f"first proposal {n['first_ok']}/{n['scored_runs']}, kind pass {n['kind_pass']}/"
            f"{n['scored_runs']} ({n['kind_pass_tasks']}/{n['kind_tasks']} tasks)"
        )

    def pair(result: Mapping[str, Any] | None) -> str:
        return "n/a" if result is None else f"{result['better']} better, {result['worse']} worse"

    verdict = Verdict(
        "O1 outcomes against the baseline",
        arm,
        None if not paired else (not fired and not fell),
        f"{rate(arm)}; paired first proposal {pair(first)}, kind {pair(kind)}; triggers fired: "
        f"{', '.join(fired) or 'none'}; {len(fell)} tasks fall from passing"
        + (f" ({', '.join(fell)})" if fell else ""),
        rate(BASELINE),
    )
    return verdict, report


def build_results(
    records: Sequence[Mapping[str, Any]],
    tap: Sequence[Call],
    tasks: Sequence[Task],
    *,
    parity_json: Path | None = None,
    summaries: Path | None = None,
) -> str:
    """The RESULTS.md text. `summaries` is a directory to write each arm's summary to."""
    entries = chosen_entries(records, tap)
    if BASELINE not in entries:
        raise SystemExit("the records hold no baseline runs to set the arms against")
    arms = [arm for arm in entries if arm != BASELINE]
    firsts = first_requests(records, tap)
    counts = Counter(r["status"] for r in latest(records).values())
    tap_errors = sum(len(c["tap_errors"]) for c in tap) + sum(
        c["kind"] == "tap_failure" for c in tap
    )
    base = {"w2": w2(entries[BASELINE]), "w3": w3(entries[BASELINE]), "w4": w4(entries[BASELINE])}
    base["w5"] = w5(entries[BASELINE])
    verdicts: list[Verdict] = []
    details: list[str] = []
    for arm in arms:
        ours = entries[arm]
        found = parity(firsts, arm)
        o1, report = _o1(records, tasks, arm)
        if summaries is not None:
            summaries.mkdir(parents=True, exist_ok=True)
            (summaries / f"summary-{arm}.md").write_text(report, encoding="utf-8")
        verdicts += [
            _w2_verdict(arm, w2(ours), base["w2"]),
            _w3_verdict(arm, w3(ours), base["w3"]),
            _w4_verdict(arm, w4(ours), base["w4"]),
            _w5_verdict(arm, w5(ours), base["w5"]),
            _parity_verdict(arm, found),
            o1,
        ]
        details.append(
            f"### Contents parity, {arm}\n\n"
            + table(
                ["Task", "Status", "First difference"],
                [
                    [f"`{task}`", v["status"], f"`{v['where']}`" if v["where"] else ""]
                    for task, v in found.items()
                    if v["status"] != "identical"
                ]
                or [["all tasks", "identical", ""]],
            )
        )
    sections = [
        "# Gemini comparison: results",
        f"{len(latest(records))} runs over {len({r['task'] for r in records})} tasks "
        f"({', '.join(f'{arm} {len(e)}' for arm, e in entries.items())}), "
        f"{counts['error']} of them errors; {sum(c['kind'] == 'generate' for c in tap)} model "
        f"calls and {sum(c['kind'] == 'cache_create' for c in tap)} cache creates tapped, "
        f"{tap_errors} tap bookkeeping errors.",
        "## Baseline alone",
        table(
            ["Line", "Numbers"],
            [
                [v.line, v.numbers]
                for v in (
                    _w2_verdict(BASELINE, base["w2"], base["w2"]),
                    _w3_verdict(BASELINE, base["w3"], base["w3"]),
                    _w4_verdict(BASELINE, base["w4"], base["w4"]),
                    _w5_verdict(BASELINE, base["w5"], base["w5"]),
                )
            ],
        ),
        "## Pass lines",
        table(
            ["Line", "Arm", "Result", "Arm numbers", "Baseline numbers"],
            [[v.line, v.arm, v.word, v.numbers, v.baseline] for v in verdicts],
        ),
        "W2 passes only when at least one request carried a cache. W3 passes only when at least "
        "one replayed call had a signed original. W4 passes when the arm's share is not more "
        "than 5 points below the baseline's. O1 passes when no switch trigger fires and no task "
        "the baseline passes in a majority of runs falls below that.",
        "## W1",
        _w1(parity_json),
        "## Detail",
        *details,
    ]
    return "\n\n".join(sections) + "\n"


def _w1(path: Path | None) -> str:
    if path is None or not path.exists():
        return "`parity.json` is not present, so there is no W1 figure to quote."
    return (
        "From `parity.json`:\n\n```json\n"
        + json.dumps(json.loads(path.read_text()), indent=1)
        + "\n```"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="the driver's run directory")
    parser.add_argument(
        "--parity", type=Path, default=Path(__file__).with_name("parity.json"), help="W1's file"
    )
    args = parser.parse_args(argv)
    text = build_results(
        read_records([args.out / "records.jsonl"]),
        read_tap(args.out / "tap.jsonl"),
        load_tasks(),
        parity_json=args.parity,
        summaries=args.out,
    )
    (args.out / "RESULTS.md").write_text(text, encoding="utf-8")
    print(args.out / "RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
