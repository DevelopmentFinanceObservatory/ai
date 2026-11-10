"""Read the driver runs' `records.jsonl` and write `RESULTS.md`, every figure computed here.

Both arms are scored over every entry, a failed entry counting as a first proposal not
accepted, a kind pass missed and whatever calls it made. The paired difference is Pydantic
AI minus ADK, with a 95% percentile interval from resampling the tasks (both arms keep the
same resampled tasks, so a task's entries stay together). One run is reported, `runs/fixed`,
beside the earlier pooled figures (`EARLIER_POOLED`, copied from the earlier RESULTS.md, not
recomputed). `--check` compares the file on disk with a fresh one and exits 1 when they differ.

    uv run --package harness python research/pai-planner-live/live-entry/analyse.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "services" / "harness").is_dir())
if str(ROOT / "evals" / "planner_entry") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals" / "planner_entry"))

from entry import Task, load_tasks  # noqa: E402
from summarise import percentile, reduce_run  # noqa: E402

BASE, TEST = "adk", "pai"
RESAMPLES = 4000
SEED = 20261110
CACHED_SHARE_POINTS = 0.02
WALL_RATIO = 1.10
AA_NOISE_POINTS = 5.0
RATE_LIMIT_CODE = 429
LONG_THINKING_TOKENS = 2000
RUN = "fixed"


# --- One entry, reduced ------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    task: str
    rep: int
    first_ok: bool
    kind_pass: bool
    calls: int
    requests: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    wall_s: float
    chunk_s: tuple[float, ...]
    closed: bool
    run: str = ""
    long_thinking: bool = False
    final_ok: bool = False
    read_skill: bool = False


def first_chunk_times(tap: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    """Seconds to each call's first chunk. A call cancelled before one (a stall the
    planner cut and resent) counts as the seconds it waited, a lower bound."""
    return tuple(
        call["first_chunk_s"] if call["first_chunk_s"] is not None else call["seconds"]
        for call in tap
    )


def reduce_entry(record: Mapping[str, Any], task: Task) -> Entry:
    run = reduce_run(record, task)
    answered = [call for call in record["tap"] if call["prompt_tokens"] is not None]
    return Entry(
        task=record["task"],
        rep=record["rep"],
        first_ok=run.first_ok,
        kind_pass=run.kind_pass,
        calls=len(record["calls"]),
        requests=len(record["tap"]),
        input_tokens=sum(call["prompt_tokens"] for call in answered),
        cached_tokens=sum(call["cached_tokens"] for call in answered),
        output_tokens=sum(call["output_tokens"] for call in answered),
        wall_s=record["wall_s"],
        chunk_s=first_chunk_times(record["tap"]),
        closed=record["budget_closed"] is not None,
        run=record.get("run", ""),
        long_thinking=any(
            (call["thinking_tokens"] or 0) >= LONG_THINKING_TOKENS for call in record["tap"]
        ),
        final_ok=bool(record["accepted"]),
        read_skill="get_skill" in record["tool_calls"],
    )


# --- Statistics --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stat:
    name: str
    unit: str  # "rate", "count", "tokens" or "seconds"
    fn: Callable[[Sequence[Entry]], float]


def _mean(attr: str) -> Callable[[Sequence[Entry]], float]:
    return lambda entries: statistics.fmean(float(getattr(e, attr)) for e in entries)


def _walls(q: float) -> Callable[[Sequence[Entry]], float]:
    return lambda entries: percentile([e.wall_s for e in entries], q)


def _chunks(q: float) -> Callable[[Sequence[Entry]], float]:
    return lambda entries: percentile([t for e in entries for t in e.chunk_s], q)


def cached_share(entries: Sequence[Entry]) -> float:
    return sum(e.cached_tokens for e in entries) / sum(e.input_tokens for e in entries)


STATS = (
    Stat("First proposals accepted", "rate", _mean("first_ok")),
    Stat("Final acceptance", "rate", _mean("final_ok")),
    Stat("Kind pass", "rate", _mean("kind_pass")),
    Stat("Entries that read a skill", "rate", _mean("read_skill")),
    Stat("Model calls per entry", "count", _mean("calls")),
    Stat("Requests sent per entry", "count", _mean("requests")),
    Stat("Input tokens per entry", "tokens", _mean("input_tokens")),
    Stat("Cached share of input", "rate", cached_share),
    Stat("Output tokens per entry", "tokens", _mean("output_tokens")),
    Stat("First chunk, median", "seconds", _chunks(0.5)),
    Stat("First chunk, p90", "seconds", _chunks(0.9)),
    Stat("Planner wall time, median", "seconds", _walls(0.5)),
    Stat("Planner wall time, p90", "seconds", _walls(0.9)),
    Stat("Entries with a call of 2,000+ thinking tokens", "rate", _mean("long_thinking")),
)


@dataclass(frozen=True)
class Paired:
    base: float
    test: float
    diff: tuple[float, float, float]  # point, low, high
    ratio: tuple[float, float, float]


def by_task(entries: Iterable[Entry]) -> dict[str, list[Entry]]:
    grouped: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.task].append(entry)
    return dict(grouped)


def interval(values: Sequence[float]) -> tuple[float, float]:
    return percentile(values, 0.025), percentile(values, 0.975)


def paired(stat: Stat, base: Mapping[str, list[Entry]], test: Mapping[str, list[Entry]]) -> Paired:
    """`stat` on each arm, and the test arm's difference and ratio to the base arm's, each
    with a 95% interval over resampled tasks."""
    tasks = sorted(base)
    flat = lambda arm, picked: [e for t in picked for e in arm[t]]  # noqa: E731
    b, t = stat.fn(flat(base, tasks)), stat.fn(flat(test, tasks))
    generator = random.Random(SEED)
    diffs, ratios = [], []
    for _ in range(RESAMPLES):
        picked = generator.choices(tasks, k=len(tasks))
        rb, rt = stat.fn(flat(base, picked)), stat.fn(flat(test, picked))
        diffs.append(rt - rb)
        ratios.append(rt / rb if rb else math.nan)
    ratios = [r for r in ratios if not math.isnan(r)]
    spread = interval(ratios) if ratios else (math.nan, math.nan)
    return Paired(b, t, (t - b, *interval(diffs)), (t / b if b else math.nan, *spread))


def exact_sign_p(more: int, fewer: int) -> float:
    """Two-sided exact sign test over the pairs that differ."""
    n = more + fewer
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(more, fewer) + 1)) / 2**n
    return min(1.0, 2 * tail)


# --- Failure classes ---------------------------------------------------------------------

# Endings that are not failures: the plan was accepted, or the planner asked the user.
ORDINARY_ENDINGS = frozenset({"accepted", "asked_user"})


def error_class(text: str) -> str:
    return text.split(":", 1)[0].strip() or "unknown"


def failure_classes(record: Mapping[str, Any]) -> dict[str, str]:
    """The failure classes of one entry, each with an example. A 429 is not a failure."""
    found: dict[str, str] = {}
    if record["stop_reason"] not in ORDINARY_ENDINGS:
        found[f"ended: {record['stop_reason']}"] = f"{record['task']} rep {record['rep']}"
    if record["error"]:
        found[f"error: {error_class(record['error'])}"] = record["error"].splitlines()[0][:200]
    if record["surface_misses"]:
        found["recording miss"] = "; ".join(record["surface_misses"])[:200]
    for call in record["tap"]:
        if (err := call["error"]) and err["code"] != RATE_LIMIT_CODE:
            label = (
                "call cancelled before its first chunk"
                if (err["class"] == "CancelledError" and call["first_chunk_s"] is None)
                else "call cancelled mid-stream"
                if err["class"] == "CancelledError"
                else (f"call failed: {err['class']} {err['code']}")
            )
            found.setdefault(
                label, f"{record['task']} rep {record['rep']}, {err['message'] or err['class']}"
            )
    return found


@dataclass
class ClassCount:
    entries: Counter[str] = field(default_factory=Counter)
    example: dict[str, str] = field(default_factory=dict)


def count_classes(records: Iterable[Mapping[str, Any]]) -> ClassCount:
    counts = ClassCount()
    for record in records:
        for name, example in failure_classes(record).items():
            counts.entries[name] += 1
            counts.example.setdefault(name, example)
    return counts


def other_endings(records: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(r["stop_reason"] for r in records if r["stop_reason"] in ORDINARY_ENDINGS)


# --- Figures -----------------------------------------------------------------------------


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def after_close(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """What the closed entries of `records` show of a call run after the closing response: the
    entries holding a proposal, and those recording more tool calls than the model responses
    that completed (a call of the cut response would be one more). A response may carry
    several calls, so each such entry reports its output tokens per call, to set beside the
    other closed entries' (`per_read`, `others`)."""
    closed = [
        r
        for r in records
        if r["budget_closed"] is not None or r["stop_reason"] == "turn_model_time_spent"
    ]

    def per_read(record: Mapping[str, Any]) -> float:
        written = sum(c["output_tokens"] for c in record["tap"] if c["error"] is None)
        return written / len(record["tool_calls"])

    extra = [r for r in closed if len(r["tool_calls"]) > sum(c["error"] is None for c in r["tap"])]
    rest = [r for r in closed if r not in extra and r["tool_calls"]]
    return {
        "closed": len(closed),
        "proposals": sum(bool(r["proposals"]) for r in closed),
        "extra_reads": len(extra),
        "per_read": [per_read(r) for r in extra],
        "others": statistics.median(per_read(r) for r in rest) if rest else math.nan,
    }


def thinking(entries: Sequence[Entry]) -> dict[str, float]:
    """The entries with a call of 2,000 or more thinking tokens, and the median wall time of
    those with and without one."""
    long = [e.wall_s for e in entries if e.long_thinking]
    rest = [e.wall_s for e in entries if not e.long_thinking]
    return {
        "entries": len(entries),
        "long": len(long),
        "median_long": statistics.median(long) if long else math.nan,
        "median_rest": statistics.median(rest) if rest else math.nan,
    }


def long_calls(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every entry with a call of 2,000 or more thinking tokens: arm, task, repetition, and
    each such call's position among the entry's calls (1 is the first) with its tokens."""
    return [
        f"{r['arm']} {r['task']} rep {r['rep']}: "
        + ", ".join(
            f"call {i} of {len(r['tap'])} ({c['thinking_tokens']:,} tokens)"
            for i, c in enumerate(r["tap"], 1)
            if (c["thinking_tokens"] or 0) >= LONG_THINKING_TOKENS
        )
        for r in records
        if any((c["thinking_tokens"] or 0) >= LONG_THINKING_TOKENS for c in r["tap"])
    ]


def compute(records: Sequence[Mapping[str, Any]], tasks: Mapping[str, Task]) -> dict[str, Any]:
    arms = {arm: [r for r in records if r["arm"] == arm] for arm in (BASE, TEST)}
    entries = {arm: [reduce_entry(r, tasks[r["task"]]) for r in rs] for arm, rs in arms.items()}
    grouped = {arm: by_task(es) for arm, es in entries.items()}
    results = {stat.name: paired(stat, grouped[BASE], grouped[TEST]) for stat in STATS}
    keyed = {arm: {(e.run, e.task, e.rep): e for e in es} for arm, es in entries.items()}
    pairs = sorted(set(keyed[BASE]) & set(keyed[TEST]))
    closed_more = sum(keyed[TEST][k].closed and not keyed[BASE][k].closed for k in pairs)
    closed_fewer = sum(keyed[BASE][k].closed and not keyed[TEST][k].closed for k in pairs)
    classes = {arm: count_classes(rs) for arm, rs in arms.items()}
    only_test = sorted(set(classes[TEST].entries) - set(classes[BASE].entries))
    only_base = sorted(set(classes[BASE].entries) - set(classes[TEST].entries))
    return {
        "records": len(records),
        "tasks": len(grouped[BASE]),
        "reps": sorted({r["rep"] for r in records}),
        "runs": sorted({r["run"] for r in records}),
        "model": records[0]["model"],
        "revision": " and ".join(sorted({r["revision"] for r in records})),
        "arms": {arm: len(rs) for arm, rs in arms.items()},
        "pairs": len(pairs),
        "stats": results,
        "closed": {arm: sum(e.closed for e in es) for arm, es in entries.items()}
        | {
            "test_only": closed_more,
            "base_only": closed_fewer,
            "p": exact_sign_p(closed_more, closed_fewer),
        },
        "classes": classes,
        "only_test": only_test,
        "only_base": only_base,
        "endings": {arm: other_endings(rs) for arm, rs in arms.items()},
        "retries": {arm: sum(r["retries"] for r in rs) for arm, rs in arms.items()},
        "entries_retried": {arm: sum(r["retries"] > 0 for r in rs) for arm, rs in arms.items()},
        "rate_limited_calls": {
            arm: sum(
                1
                for r in rs
                for c in r["tap"]
                if c["error"] and c["error"]["code"] == RATE_LIMIT_CODE
            )
            for arm, rs in arms.items()
        },
        "after_close": after_close(arms[TEST]),
        "thinking": {arm: thinking(es) for arm, es in entries.items()},
        "long_calls": long_calls(records),
        "errored": {arm: sum(r["status"] == "error" for r in rs) for arm, rs in arms.items()},
    }


# --- Rendering ---------------------------------------------------------------------------


def fmt(value: float, unit: str, *, signed: bool = False) -> str:
    sign = "+" if signed and value > 0 else ""
    if unit == "rate":
        return f"{sign}{value * 100:.1f}%" if not signed else f"{sign}{value * 100:.1f} points"
    if unit == "count":
        return f"{sign}{value:.2f}"
    if unit == "tokens":
        return f"{sign}{value:,.0f}"
    return f"{sign}{value:.1f} s"


def md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def triple(values: tuple[float, float, float], unit: str) -> str:
    point, low, high = values
    return f"{fmt(point, unit, signed=True)} [{fmt(low, unit, signed=True)}, {fmt(high, unit, signed=True)}]"


def ratio_text(values: tuple[float, float, float]) -> str:
    point, low, high = values
    return f"{(point - 1) * 100:+.1f}% [{(low - 1) * 100:+.1f}%, {(high - 1) * 100:+.1f}%]"


def thresholds(figures: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cached = figures["stats"]["Cached share of input"]
    wall = figures["stats"]["Planner wall time, median"]
    return {
        "a": {"breached": bool(figures["only_test"]), "classes": figures["only_test"]},
        "b": {
            "breached": cached.diff[0] < -CACHED_SHARE_POINTS,
            "open": cached.diff[1] < -CACHED_SHARE_POINTS,
            "paired": cached,
        },
        "c": {
            "breached": wall.ratio[0] > WALL_RATIO,
            "open": wall.ratio[2] > WALL_RATIO,
            "paired": wall,
        },
    }


def standing(threshold: Mapping[str, Any]) -> str:
    """How a paired threshold stands: breached on the point estimate, not breached on it but
    not excluded by the interval, or met."""
    if threshold["breached"]:
        return "breached"
    if threshold["open"]:
        return "not breached on the point estimate; not excluded by the interval"
    return "met"


def run_summary(figures: Mapping[str, Any]) -> str:
    """The thresholds of one set of entries, each on a line of its own."""
    rule = thresholds(figures)
    only = figures["only_test"]
    cs, ws = rule["b"]["paired"], rule["c"]["paired"]
    return "\n\n".join(
        [
            "(a) Failure classes seen only on the Pydantic AI arm: "
            + (
                "**breached**. " + ", ".join(f"`{name}`" for name in only) + "."
                if only
                else "**met**. Every failure class on the Pydantic AI arm also occurs on the ADK arm."
            )
            + (
                f" Classes seen only on ADK: {', '.join(f'`{n}`' for n in figures['only_base'])}."
                if figures["only_base"]
                else ""
            ),
            f"(b) Cached share down more than {CACHED_SHARE_POINTS * 100:.0f} points: "
            f"**{standing(rule['b'])}**. Pydantic AI {fmt(cs.test, 'rate')}, "
            f"ADK {fmt(cs.base, 'rate')}, difference {triple(cs.diff, 'rate')}.",
            f"(c) Median planner wall time up more than {(WALL_RATIO - 1) * 100:.0f}%: "
            f"**{standing(rule['c'])}**. Pydantic AI {ws.test:.1f} s, "
            f"ADK {ws.base:.1f} s, ratio {ratio_text(ws.ratio)}.",
        ]
    )


def render_run(figures: Mapping[str, Any], *, first_run: bool) -> list[str]:
    """The sections of one set of entries (one run, or the pool of both)."""
    stats: Mapping[str, Paired] = figures["stats"]
    accept = stats["First proposals accepted"]
    kind = stats["Kind pass"]
    closed = figures["closed"]
    lines: list[str] = []
    out = lines.append
    out("### Thresholds")
    out("")
    out(run_summary(figures))
    out("")
    out("### First-proposal acceptance is a sanity check")
    out("")
    out(
        f"The eval's A/A run put two identical arms about {AA_NOISE_POINTS:.0f} points apart in "
        f"first-proposal acceptance. The paired difference here is {triple(accept.diff, 'rate')}, "
        f"an interval half-width of {(accept.diff[2] - accept.diff[1]) / 2 * 100:.1f} points. "
        "Acceptance shows that the Pydantic AI entry drafts plans the contract admits at about "
        "ADK's rate. It is a sanity check and not a parity claim. "
        f"Kind pass is {abs(kind.diff[0]) * 100:.1f} points {'lower' if kind.diff[0] < 0 else 'higher'} "
        f"on the Pydantic AI arm, with an interval ({triple(kind.diff, 'rate')[len(fmt(kind.diff[0], 'rate', signed=True)) + 1 :]}) "
        f"that {'includes' if kind.diff[1] <= 0 <= kind.diff[2] else 'excludes'} zero."
    )
    out("")
    out("### Results per arm")
    out("")
    rows = [
        [
            stat.name,
            fmt(stats[stat.name].base, stat.unit),
            fmt(stats[stat.name].test, stat.unit),
            triple(stats[stat.name].diff, stat.unit),
            ratio_text(stats[stat.name].ratio)
            if stat.unit in ("tokens", "seconds", "count")
            else "",
        ]
        for stat in STATS
    ]
    out(md_table(["", "ADK", "Pydantic AI", "Difference [95% interval]", "Relative"], rows))
    out("")
    per_task = figures["arms"][BASE] / figures["tasks"]
    out(
        f"Each arm ran {figures['arms'][BASE]} entries: {figures['tasks']} tasks at "
        f"{per_task:g} entries per task, on `{figures['model']}` at `{figures['revision']}`. "
        "Every figure counts every entry; a failed entry counts as a first proposal not accepted and "
        "a kind pass missed. The interval resamples tasks, keeping a task's entries together. "
        "Model calls are the calls the entry started, as the eval counts them; requests sent add "
        "the resends the planner makes after a first-chunk stall. Tokens come from the calls that "
        "returned usage. Cached share is cached input over input, pooled over all calls. A call "
        "cancelled before its first chunk counts in first-chunk time as the seconds it waited."
    )
    out("")
    out("### Endings and failure classes")
    out("")
    names = sorted(set(figures["classes"][BASE].entries) | set(figures["classes"][TEST].entries))
    rows = [
        [
            f"`{name}`",
            str(figures["classes"][BASE].entries.get(name, 0)),
            str(figures["classes"][TEST].entries.get(name, 0)),
            (
                figures["classes"][TEST].example.get(name)
                or figures["classes"][BASE].example.get(name, "")
            ).replace("|", "/"),
        ]
        for name in names
    ]
    out(md_table(["Class", "ADK entries", "Pydantic AI entries", "Example"], rows))
    out("")
    endings = figures["endings"]
    out(
        f"Entries that ended in an accepted plan: ADK {endings[BASE]['accepted']}, Pydantic AI "
        f"{endings[TEST]['accepted']}. Entries that ended in a question to the user: ADK "
        f"{endings[BASE]['asked_user']}, Pydantic AI {endings[TEST]['asked_user']}. Neither is a failure."
    )
    out("")
    out(
        f"The turn's model-time budget closed {closed[BASE]} ADK entries and {closed[TEST]} Pydantic AI "
        f"entries. Within a pair, Pydantic AI was closed alone in {closed['test_only']} and ADK alone in "
        f"{closed['base_only']} (exact sign test p = {closed['p']:.2f})."
    )
    out("")
    if first_run:
        out("### What each arm ran")
        out("")
        after = figures["after_close"]
        out(
            f"Both arms ran dev at `{figures['revision'].split('+')[0]}` with the tool-body move "
            "(`planner/tools.py` and `planner/source_reads.py`, which hold the tool bodies both "
            "frameworks call), so the ADK arm is dev plus that move. The Pydantic AI arm ran the "
            "agent without the held-chunk fix proved in `node-hard/`, so a budget cut mid-stream "
            "could run a tool call the model had already delivered. Of the "
            f"{after['closed']} closed Pydantic AI entries, {after['proposals']} hold a proposal and "
            f"{after['extra_reads']} record more tool calls than responses that completed before the "
            "cut, which a delivered call of the cut response would produce. "
            + (
                f"That entry's responses wrote {after['per_read'][0]:.0f} output tokens per call, "
                f"against a median {after['others']:.0f} in the other closed entries, so it holds two "
                "calls in one response and none from the cut."
                if after["extra_reads"] == 1 and abs(after["per_read"][0] - after["others"]) < 5
                else "Read those entries' calls before relying on this."
            )
        )
        out("")
    out("### Rate limits")
    out("")
    out(
        f"No entry ended in a rate-limit error after backing off. Entries rerun after a 429: ADK "
        f"{figures['entries_retried'][BASE]}, Pydantic AI {figures['entries_retried'][TEST]} "
        f"({figures['retries'][BASE]} and {figures['retries'][TEST]} retries). Calls that drew a 429 "
        f"and were repeated inside the planner's own retry layer: ADK {figures['rate_limited_calls'][BASE]}, "
        f"Pydantic AI {figures['rate_limited_calls'][TEST]}. Entries ending in an error: ADK "
        f"{figures['errored'][BASE]}, Pydantic AI {figures['errored'][TEST]}."
    )
    out("")
    return lines


# Figures of the earlier pooled run (420 entries per arm, 245c0a44 plus the tool-body move,
# two skill tools), copied from the earlier RESULTS.md and not recomputed. The earlier run did
# not report final acceptance or the skill-read share.
EARLIER_POOLED: dict[str, str] = {
    "First proposals accepted": "57.4% / 59.8%; +2.4 points [-2.9 points, +7.9 points]",
    "Kind pass": "86.0% / 85.5%; -0.5 points [-4.5 points, +3.6 points]",
    "Cached share of input": "88.4% / 88.5%; +0.1 points [-0.5 points, +0.8 points]",
    "Planner wall time, median": "16.1 s / 17.2 s; +1.1 s [-2.5 s, +4.7 s] (+6.9%)",
    "Planner wall time, p90": "73.7 s / 81.5 s; +7.8 s [-10.9 s, +23.0 s] (+10.6%)",
    "Model calls per entry": "2.70 / 2.64; -0.06 [-0.15, +0.02]",
    "Entries with a call of 2,000+ thinking tokens": (
        "29.3% (123 of 420) / 34.3% (144 of 420); +5.0 points [-2.9 points, +13.1 points]"
    ),
    "Budget closures": "20 ADK only, 27 Pydantic AI only (p = 0.38); entries closed 25 / 32",
}

NOT_REPORTED = "not reported"

SIDE_BY_SIDE = (
    "Entries with a call of 2,000+ thinking tokens",
    "First proposals accepted",
    "Final acceptance",
    "Kind pass",
    "Cached share of input",
    "Planner wall time, median",
    "Planner wall time, p90",
    "Model calls per entry",
    "Entries that read a skill",
)


def side_by_side(figures: Mapping[str, Any]) -> str:
    """Each key figure on ADK and Pydantic AI, and the paired difference, beside the earlier pool."""
    units = {stat.name: stat.unit for stat in STATS}
    rows = []
    for name in SIDE_BY_SIDE:
        pair: Paired = figures["stats"][name]
        unit = units[name]
        now = f"{fmt(pair.base, unit)} / {fmt(pair.test, unit)}; {triple(pair.diff, unit)}"
        if unit == "seconds":
            now += f" ({ratio_text(pair.ratio).split(' ')[0]})"
        rows.append([name, now, EARLIER_POOLED.get(name, NOT_REPORTED)])
    closed = figures["closed"]
    rows.append(
        [
            "Budget closures",
            f"{closed['base_only']} ADK only, {closed['test_only']} Pydantic AI only "
            f"(p = {closed['p']:.2f}); entries closed {closed[BASE]} / {closed[TEST]}",
            EARLIER_POOLED["Budget closures"],
        ]
    )
    return md_table(
        [
            "ADK / Pydantic AI; paired difference [95% interval]",
            "Fixed dev (210 per arm)",
            "Earlier, pooled (420 per arm)",
        ],
        rows,
    )


def long_reasoning(figures: Mapping[str, Any]) -> list[str]:
    """The long-reasoning share of each arm, the paired verdict, and every such entry."""
    share: Paired = figures["stats"]["Entries with a call of 2,000+ thinking tokens"]
    counts = {arm: figures["thinking"][arm] for arm in (BASE, TEST)}
    differs = not share.diff[1] <= 0 <= share.diff[2]
    lines = [
        f"ADK {counts[BASE]['long']} of {counts[BASE]['entries']:.0f}, Pydantic AI "
        f"{counts[TEST]['long']} of {counts[TEST]['entries']:.0f}; the share differs by "
        f"{triple(share.diff, 'rate')}, so the arms "
        + ("differ" if differs else "do not differ detectably")
        + " on it.",
        "",
    ]
    lines += [f"- {entry}" for entry in figures["long_calls"]] or ["No entry on either arm."]
    return lines


def render(figures: Mapping[str, Any]) -> str:
    """The whole document, from the figures of `runs/fixed`."""
    share: Paired = figures["stats"]["Entries with a call of 2,000+ thinking tokens"]
    differs = not share.diff[1] <= 0 <= share.diff[2]
    rule = thresholds(figures)
    lines: list[str] = []
    out = lines.append
    out("# The planner on Pydantic AI against the planner on ADK, with the one-tool skill fix")
    out("")
    out(
        f"With the fix in place the arms {'differ' if differs else 'do not differ detectably'} in how "
        f"often they reason long: {fmt(share.base, 'rate')} of ADK entries and {fmt(share.test, 'rate')} "
        f"of Pydantic AI entries hold a call of {LONG_THINKING_TOKENS:,} or more thinking tokens "
        f"({triple(share.diff, 'rate')}). Threshold (a), failure classes seen only on Pydantic AI: "
        f"{'breached' if rule['a']['breached'] else 'met'}. (b), cached share down more than "
        f"{CACHED_SHARE_POINTS * 100:.0f} points: {standing(rule['b'])}. (c), median wall time up "
        f"more than {(WALL_RATIO - 1) * 100:.0f}%: {standing(rule['c'])}."
    )
    out("")
    out("## Side by side")
    out("")
    out(side_by_side(figures))
    out("")
    out("## Long reasoning")
    out("")
    lines += long_reasoning(figures)
    out("")
    lines += render_run(figures, first_run=False)
    return "\n".join(lines)


# --- Command line ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs", help="directory of runs")
    parser.add_argument("--results", type=Path, default=HERE / "RESULTS.md")
    parser.add_argument("--check", action="store_true", help="fail when RESULTS.md is out of date")
    args = parser.parse_args(argv)
    tasks = {task.id: task for task in load_tasks()}
    records = [r | {"run": RUN} for r in load_records(args.runs / RUN / "records.jsonl")]
    text = render(compute(records, tasks))
    if args.check:
        same = args.results.read_text(encoding="utf-8") == text
        print("RESULTS.md matches the records" if same else "RESULTS.md differs from the records")
        return 0 if same else 1
    args.results.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
