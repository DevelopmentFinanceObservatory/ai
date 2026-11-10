"""Rerun one question on both arms and record what the planner seam saw in every planner run.

The seam (`harness.planner_seam.PlannerSeam`) is the same object on both arms: the Pydantic
AI node calls its `_propose` and its tool callbacks as the ADK node does. Two class-level
wrappers record, for each `_propose` run, every tool answer the seam read
(`_track_tool_call`, which sees each `propose_plan` answer as the model gets it), every
tool error, the tracker's final state, and the run's result: the declared plan, the refusal
it ended on, or neither. Records go to `runs/q18.jsonl`, one per turn, resumable.

    uv run --env-file <env> --with 'pydantic-ai-slim[google]==2.66.0' --package harness \\
        python research/pai-planner-live/live-turns/q18.py --out research/pai-planner-live/live-turns/runs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import driver
import tap_agents
from harness.planner_seam import PlannerSeam

QUESTION_INDEX = 18
CLIP = 1500

_RUNS: list[dict[str, Any]] = []
_CURRENT: list[dict[str, Any]] = []


def clip(value: Any) -> Any:
    """`value` as JSON-safe data, with long text cut."""
    text = json.dumps(value, default=str, ensure_ascii=False)
    return json.loads(text) if len(text) <= CLIP else text[:CLIP] + "..."


def referral(value: Any) -> dict[str, Any] | None:
    return None if value is None else clip(value)


def install() -> None:
    """Wrap `_propose`, `_track_tool_call` and `_on_planner_tool_error` on `PlannerSeam`."""
    propose = PlannerSeam._propose
    track = PlannerSeam._track_tool_call
    on_error = PlannerSeam._on_planner_tool_error

    async def recording_propose(self: PlannerSeam, **kwargs: Any) -> Any:
        run: dict[str, Any] = {"at": time.monotonic(), "calls": [], "errors": [], "tracker": None}
        _RUNS.append(run)
        _CURRENT.append(run)
        try:
            submitted, plan, refused = await propose(self, **kwargs)
        except BaseException as exc:
            run["raised"] = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            _CURRENT.pop()
            tracker = run.pop("tracker")
            run["seconds"] = round(time.monotonic() - run.pop("at"), 3)
            if tracker is not None:
                run["tracker"] = {
                    "events": tracker.events,
                    "ready_to_break": tracker.ready_to_break,
                    "referral_counts": [
                        {"key": clip(k), "count": c} for k, c in tracker.referral_counts.items()
                    ],
                    "last_referral": referral(tracker.last_referral),
                }
        run["submitted"] = submitted is not None
        run["plan"] = plan is not None
        run["refused"] = None if refused is None else clip(refused)
        return submitted, plan, refused

    def recording_track(tracker: Any, on_progress: Any, **kwargs: Any) -> None:
        if _CURRENT:
            _CURRENT[-1]["tracker"] = tracker
            response = kwargs["tool_response"]
            _CURRENT[-1]["calls"].append({"tool": kwargs["tool"].name, "response": clip(response)})
        return track(tracker, on_progress, **kwargs)

    def recording_error(self: PlannerSeam, tool: Any, args: Any, context: Any, error: Any) -> Any:
        if _CURRENT:
            _CURRENT[-1]["errors"].append(
                {"tool": tool.name, "error": f"{type(error).__name__}: {error}"[:400]}
            )
        return on_error(self, tool, args, context, error)

    PlannerSeam._propose = recording_propose  # type: ignore[method-assign]
    PlannerSeam._track_tool_call = staticmethod(recording_track)  # type: ignore[method-assign]
    PlannerSeam._on_planner_tool_error = recording_error  # type: ignore[method-assign]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def order(reps: int) -> list[tuple[str, int]]:
    """(arm, repetition) in run order; the arm that goes first alternates by repetition."""
    return [(arm, rep) for rep in range(reps) for arm in driver.question_order(rep)]


async def run_all(args: argparse.Namespace, question: str) -> int:
    path: Path = args.out / "q18.jsonl"
    done = {(r["arm"], r["rep"]) for r in read_jsonl(path)}
    pending = [key for key in order(args.reps) if key not in done]
    token = driver.Settings().studio_api_token or "turns-driver"
    deadline = None if args.stop_after_s is None else time.monotonic() + args.stop_after_s
    finished = 0
    plain = driver.harness_server.build_planner
    for arm, rep in pending:
        if deadline is not None and time.monotonic() > deadline:
            break
        _RUNS.clear()
        if arm == "pai":
            driver.use_pai_planner()
        try:
            record = await driver.run_turn(arm, question, token=token, limit_s=args.turn_limit_s)
        finally:
            driver.harness_server.build_planner = plain
        record |= {"arm": arm, "rep": rep, "question": question, "planner_runs": list(_RUNS)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        finished += 1
        ends = [
            (r["refused"] or {}).get("code") or ("plan" if r["submitted"] else "asked")
            for r in _RUNS
        ]
        print(
            f"[{finished}/{len(pending)}] {arm} rep{rep} {record['ending']} "
            f"{record.get('wall_s', '-')}s planner runs: {ends}",
            file=sys.stderr,
        )
    return len(pending) - finished


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5, help="turns per arm")
    parser.add_argument("--turn-limit-s", type=float, default=240.0)
    parser.add_argument("--stop-after-s", type=float)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    question = driver.load_questions(driver.HERE / "questions.txt")[QUESTION_INDEX]
    logging.disable(logging.WARNING)
    install()
    with contextlib.ExitStack() as stack:
        stack.enter_context(tap_agents.installed())
        left = asyncio.run(run_all(args, question))
    if left:
        print(f"{left} turns not started; rerun with the same --out", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
