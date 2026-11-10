"""Run full Studio turns on live Gemini and the dev data services, on two planner arms.

Each question runs once on each arm: dev's ADK planner node (`adk`) and the hardened
Pydantic AI planner node (`pai`, `node-hard/hardened.py`). The harness app is dev's own
`harness.server.create_app`, built in process for every turn; the `pai` arm replaces only the
planner `harness.server.build_planner` returns. The orchestrator, translator, `execute` node
and reply are dev's code on both arms.

The driver plays the user: it answers every elicitation with the recommended option (else the
first), approves every proposed plan, and reads each stream to its end. A record is appended to
`records.jsonl` when a turn ends; a rerun with the same `--out` skips the (arm, question) pairs
that have one, and `--stop-after-s` starts no new turn after that many seconds.

The planner's progress reports ("called <tool>") become running status frames on both arms
through `show_planner_progress`, as in `node/test_node.py`, because dev's translator draws no
frame for them.

    uv run --env-file <env> --package harness python research/pai-planner-live/live-turns/driver.py \\
        --out <dir> --stop-after-s 480
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "services" / "harness").is_dir())
PAI = ROOT / "research" / "pai-planner-live"
for path in (
    HERE,
    PAI / "node-hard",
    PAI / "node",
    PAI / "agent",
    ROOT / "evals" / "planner_entry",
    ROOT / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

APIS_URL = "http://localhost:8083"
PORT = 18080


def configure_env(*, apis_url: str) -> None:
    """The environment a local harness needs against the dev data services, and no trace
    export: the driver measures what it needs itself."""
    os.environ["DFO_APIS_URL"] = apis_url
    os.environ["PORT"] = str(PORT)
    os.environ["PRESENTATION_PUBLIC_URL"] = f"http://localhost:{PORT}/mcp/presentation"
    os.environ["SEED_FIXTURE_FRAMES"] = "true"
    for name in [n for n in os.environ if n.startswith("LANGFUSE_")]:
        del os.environ[name]


configure_env(apis_url=os.environ.get("TURNS_APIS_URL", APIS_URL))

import hardened  # noqa: E402
import harness.server as harness_server  # noqa: E402
import httpx2  # noqa: E402
import tap_agents  # noqa: E402
import uvicorn  # noqa: E402
from harness.planner_seam import PlannerSeam  # noqa: E402
from harness.settings import Settings  # noqa: E402
from harness.studio_translate import _WORKING_ON_PLAN, _node_name, _Translator  # noqa: E402
from harness.wire import (  # noqa: E402
    STUDIO_FRAME_ADAPTER,
    ActivityStep,
    ElicitationQuestion,
    ElicitationRequest,
    InterruptDecision,
    InterruptResponse,
    OptionAnswer,
    PlanProposed,
    SkippedAnswer,
    StepStatus,
    TurnRequest,
)
from planner.agent import AGENT_NAME  # noqa: E402

ARMS = ("adk", "pai")
MAX_PAUSES = 8
Key = tuple[str, int]


def question_order(index: int) -> tuple[str, str]:
    """The arms of one question in run order; it alternates question by question."""
    return ARMS if index % 2 == 0 else ARMS[::-1]


def show_planner_progress() -> None:
    """Render each progress report of the planner as a running status frame, on both arms.
    Dev's translator draws no frame for the seam's late reports
    (`harness.studio_translate._Translator._plan_path_frames`)."""
    original = _Translator._plan_path_frames
    if getattr(original, "shows_progress", False):
        return

    def plan_path_frames(self: _Translator, event: Any) -> list[Any]:
        progress = (event.custom_metadata or {}).get("progress")
        if _node_name(event) not in {"planner", AGENT_NAME} or not isinstance(progress, dict):
            return original(self, event)
        frames: list[Any] = self._open_status(_WORKING_ON_PLAN)
        assert self._status is not None
        step_id, label, _ = self._status
        frames.append(
            ActivityStep(
                event="activity.step",
                seq=self._conversation.next_seq(),
                step_id=step_id,
                label=label,
                status=StepStatus.RUNNING,
                detail=f"called {progress['tools'][0]}",
            )
        )
        return frames

    plan_path_frames.shows_progress = True  # ty: ignore[unresolved-attribute] -- a marker
    _Translator._plan_path_frames = plan_path_frames  # type: ignore[method-assign]


def use_pai_planner() -> None:
    """Make `create_app` build the hardened Pydantic AI planner where it builds dev's."""
    plain = harness_server.build_planner

    def build_planner(**kwargs: Any) -> Any:
        seam: PlannerSeam = plain(**kwargs)
        return hardened.hard_planner(
            seam,
            client=None,
            surface=kwargs["surface"],
            translate=seam._translate,
            prefix_cache=kwargs.get("prefix_cache"),
        )

    harness_server.build_planner = build_planner


class ErrorLogs(logging.Handler):
    """The ERROR records the harness and planner log during a turn."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.name}: {record.getMessage()}"[:300])


def auto_answer(question: ElicitationQuestion) -> tuple[OptionAnswer | SkippedAnswer, str] | None:
    """The question's recommended option, else its first; with no options, left to the
    agent when the question allows it. Also the label that was chosen."""
    if question.options:
        chosen = next((o for o in question.options if o.recommended), question.options[0])
        how = "recommended" if chosen.recommended else "first"
        return OptionAnswer(kind="option", option_id=chosen.id), f"{how}: {chosen.label}"
    if question.skippable:
        return SkippedAnswer(kind="skipped"), "skipped"
    return None


async def frames_of(response: httpx2.Response) -> AsyncIterator[dict[str, Any]]:
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            parsed = json.loads(line.removeprefix("data: "))
            STUDIO_FRAME_ADAPTER.validate_python(parsed)
            yield parsed


def frame_summary(frame: dict[str, Any], at: float) -> dict[str, Any]:
    """The part of a frame the analysis reads, with its time in seconds from the turn's start."""
    keep = {"event", "step_id", "label", "status", "detail", "tool", "text", "type", "title"}
    summary = {k: v for k, v in frame.items() if k in keep}
    if frame["event"] == "error":
        summary["error"] = frame["error"]
    if frame["event"] == "plan.proposed":
        summary["steps"] = len(frame.get("steps") or [])
    if frame["event"] == "elicitation.request":
        summary["questions"] = [q["question"] for q in frame["questions"]]
    for key in ("text", "detail"):
        if isinstance(summary.get(key), str):
            summary[key] = summary[key][:600]
    return {"at": round(at, 3), **summary}


async def drive(client: httpx2.AsyncClient, question: str, token: str, *, limit_s: float) -> dict:
    """One turn: the frames in order with times, the asks and how each was answered, the plans
    approved. Ends on `done`, an error frame, a stream with no pause or end, or `limit_s`."""
    conversation_id = f"turns_{uuid.uuid4().hex[:12]}"
    headers = {"Authorization": f"Bearer {token}"}
    body: dict[str, Any] = TurnRequest(
        kind="turn.request", schema_version=1, conversation_id=conversation_id, message=question
    ).model_dump(mode="json", exclude_none=True)
    path = "/api/studio/turn"
    started = time.monotonic()
    frames: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    approvals = 0
    ending = "no_end"
    http_error: str | None = None
    for _ in range(MAX_PAUSES + 1):
        pause: dict[str, Any] | None = None
        async with client.stream("POST", path, json=body, headers=headers) as response:
            if response.status_code != 200:
                await response.aread()
                http_error = f"{path} HTTP {response.status_code}: {response.text[:300]}"
                ending = "http_error"
                break
            async for frame in frames_of(response):
                frames.append(frame_summary(frame, time.monotonic() - started))
                if frame["event"] in ("plan.proposed", "elicitation.request"):
                    pause = frame
                elif frame["event"] in ("done", "error"):
                    ending = frame["event"]
        if ending in ("done", "error", "http_error"):
            break
        if pause is None:
            ending = "stream_ended_without_pause"
            break
        if time.monotonic() - started > limit_s:
            ending = "time_limit"
            break
        if pause["event"] == "plan.proposed":
            plan = PlanProposed.model_validate(pause)
            approvals += 1
            resume = InterruptResponse(
                kind="interrupt.response",
                schema_version=1,
                conversation_id=conversation_id,
                interrupt_id=plan.interrupt_id,
                decision=InterruptDecision.APPROVE,
                plan_revision=plan.revision,
            )
        else:
            request = ElicitationRequest.model_validate(pause)
            answers: dict[str, Any] = {}
            record: dict[str, Any] = {"at": frames[-1]["at"], "questions": []}
            decision = InterruptDecision.ANSWER
            for q in request.questions:
                picked = auto_answer(q)
                record["questions"].append(
                    {"question": q.question, "answer": picked[1] if picked else "dismissed"}
                )
                if picked is None:
                    decision = InterruptDecision.SKIP
                    answers = {}
                    break
                answers[q.id] = picked[0]
            asks.append(record)
            resume = InterruptResponse(
                kind="interrupt.response",
                schema_version=1,
                conversation_id=conversation_id,
                interrupt_id=request.interrupt_id,
                decision=decision,
                answers=answers or None,
            )
        path = "/api/studio/interrupt"
        body = resume.model_dump(mode="json", exclude_none=True)
    else:
        ending = "pause_limit"
    return {
        "ending": ending,
        "http_error": http_error,
        "frames": frames,
        "asks": asks,
        "approvals": approvals,
        "wall_s": round(time.monotonic() - started, 3),
    }


async def run_turn(arm: str, question: str, *, token: str, limit_s: float) -> dict[str, Any]:
    """One turn on `arm` against a freshly built harness app."""
    show_planner_progress()
    errors = ErrorLogs()
    logging.getLogger().addHandler(errors)
    result: dict[str, Any]
    with tap_agents.collecting() as calls:
        try:
            app = harness_server.create_app(settings=Settings())
            async with (
                serving(app),
                httpx2.AsyncClient(
                    base_url=f"http://127.0.0.1:{PORT}", timeout=limit_s + 60
                ) as client,
            ):
                async with asyncio.timeout(limit_s + 30):
                    result = await drive(client, question, token, limit_s=limit_s)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            result = {"ending": "driver_exception", "frames": [], "asks": [], "approvals": 0}
            result["exception"] = f"{type(exc).__name__}: {exc}"[:1500]
        finally:
            logging.getLogger().removeHandler(errors)
    return {**result, "model_calls": calls, "error_logs": errors.lines[:20]}


@contextlib.asynccontextmanager
async def serving(app: Any) -> AsyncIterator[None]:
    """`app` served on `PORT`: the harness calls its own mounted MCP servers over HTTP."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="critical", lifespan="on")
    )
    task = asyncio.ensure_future(server.serve())
    try:
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError("the harness server stopped before it started")
            await asyncio.sleep(0.1)
        yield
    finally:
        server.should_exit = True
        await task


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_questions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_all(args: argparse.Namespace, questions: Sequence[str]) -> int:
    """Run every (arm, question) without a record. Returns how many were left."""
    records_path: Path = args.out / "records.jsonl"
    done = {(r["arm"], r["question_index"]) for r in read_jsonl(records_path)}
    pending = [
        (arm, i)
        for i in range(len(questions))
        for arm in question_order(i)
        if (arm, i) not in done and (args.only is None or i in args.only)
    ]
    print(f"{len(done)} done, {len(pending)} to do", file=sys.stderr)
    token = Settings().studio_api_token or "turns-driver"
    deadline = None if args.stop_after_s is None else time.monotonic() + args.stop_after_s
    finished = 0
    for arm, index in pending:
        if deadline is not None and time.monotonic() > deadline:
            break
        if arm == "pai":
            use_pai_planner()
        record = await run_turn(arm, questions[index], token=token, limit_s=args.turn_limit_s)
        if arm == "pai":
            harness_server.build_planner = _PLAIN_BUILD_PLANNER
        record |= {"arm": arm, "question_index": index, "question": questions[index]}
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        finished += 1
        print(
            f"[{finished}/{len(pending)}] {arm} q{index} {record['ending']} "
            f"{record.get('wall_s', '-')}s"
            f"calls={len(record['model_calls'])}"
            + (f" EXC {record['exception'][:140]}" if record.get("exception") else ""),
            file=sys.stderr,
        )
    return len(pending) - finished


_PLAIN_BUILD_PLANNER = harness_server.build_planner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="run directory; reuse to resume")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.txt")
    parser.add_argument(
        "--only",
        type=lambda s: {int(i) for i in s.split(",")},
        help="comma-separated question indexes (from 0); default every question",
    )
    parser.add_argument("--turn-limit-s", type=float, default=420.0, help="seconds per turn")
    parser.add_argument("--stop-after-s", type=float, help="start no new turn after this long")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions)
    logging.disable(logging.WARNING)
    with contextlib.ExitStack() as stack:
        stack.enter_context(tap_agents.installed())
        left = asyncio.run(run_all(args, questions))
    if left:
        print(f"{left} turns not started; rerun with --out {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
