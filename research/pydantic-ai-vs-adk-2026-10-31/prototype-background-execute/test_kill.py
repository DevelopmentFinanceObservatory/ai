"""X2: a process killed inside a step, the run finished by another process.

Process 1 (DBOS executor `p1`) posts the brief, approves the card, and is
killed with SIGKILL once the aggregate step has started. Process 2 (executor
`p2`) posts the same approval again, Studio's retry for a resume whose process
died, with its store's clock 120 s on so process 1's segment lease has lapsed.
Both share the SQLite session database, the run ledger's file and the DBOS
system database.
"""

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).parent


def run_child(
    args: dict[str, Any], *, kill_when: Callable[[dict], bool] | None = None, timeout: float = 90
) -> tuple[list[dict], int, str]:
    """Run `kill_child.py` under pytest; SIGKILL it once `kill_when` accepts a line."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", "kill_child.py", "-q", "-s", "-p", "no:cacheprovider"],
        cwd=HERE,
        env={**os.environ, "BG_CHILD": json.dumps(args), "PYDANTIC_AI_NO_BANNER": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    lines: list[dict] = []
    other: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if not raw.startswith("{"):
                other.append(raw)
                continue
            line = json.loads(raw)
            lines.append(line)
            if kill_when is not None and kill_when(line):
                proc.kill()
                break
        proc.wait()
        return lines, proc.returncode, "".join(other)
    finally:
        watchdog.cancel()
        proc.kill()
        proc.wait()


def _tool_lines(path: Path) -> list[str]:
    return [
        f"p{index} {entry['tool']} {entry['event']}"
        for entry, index in _with_process_index(path)
        if entry["event"] != "progress"
    ]


def _with_process_index(path: Path) -> list[tuple[dict, int]]:
    pids: list[int] = []
    out = []
    for raw in path.read_text().splitlines():
        entry = json.loads(raw)
        if entry["pid"] not in pids:
            pids.append(entry["pid"])
        out.append((entry, pids.index(entry["pid"]) + 1))
    return out


def check_kill(tmp_path: Path, arm: str) -> dict[str, Any]:
    shared = {
        "arm": arm,
        "sessions": str(tmp_path / "sessions.sqlite"),
        "ledger": str(tmp_path / "ledger.json"),
        "dbos": str(tmp_path / "dbos.sqlite"),
        "tool_log": str(tmp_path / "tools.jsonl"),
    }
    first, code, _ = run_child(
        {**shared, "executor": "p1", "steps": ["turn", "approve"], "hold": ["aggregate"]},
        kill_when=lambda line: line.get("tool") == "aggregate" and line.get("event") == "start",
    )
    assert code == -9, f"process 1 was not killed: {code}"
    (card,) = [line for line in first if line["type"] == "card"]
    killed_at = _tool_lines(tmp_path / "tools.jsonl")
    second, code, output = run_child(
        {
            **shared,
            "executor": "p2",
            "steps": ["approve"],
            "interrupt_id": card["interrupt_id"],
            "clock_offset": 120,
            "plans": 0,
        }
    )
    assert code == 0, output[-3000:]
    (end,) = [line for line in second if line["type"] == "end"]
    (frames,) = [line["frames"] for line in second if line["type"] == "frames"]
    (run,) = end["runs"]
    result = {
        "tool_lines_at_kill": killed_at,
        "tool_lines": _tool_lines(tmp_path / "tools.jsonl"),
        "opened_in_process_2": end["opened"],
        "run": {
            "status": run["status"],
            "resumes": run["resumes"],
            "attempts": [(a["step_id"], a["attempt"]) for a in run["attempts"]],
            "delivered": run["delivered"],
        },
        "process_2_frames": frames,
        "hand_back_keys": sorted(end["hand_back"] or {}),
        "hand_back_run": (end["hand_back"] or {}).get("run"),
    }
    print(json.dumps(result))
    assert frames[-1][0] == "done"
    assert run["status"] == "completed"
    assert "frames" in (end["hand_back"] or {}), "plan_done handed back no frames"
    starts = [line for line in result["tool_lines"] if line.endswith("start")]
    assert starts == ["p1 get_data start", "p1 aggregate start", "p2 aggregate start"], (
        f"a finished step ran again: {starts}"
    )
    return result


def test_x2_another_process_finishes_the_run_from_its_last_step(tmp_path: Path) -> None:
    check_kill(tmp_path, "durable")


def test_x2_control_dev_node_reruns_every_step(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="a finished step ran again") as caught:
        check_kill(tmp_path, "dev")
    print(repr(caught.value))
