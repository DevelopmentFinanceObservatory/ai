"""Run `driver.py` as a child process and read its lines as they arrive."""

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DRIVER = Path(__file__).with_name("driver.py")


@dataclass
class DriverRun:
    lines: list[dict]
    returncode: int
    stderr: str


def run_driver(
    db: Path,
    *args: str,
    kill_when: Callable[[dict], bool] | None = None,
    timeout: float = 60,
) -> DriverRun:
    """Run the driver; SIGKILL it as soon as `kill_when` accepts a line."""
    command = [sys.executable, str(DRIVER), "--db", str(db), *args]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=DRIVER.parent,
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    lines: list[dict] = []
    try:
        assert proc.stdout is not None and proc.stderr is not None
        for raw in proc.stdout:
            line = json.loads(raw)
            lines.append(line)
            if kill_when is not None and kill_when(line):
                proc.kill()
                break
        proc.wait()
        return DriverRun(lines, proc.returncode, proc.stderr.read())
    finally:
        watchdog.cancel()
        proc.kill()
        proc.wait()
