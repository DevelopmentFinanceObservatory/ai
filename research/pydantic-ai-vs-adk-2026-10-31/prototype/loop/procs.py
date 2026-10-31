"""Run `driver.py` as a child process and read its frames as they arrive."""

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
    frames: list[dict]
    returncode: int
    stderr: str


def run_driver(
    db: Path,
    *args: str,
    global_args: tuple[str, ...] = (),
    kill_when: Callable[[dict], bool] | None = None,
    timeout: float = 60,
) -> DriverRun:
    """Run one driver step; SIGKILL it as soon as `kill_when` accepts a frame."""
    command = [sys.executable, str(DRIVER), "--db", str(db), *global_args, *args]
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
    frames: list[dict] = []
    try:
        assert proc.stdout is not None and proc.stderr is not None
        for line in proc.stdout:
            frame = json.loads(line)
            frames.append(frame)
            if kill_when is not None and kill_when(frame):
                proc.kill()
                break
        proc.wait()
        return DriverRun(frames, proc.returncode, proc.stderr.read())
    finally:
        watchdog.cancel()
        proc.kill()
        proc.wait()
