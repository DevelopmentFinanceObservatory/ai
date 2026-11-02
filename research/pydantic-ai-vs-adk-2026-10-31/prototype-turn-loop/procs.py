"""Driver child processes, read line by line from a thread so a test can wait on
several at once and kill one on a chosen line."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent


@dataclass
class Child:
    proc: subprocess.Popen
    lines: list[dict] = field(default_factory=list)
    inbox: queue.Queue = field(default_factory=queue.Queue)

    def wait_for(self, accept: Callable[[dict], bool], timeout: float = 30) -> dict:
        """The first line, already read or still to come, that `accept` takes."""
        for line in self.lines:
            if accept(line):
                return line
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            try:
                line = self.inbox.get(timeout=left)
            except queue.Empty:
                break
            if line is None:
                break
            self.lines.append(line)
            if accept(line):
                return line
        raise TimeoutError(f"no matching line; got {self.lines}; stderr {self.stderr()}")

    def finish(self, timeout: float = 60) -> list[dict]:
        """Every line, once the process has exited."""
        self.proc.wait(timeout=timeout)
        while (line := self.inbox.get(timeout=5)) is not None:
            self.lines.append(line)
        return self.lines

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()

    def stderr(self) -> str:
        if self.proc.poll() is None or self.proc.stderr is None:
            return ""
        return self.proc.stderr.read()

    def last(self) -> dict:
        return self.lines[-1]


def spawn(script: str, *args: str) -> Child:
    proc = subprocess.Popen(
        [sys.executable, str(HERE / script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=HERE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    child = Child(proc)

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if raw.startswith("{"):
                child.inbox.put(json.loads(raw))
        child.inbox.put(None)

    threading.Thread(target=pump, daemon=True).start()
    return child


def run(script: str, *args: str, timeout: float = 60) -> list[dict]:
    child = spawn(script, *args)
    try:
        lines = child.finish(timeout)
    finally:
        if child.proc.poll() is None:
            child.kill()
    if child.proc.returncode != 0:
        raise RuntimeError(f"{script} {args} exited {child.proc.returncode}: {child.stderr()}")
    return lines
