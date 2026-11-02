"""The shared run log: one JSON line per node run or execute step, from any process."""

import json
import os
import time
from pathlib import Path
from typing import Any


def record(
    path: Path | None, *, runtime: str, turn: str, node: str, what: str, **extra: Any
) -> None:
    """Append one line. A single `write` on an `O_APPEND` descriptor keeps lines
    from concurrent processes whole."""
    if path is None:
        return
    line = {"t": time.time(), "pid": os.getpid(), "runtime": runtime, "turn": turn}
    line |= {"node": node, "what": what, **extra}
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(line) + "\n").encode())
    finally:
        os.close(fd)


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(raw) for raw in path.read_text().splitlines() if raw]


def runs(path: Path, *, turn: str | None = None, what: str = "run") -> list[str]:
    """The nodes that ran, in order, optionally for one turn."""
    return [
        line["node"]
        for line in read(path)
        if line["what"] == what and (turn is None or line["turn"] == turn)
    ]
