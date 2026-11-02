"""Run `durable_driver.py`, or a one-off DBOS command, as a child process."""

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
DRIVER = HERE / "durable_driver.py"
ENV = {
    **os.environ,
    "PYDANTIC_AI_NO_BANNER": "1",
    "PYTHONPATH": os.pathsep.join([str(HERE), str(HERE.parent / "minimal")]),
}


@dataclass
class DriverRun:
    lines: list[dict]
    returncode: int
    stderr: str


def run_driver(
    db: Path,
    dbos_db: Path,
    executor_id: str,
    *args: str,
    kill_when: Callable[[dict], bool] | None = None,
    timeout: float = 60,
) -> DriverRun:
    """Run the driver; SIGKILL it as soon as `kill_when` accepts a line."""
    command = [
        sys.executable,
        str(DRIVER),
        "--db",
        str(db),
        "--dbos-db",
        str(dbos_db),
        "--executor-id",
        executor_id,
        *args,
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=HERE,
        env=ENV,
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


CANCEL = """
import sys, time
from dbos import DBOSClient
client = DBOSClient(system_database_url=sys.argv[1])
client.cancel_workflow(sys.argv[2])
print(time.time())
"""


def cancel_from_another_process(dbos_db: Path, workflow_id: str) -> float:
    """Cancel `workflow_id` from a fresh process through `DBOSClient`; returns the
    wall-clock time the cancel was committed."""
    done = subprocess.run(
        [sys.executable, "-c", CANCEL, f"sqlite:///{dbos_db}", workflow_id],
        capture_output=True,
        text=True,
        env=ENV,
        timeout=60,
        check=True,
    )
    return float(done.stdout.strip().splitlines()[-1])
