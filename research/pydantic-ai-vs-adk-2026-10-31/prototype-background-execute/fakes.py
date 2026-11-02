"""Test doubles: a tool surface that logs, reports and can be slowed or held,
a run ledger kept in a JSON file, and a publisher that records what it published.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.materialisation import FrameRef
from compiler.surface import ProgressCallback, ProgressReport, ToolCallResult, deliver_progress
from executor.completion import FramePublished
from executor.ledger import InMemoryRunLedger, RunChange, RunRecord, _Book, stored_run
from harness.scripted import ScriptedSurface, scripted_publish_result
from planner.workflow import ToolId
from pydantic import JsonValue


@dataclass(kw_only=True)
class SlowSurface(ScriptedSurface):
    """`ScriptedSurface` that logs every call's start, end or cancellation, sends
    `reports` progress reports per call, and can be slowed or held per tool.

    `delay[tool]` spreads that many seconds evenly after the reports. `hold[tool]` waits on the
    event after the reports. `log` holds one entry per line, with `time.monotonic()`;
    `log_file`, when set, gets each as a JSON line with the process id, and
    `echo` prints each to stdout for a parent process to read.
    """

    reports: int = 2
    delay: dict[str, float] = field(default_factory=dict)
    hold: dict[str, asyncio.Event] = field(default_factory=dict)
    hold_forever: frozenset[str] = frozenset()
    log: list[dict[str, Any]] = field(default_factory=list)
    log_file: Path | None = None
    echo: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        super().__init__()

    def record(self, tool: str, event: str, **extra: Any) -> None:
        entry = {"pid": os.getpid(), "tool": tool, "event": event, "t": time.monotonic(), **extra}
        with self._lock:
            self.log.append(entry)
            if self.log_file is not None:
                with self.log_file.open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")
        if self.echo:
            sys.stdout.write(json.dumps({"type": "tool", **entry}) + "\n")
            sys.stdout.flush()

    def lines(self) -> list[str]:
        return [f"{e['tool']} {e['event']}" for e in self.log if e["event"] != "progress"]

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        name = tool.name
        self.record(name, "start")
        try:
            for n in range(1, self.reports + 1):
                message = f"{name} {n} of {self.reports}"
                self.record(name, "progress", message=message)
                if on_progress is not None:
                    deliver_progress(
                        on_progress, ProgressReport(progress=n, total=self.reports, message=message)
                    )
                if name in self.delay:
                    await asyncio.sleep(self.delay[name] / self.reports)
            if name in self.hold:
                await self.hold[name].wait()
            if name in self.hold_forever:
                await asyncio.sleep(600)
            result = await super().call_tool(tool=tool, arguments=arguments)
        except asyncio.CancelledError:
            self.record(name, "cancelled")
            raise
        self.record(name, "end")
        return result


@dataclass
class RecordingPublisher:
    """`scripted_publish_result`, recording every frame it publishes."""

    published: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, frame: FrameRef) -> FramePublished:
        self.published.append(frame.model_dump(mode="json"))
        return await scripted_publish_result(frame)


class FileRunLedger(InMemoryRunLedger):
    """`InMemoryRunLedger` read from and written back to one JSON file around
    every call, so a second process sees what the first recorded."""

    def __init__(self, path: Path, *, scope: Callable[[], str]) -> None:
        super().__init__(scope=scope)
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> None:
        self._books = defaultdict(_Book)
        if not self._path.exists():
            return
        for scope, raw in json.loads(self._path.read_text()).items():
            self._books[scope] = _Book(
                runs={run_id: stored_run(run) for run_id, run in raw["runs"].items()},
                by_key=raw["by_key"],
                submissions=raw["submissions"],
                retrievals=raw["retrievals"],
                plans=raw["plans"],
            )

    def _save(self) -> None:
        data = {
            scope: {
                "runs": {run_id: run.model_dump(mode="json") for run_id, run in book.runs.items()},
                "by_key": book.by_key,
                "submissions": book.submissions,
                "retrievals": book.retrievals,
                "plans": book.plans,
            }
            for scope, book in self._books.items()
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._path)

    async def admit(self, candidate: RunRecord, *, submission: JsonValue) -> Any:
        with self._lock:
            self._load()
            admission = await super().admit(candidate, submission=submission)
            self._save()
            return admission

    async def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            self._load()
            return await super().get(run_id)

    async def submission(self, run_id: str) -> JsonValue:
        with self._lock:
            self._load()
            return await super().submission(run_id)

    async def retrievals(self, run_id: str) -> JsonValue:
        with self._lock:
            self._load()
            return await super().retrievals(run_id)

    async def store_plan(self, revision: str, plan: JsonValue) -> None:
        with self._lock:
            self._load()
            await super().store_plan(revision, plan)
            self._save()

    async def plan(self, revision: str) -> JsonValue:
        with self._lock:
            self._load()
            return await super().plan(revision)

    async def unfinished(self) -> tuple[RunRecord, ...]:
        with self._lock:
            self._load()
            return await super().unfinished()

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        with self._lock:
            self._load()
            recorded = await super().record(run_id, change)
            self._save()
            return recorded


def ledger_snapshot(ledger: InMemoryRunLedger) -> dict[str, Any]:
    """Every book the ledger holds, as JSON."""
    return {
        scope: {
            "runs": [run.model_dump(mode="json") for run in book.runs.values()],
            "by_key": sorted(book.by_key),
            "submissions": list(book.submissions.values()),
            "retrievals": list(book.retrievals.values()),
            "plans": book.plans,
        }
        for scope, book in ledger._books.items()
    }
