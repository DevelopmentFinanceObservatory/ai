"""Record what data-mcp answers for each task: its searches and the skills they name.

Runs data-mcp in process (`data_mcp.server.mcp`) against the apis service at `DFO_APIS_URL`,
so that platform has to be reachable. Each task's searches go through the
orchestrator's own `search` tool, as `entry.orchestrator_state` replays them, so the
recording holds exactly the calls a replay makes. Then every skill a search names, and
`dfo-faq`, is read with each file it lists, page by page, and each first-year read
`propose_plan` can make for a dataset the planner is shown (`submission.service.first_year_reads`).

    just eval-planner-entry-record                   # tasks with no recording, a stale one,
                                                     # or one lacking a first-year read
    just eval-planner-entry-record --all             # every task again
    just eval-planner-entry-record --only share-health-oda-kenya

A task whose search answers an error, or whose `plan` call is refused, is not recorded.
A recording is data the eval replays as it stands, so record again only to take in a
change to search or the skills, and run the baseline again after.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast

from compiler.surface import ToolCallResult, ToolId, ToolSchema
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp
from data_mcp.settings import Settings as DataSettings
from entry import (
    RECORDING_SCHEMA,
    RecordedResolver,
    RecordingMismatchError,
    Task,
    TaskStateError,
    load_recording,
    load_tasks,
    orchestrator_state,
    quiet_logs,
    run_entry,
    save_recording,
    stray_recordings,
    surface_key,
)
from fastmcp import Client
from harness.tool_contracts import ResolverError, resolver_digest
from mcp.types import TextContent
from planner.agent import read_full_history
from planner.plan import SessionState
from planner.skills import DFO_FAQ_SKILL_ID
from submission.service import first_year_reads
from validate import ScriptedLlm


class RecordingResolver:
    """The orchestrator's `Resolver` over in-process data-mcp, keeping every call and
    answer. Its arguments are the ones `harness.mcp_client.McpResolver` sends."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.calls: list[dict[str, Any]] = []
        self.skills: list[str] = []

    async def __call__(
        self, *, question: str, parts: list[dict[str, Any]], raw_text: bool = False
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = (
            {"raw_query": question} if raw_text else {"parts": parts, "raw_query": question}
        )
        result = await self._client.call_tool("search", arguments, raise_on_error=False)
        if result.is_error:
            # The search tool answers this as an error, and the task is not recorded.
            refusal = dict(result.structured_content or {})
            raise ResolverError(f"search refused: {refusal}", refusal=refusal)
        envelope = (result.meta or {}).get(SEARCH_ENVELOPE_META_KEY)
        if not isinstance(envelope, dict):
            raise RuntimeError(f"search answered no envelope for {question!r}")
        self.calls.append(
            {"question": question, "parts": parts, "raw_text": raw_text, "envelope": envelope}
        )
        digest = resolver_digest(envelope)
        self.skills += [skill["id"] for skill in digest.get("skills") or []]
        return digest


def _result(result: Any) -> dict[str, Any]:
    """A tool result as `compiler.surface.McpSurface` hands it on, without `_meta`."""
    structured = result.structured_content
    return {
        "is_error": result.is_error,
        "structured_content": dict(structured) if isinstance(structured, Mapping) else None,
        "text": "\n".join(block.text for block in result.content if isinstance(block, TextContent)),
    }


async def _skill_results(client: Any, skill_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Each skill, and every page of `SKILL.md` and of each file it lists, by the key
    `entry.RecordedSurface` looks it up under. The arguments are the ones
    `planner.skills.build_skill_tools` sends."""
    results: dict[str, dict[str, Any]] = {}
    for skill_id in dict.fromkeys(skill_ids):
        arguments: dict[str, Any] = {"skill_id": skill_id}
        skill = _result(await client.call_tool("get_skill", arguments, raise_on_error=False))
        results[surface_key("get_skill", arguments)] = skill
        listed = (skill["structured_content"] or {}).get("resources") or []
        prefix = f"skill://{skill_id}/"
        paths = [
            "SKILL.md",
            *(
                item["uri"].removeprefix(prefix)
                for item in listed
                if isinstance(item, Mapping) and str(item.get("uri", "")).startswith(prefix)
            ),
        ]
        for path in dict.fromkeys(paths):
            page: dict[str, Any] | None = {"skill_id": skill_id, "path": path, "offset": 0}
            while page is not None:
                answer = _result(
                    await client.call_tool("get_skill_file", page, raise_on_error=False)
                )
                results[surface_key("get_skill_file", page)] = answer
                content = answer["structured_content"] or {}
                following = content.get("next_offset")
                page = (
                    {**page, "offset": following, "digest": content.get("digest")}
                    if isinstance(following, int) and not answer["is_error"]
                    else None
                )
    return results


# How long a first-year read may take while recording. Production allows
# `planner.agent.FIRST_YEAR_READ_DEADLINE_SECONDS`; a replay answers at once, so a recording
# keeps whatever Search answers, however long it takes.
RECORDING_READ_SECONDS = 180.0


class _RecordingSurface:
    """The planner's `ToolSurface` over in-process data-mcp: a call `held` answers is
    answered from it, and any other is made live and kept in `results`, by the key
    `entry.RecordedSurface` looks it up under."""

    def __init__(self, client: Any, held: Mapping[str, Mapping[str, Any]]) -> None:
        self._client = client
        self._held = held
        self.results: dict[str, dict[str, Any]] = {}

    async def list_tools(self) -> list[ToolSchema]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        key = surface_key(tool.name, arguments)
        if key not in self._held:
            answer = await self._client.call_tool(tool.name, dict(arguments), raise_on_error=False)
            self.results[key] = _result(answer)
        return ToolCallResult.model_validate(self._held.get(key) or self.results[key])


async def _first_year_results(
    client: Any, task: Task, recording: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The answers to each first-year read an entry over `recording` can make that the
    recording does not hold, read now. A scripted entry with no plan leaves the datasets
    the seam showed the planner in its state, and those are read."""
    entry = await run_entry(task, recording, prefix_cache=None, model=ScriptedLlm())
    surface = _RecordingSurface(client, recording["tools"])
    for identity in first_year_reads(cast(SessionState, entry.state)):
        await read_full_history(surface, identity, deadline_s=RECORDING_READ_SECONDS)
    return surface.results


async def record_task(client: Any, task: Task) -> dict[str, Any]:
    """`task`'s recording: its searches' envelopes, every skill they name, and each
    first-year read the planner can make."""
    resolver = RecordingResolver(client)
    await orchestrator_state(task, resolver)
    recording = {
        "schema": RECORDING_SCHEMA,
        "task": task.id,
        "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "dfo_apis_url": DataSettings().dfo_apis_url,
        "searches": resolver.calls,
        # The planner's `get_skill` declaration names `dfo-faq` whatever the readings say.
        "tools": await _skill_results(client, [*resolver.skills, DFO_FAQ_SKILL_ID]),
    }
    recording["tools"] |= await _first_year_results(client, task, recording)
    return recording


async def _replaying(task: Task) -> dict[str, Any] | None:
    """`task`'s recording when its searches still replay against it, else `None`."""
    try:
        recording = load_recording(task.id)
        resolver = RecordedResolver(recording["searches"])
        await orchestrator_state(task, resolver)
    except (FileNotFoundError, RecordingMismatchError, TaskStateError):
        return None
    return recording if resolver.left() == 0 else None


async def main_async(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [task for task in tasks if task.id in wanted]
        if missing := wanted - {task.id for task in tasks}:
            raise SystemExit(f"unknown task ids {sorted(missing)}")
    quiet_logs()
    failed = 0
    gate = asyncio.Semaphore(args.concurrency)

    async def one(client: Any, task: Task) -> None:
        nonlocal failed
        async with gate:
            held = None if args.all else await _replaying(task)
            try:
                if held is not None:
                    added = await _first_year_results(client, task, held)
                    if added:
                        save_recording(task.id, {**held, "tools": held["tools"] | added})
                        print(f"ok   {task.id}: first-year reads {len(added)}", file=sys.stderr)
                    return
                recording = await record_task(client, task)
            except Exception as exc:
                failed += 1
                print(f"FAIL {task.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return
        save_recording(task.id, recording)
        statuses = [
            part.get("status")
            for call in recording["searches"]
            for part in (call.get("envelope") or {}).get("parts") or []
        ]
        print(
            f"ok   {task.id}: searches {len(recording['searches'])} {statuses}, "
            f"tool answers {len(recording['tools'])}",
            file=sys.stderr,
        )

    async with Client(mcp) as client:
        await asyncio.gather(*(one(client, task) for task in tasks))
    for path in stray_recordings({task.id for task in load_tasks()}):
        path.unlink()
        print(f"removed {path.parent.name}/{path.name}, which no task records", file=sys.stderr)
    print(f"{len(tasks)} tasks: {failed} failed", file=sys.stderr)
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--all", action="store_true", help="record every task again")
    parser.add_argument("--concurrency", type=int, default=4, help="tasks recorded at once")
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
