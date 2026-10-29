"""Rewrites each stored spec block's `_bound_refs` mapping as `BoundEntity` rows.
Run as `python -m harness.bound_refs_migration`.

A spec block carries under `_bound_refs` the refs Search bound for each entity
of its part (`harness.tool_contracts.resolver_digest`). Blocks written by a
release before `planner.workflow.BoundEntity` hold a mapping from each entity
text, case-folded, to its refs. `compiler.sources.source_from_spec_block`
reads only a list of `BoundEntity` rows and refuses the mapping, so every
reader drops such a block's source. This job rewrites each mapping in the
session store as the list the reader takes.

Stored blocks live in the `state` of rows of the `sessions` table of the
ADK session store (`harness.session_store`), under two apps:

- `dfo_analyst_graph.ledger`, a conversation's ledger session: the `specs`
  of each turn record (`studio:turn:<n>`), which
  `ConversationState.earlier_block` hands to every later turn;
- `dfo_analyst_graph`, a turn's own session: `orchestrator:specs`, which
  `StudioStore.load` reads for a turn still open, each planner pin's
  `block` (`planner:pin_sources`), and the `spec_block` and
  `candidate_blocks` of the search digests the planner holds
  (`planner:aligned_pins`, `planner:searched_pins`).

The job finds a block as any object carrying `_bound_refs`, a key only
`harness.tool_contracts.resolver_digest` writes, at any depth of either
app's session state, so it reaches every place a block is held.

Each mapping entry `text: refs` becomes `BoundEntity(identity=text,
text=text, role=None, refs=refs)`. The mapping kept neither the row's pinned
`dcid` nor its stated `name` nor its `role_text`, and no other field of the
block holds them, so `identity` takes the documented fallback, the text, and
`role` is unstated. `compiler.evidence.bound_refs_of` then finds the row by
its text, case-folded, with any role fitting, which is how the mapping was
read. An entry whose refs hold no string was read as no entry and is
dropped; a mapping left with no entry removes the key, as a block whose part
bound nothing is written. A list is left alone. A converted block is kept
only when `source_from_spec_block` reads it, and a block that cannot be
converted is left as it is and reported with its reason.

Each session is read and written on its own. A write sets `state` only
where the row's `update_time` still holds the value read, and keeps that
`update_time`, so a session another writer appended to in between is left
for a rerun and an instance holding the session keeps a valid revision
marker. No event is appended. The job is safe to repeat: a second run finds
every converted block already a list.

`--dry-run`, the default, reads every session and writes nothing. `--apply`
writes. The store comes from the settings the service uses
(`harness.settings.Settings`): `DATABASE_URL`, or the three
`SESSION_STORE_*` variables of the Cloud SQL Connector path, which connects
over PSC with IAM auth as `SESSION_STORE_RUNTIME_DB_USER`. Only the `ai`
service's runtime service account is a database user, and the instance is
reachable only from the service's VPC, so on a deployed environment the job
runs as a Cloud Run job execution on the release's image, under that
service account and on that network (deploy/README.md, "Bound refs
migration").

Exit status: 0 when every session was read and, under `--apply`, every
change written; 1 when a session changed during the run and was left for a
rerun; 2 when no database session store is configured.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from compiler.sources import source_from_spec_block
from google.adk.sessions import DatabaseSessionService

# The `sessions` table, whose columns ADK's v0 and v1 schemas share.
from google.adk.sessions.schemas.v1 import StorageSession
from planner.workflow import BoundEntity
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.graph.state import AGENT_NAME
from harness.session_store import build_session_service
from harness.settings import Settings

TURN_APP = AGENT_NAME
LEDGER_APP = f"{AGENT_NAME}.ledger"


class Outcome(StrEnum):
    """What the job found in one stored block's `_bound_refs`."""

    CONVERTED = "converted"
    ALREADY_LIST = "already_list"
    UNCONVERTIBLE = "unconvertible"


@dataclass(frozen=True)
class BlockResult:
    """One block's outcome, its rewritten form when converted, and why it was not."""

    outcome: Outcome
    block: Mapping[str, Any]
    reason: str | None = None
    dropped: int = 0


def migrate_block(block: Mapping[str, Any]) -> BlockResult:
    """`block` with a mapping-form `_bound_refs` rewritten as `BoundEntity` rows.

    See the module docstring for the conversion. A block whose
    `_bound_refs` is a list comes back unchanged.
    """
    raw = block.get("_bound_refs")
    if isinstance(raw, list):
        return BlockResult(Outcome.ALREADY_LIST, block)
    if not isinstance(raw, Mapping):
        return BlockResult(
            Outcome.UNCONVERTIBLE, block, reason=f"_bound_refs is a {type(raw).__name__}"
        )
    rows: list[dict[str, Any]] = []
    dropped = 0
    for text, refs in raw.items():
        listed = refs if isinstance(refs, list | tuple) else ()
        strings = [ref for ref in listed if isinstance(ref, str)]
        if not text or not strings:
            dropped += 1
            continue
        row = BoundEntity(identity=text, text=text, role=None, refs=tuple(dict.fromkeys(strings)))
        rows.append(row.model_dump(mode="json"))
    converted = {key: value for key, value in block.items() if key != "_bound_refs"}
    if rows:
        converted["_bound_refs"] = rows
    try:
        source_from_spec_block(converted)
    except ValueError as exc:
        reason = f"unreadable after conversion: {str(exc).splitlines()[0]}"
        return BlockResult(Outcome.UNCONVERTIBLE, block, reason=reason)
    return BlockResult(Outcome.CONVERTED, converted, dropped=dropped)


@dataclass
class Report:
    """What a run found and did, by block and by session."""

    apply: bool
    sessions: Counter[str] = field(default_factory=Counter)
    blocks: Counter[tuple[str, Outcome]] = field(default_factory=Counter)
    dropped_entries: int = 0
    unconvertible: list[str] = field(default_factory=list)
    written: int = 0
    changed_during_run: list[str] = field(default_factory=list)

    def render(self) -> str:
        """The report an operator reads, one count per line."""
        lines = [
            f"bound_refs migration ({'apply' if self.apply else 'dry run'})",
            f"sessions scanned: {sum(self.sessions.values())}"
            f" (ledger {self.sessions[LEDGER_APP]}, turn {self.sessions[TURN_APP]})",
            f"blocks scanned: {sum(self.blocks.values())}",
        ]
        for outcome in Outcome:
            where = sorted((at, n) for (at, held), n in self.blocks.items() if held is outcome)
            label = f"{'' if self.apply else 'would be '}converted"
            name = label if outcome is Outcome.CONVERTED else outcome.value
            lines.append(f"  {name}: {sum(n for _, n in where)}")
            lines.extend(f"    {at}: {n}" for at, n in where)
        lines.extend(f"    {line}" for line in self.unconvertible)
        lines.append(f"mapping entries with no refs, dropped: {self.dropped_entries}")
        lines.append(f"sessions {'written' if self.apply else 'to write'}: {self.written}")
        lines.append(
            f"sessions changed during the run, left for a rerun: {len(self.changed_during_run)}"
        )
        lines.extend(f"    {line}" for line in self.changed_during_run)
        return "\n".join(lines)


def _location(app_name: str, key: str) -> str:
    """Where a block sits, for the report: the app and its state key, a
    per-item key such as `studio:turn:3` read as `studio:turn:*`."""
    head, _, _ = key.rpartition(":")
    return f"{app_name} {head}:*" if head.count(":") else f"{app_name} {key}"


def _migrate_state(
    app_name: str, session_id: str, state: Mapping[str, Any], report: Report
) -> dict[str, Any]:
    """The top-level keys of `state` holding a converted block, with their new values.

    A block is any object carrying `_bound_refs`, at any depth (module docstring).
    """

    def walk(value: Any, where: str) -> Any:
        if isinstance(value, Mapping) and "_bound_refs" in value:
            result = migrate_block(value)
            report.blocks[where, result.outcome] += 1
            report.dropped_entries += result.dropped
            if result.outcome is Outcome.UNCONVERTIBLE:
                spec_id = value.get("spec_id")
                report.unconvertible.append(f"{session_id} {where} {spec_id}: {result.reason}")
            return result.block
        if isinstance(value, Mapping):
            return {key: walk(item, where) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, where) for item in value]
        return value

    changed: dict[str, Any] = {}
    for key, value in state.items():
        rewritten = walk(value, _location(app_name, key))
        if rewritten != value:
            changed[key] = rewritten
    return changed


async def run_migration(*, engine: AsyncEngine, apply: bool) -> Report:
    """Scan every ledger and turn session under `engine`, converting under `apply`."""
    report = Report(apply=apply)
    async with engine.connect() as connection:
        keys = (
            await connection.execute(
                select(StorageSession.app_name, StorageSession.user_id, StorageSession.id)
                .where(StorageSession.app_name.in_((LEDGER_APP, TURN_APP)))
                .order_by(StorageSession.app_name, StorageSession.user_id, StorageSession.id)
            )
        ).all()
    for app_name, user_id, session_id in keys:
        await _migrate_session(engine, app_name, user_id, session_id, report)
    return report


async def _migrate_session(
    engine: AsyncEngine, app_name: str, user_id: str, session_id: str, report: Report
) -> None:
    key = (
        (StorageSession.app_name == app_name)
        & (StorageSession.user_id == user_id)
        & (StorageSession.id == session_id)
    )
    async with engine.connect() as connection:
        row = (
            await connection.execute(
                select(StorageSession.state, StorageSession.update_time).where(key)
            )
        ).one_or_none()
    if row is None:
        return
    state: dict[str, Any] = row.state or {}
    read_at: datetime = row.update_time
    report.sessions[app_name] += 1
    changed = _migrate_state(app_name, session_id, state, report)
    if not changed:
        return
    if not report.apply:
        report.written += 1
        return
    async with engine.begin() as connection:
        written = await connection.execute(
            update(StorageSession)
            .where(key & (StorageSession.update_time == read_at))
            .values(state={**state, **changed}, update_time=read_at)
        )
    if written.rowcount == 1:
        report.written += 1
    else:
        report.changed_during_run.append(f"{app_name} {session_id}")


async def _run(*, settings: Settings, apply: bool) -> Report | None:
    handle = build_session_service(settings=settings)
    try:
        if not isinstance(handle.service, DatabaseSessionService):
            return None
        return await run_migration(engine=handle.service.db_engine, apply=apply)
    finally:
        await handle.aclose()


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.bound_refs_migration",
        description="Rewrite each stored spec block's _bound_refs mapping as BoundEntity "
        "rows. Reads only, unless --apply is given.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="count what would change, write nothing (the default)",
    )
    mode.add_argument(
        "--apply", dest="apply", action="store_true", help="write the converted blocks"
    )
    parser.set_defaults(apply=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job and print its report. Returns the exit status."""
    arguments = _arguments(argv)
    report = asyncio.run(_run(settings=Settings(), apply=arguments.apply))
    if report is None:
        print(
            "bound_refs migration: set DATABASE_URL or the SESSION_STORE_* variables",
            file=sys.stderr,
        )
        return 2
    print(report.render())
    return 1 if report.changed_during_run else 0


if __name__ == "__main__":
    sys.exit(main())
