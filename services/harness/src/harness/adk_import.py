"""Copies every Studio conversation stored in ADK ledger sessions into the
harness's own tables. Run as `python -m harness.adk_import <verb>`.

A conversation's ledger is the ADK session `(dfo_analyst_graph.ledger,
principal, conversation_id)`; its `state` holds one `studio:` key per
durable field of `ConversationState`, per item of its growing collections
(`studio:turn:0003`, `studio:component:<id>`, ...) and per run-ledger item.
The job reads ADK's tables by SQL, through the columns of ADK's v1 schema
(`sessions.app_name`, `user_id`, `id`, `state`, `update_time`), and refuses to
start when `adk_internal_metadata` names another schema version. It bypasses
ADK's session service.

Every spec block the job reads, at any depth of a ledger session's state or
of an open turn's `orchestrator:specs`, is read with its `_bound_refs` in
the form `compiler.sources.source_from_spec_block` takes (`_convert_block`):
a block stored by a release before `planner.workflow.BoundEntity` maps each
entity text to its refs, and the reader refuses that mapping.

Verbs:

- `import`: applies the harness's schema migrations, then copies each
  ledger session into `conversations`, its item tables and the run
  ledger's tables, writing through `harness.studio_store`'s own write path,
  so what it writes is what `StudioStore.load` reads. Records are copied
  verbatim, bar the spec block conversion, the open turn's fields and
  `pending` included, because this release resumes a pause ADK holds.
  A conversation is refused whole, with every reason named, for a
  structural fault: a key the job does not know, a turn or approval
  position that is not a number or leaves a gap, a write token that is not
  a string, or a run-ledger item naming a run the ledger does not hold.
  Any other failure while importing one conversation refuses it too, and
  the run goes on with the rest. Records this release's load skips or
  fails on (a plan record, search reading, identity, approval, canvas,
  pending interrupt or run record its model refuses, or a spec block whose
  `_bound_refs` does not convert, which the reader drops) are reported and
  copied as they are.

  `adk_ledger_imports` records a digest of each imported ADK state, the
  ledger session's `update_time` read with it, and a digest of the rows
  written, so a rerun leaves alone a conversation whose ADK state has not
  moved, re-imports one whose ADK state moved while its rows are as
  imported, and reports a conflict when both moved, unless `--prefer-adk`
  names it as `principal/conversation_id`. Under `--apply` a conflict
  marks the conversation diverged, and the serving release refuses it
  until `--prefer-adk` re-imports it. `--prefer-adk` refuses a
  conversation a stream holds the lease of, unless `--force`, and a name
  no ledger session has stops the job before it writes. Under `--apply`
  the job locks the conversation's row before it reads the ledger session,
  the import record and the rows' digest, and rewrites the row in place,
  keeping its id, so a write the serving release makes to the same
  conversation either lands first and counts as a change on the tables'
  side, or waits for the import to commit. A dry run takes no lock and
  builds every row the write would write, so it reports the refusals the
  write would raise.

  `--keep-tables`, naming a refused or diverged conversation the tables
  hold, keeps the tables' copy and gives up ADK's: its import record is
  marked `kept_tables` at the ledger session's `update_time` and state, the
  rows are left as they are, and the serving release serves them. Once the
  ledger session moves again the record is no longer current, the
  conversation is refused or in conflict again, and the cutover gate fails
  on it. `--discard`, naming a refused conversation the tables do not hold,
  records it the same way and gives the conversation up: the release
  starts it afresh on its next turn, and the report and the gate list it
  as discarded. A name `--keep-tables` gives to a conversation the tables
  do not hold, or to one that is neither refused nor diverged, refuses it,
  and so does a name `--discard` gives to any conversation but a refused
  one the tables do not hold. One name given to two of `--prefer-adk`,
  `--keep-tables` and `--discard` stops the job before it writes.

  The report counts the spec blocks converted and those that do not
  convert, in the ledger sessions and, on a line of their own, in the open
  turns' sessions, read in the conversation's own transaction. It lists an
  open turn's blocks that do not convert under a heading of their own: the
  job does not copy them, and the release reads them from the turn session
  as stored. It gives the size distribution (p50, p95, max, in bytes of JSON
  text) of the ledger sessions' states and of the open turns'
  `orchestrator:specs`.

  `--verify` writes nothing and compares, for every ledger session, the
  conversation as the previous release loaded ADK's state
  (`_previous_release_load`, that release's own key mapping, kept apart
  from `read_ledger`, over the state with its spec blocks converted) with
  `StudioStore.load` from the tables, the canvas included, and the
  run-ledger items as that release read them (`_previous_release_runs`)
  with the run rows, leaving out the lease's cancel flag. A key
  `read_ledger` maps to the wrong field, or a value it changes, therefore
  shows as a difference.
  Both sides take the open turn's spec blocks from its turn session
  through the reader the server serves (`open_turn_specs`), converted as
  `StudioStore.load` converts them. A conversation whose ADK state moved
  since the job imported it while its rows did not (the previous release
  wrote it, serving until cutover), and one whose rows moved since,
  written by the release that reads the tables (one written while verify
  compares it included), are each listed and not compared. One where both
  moved is a conflict, listed, not compared, and fails the verify. A kept
  conversation is listed and not compared. A ledger session whose
  `studio:segment` lease is live on the database's clock is listed apart
  and does not fail the verify: the previous revision still streams on it
  and can write it again, and the gate is what fails on it.
- `close-open-turns`: closes, in the tables, every open turn that holds no
  live lease and ran on ADK (its turn session `{conversation_id}#t{n}`
  under app `dfo_analyst_graph` exists), as `ConversationState.begin_turn`
  closes an open turn: the record keeps what was asked with the error
  `abandoned`, and the blocks of the sources its plans used, read from the
  turn session's `orchestrator:specs`. Its pending interrupts and its
  lapsed lease are dropped. A turn the `turn_cursors` table holds a row
  for, by (conversation, turn_index), is left open and listed: it was
  opened there, under the same number as an ADK turn session left behind.
  A turn opens in `turn_cursors` only under the conversation's lease: its
  writer first commits the lease and the turn index on the conversation's
  row, then inserts the turn's row under `FOR SHARE` on the conversation's
  row where `lease_segment_id` is its lease. The closing write refuses a
  live lease it does not hold (`ConversationStale`) and a row whose turn
  index moved since the load, each listed as changed, reads
  `turn_cursors` under its `FOR UPDATE` on the same row, and writes the
  lease columns null. So a turn whose row committed, or waits to commit, before
  that lock is left open, and an insert that waits behind the lock finds
  the lease gone and opens no turn. Whether the table exists is read
  under the same lock. A block that does not convert is kept as stored, as
  `import` keeps one, and reported. A turn a stream holds the lease of is
  left open and listed as live on ADK when its turn session exists and
  `turn_cursors` does not hold it, and as live elsewhere otherwise. A
  stream of the previous release writes its lease to the ledger session
  and never to the tables, so a conversation whose ledger session's
  `studio:segment` lease is live on the database's clock
  (`_STREAMING_ON_ADK`) is left open and listed as live on ADK, whatever
  the tables' lease. A conversation whose import record is not current
  (`_STALE_IMPORTS`: ADK wrote its ledger session after the import
  recorded its `update_time`) is left open and listed as imported before
  ADK's last write, since the tables lack that write; `import --apply`
  is run first. Both are read once per run, where a table named
  `sessions` carries ADK's columns, and a conversation with no ledger
  session is read from the tables alone. A turn or conversation live on
  ADK fails the run, which is repeated until none is left, and so does a
  conversation imported before ADK's last write. A diverged conversation is
  left alone and listed, and fails the run: its rows lack the turns ADK
  holds, so `--prefer-adk` must take ADK's copy first. A conversation
  whose close fails is refused and left open, and the run goes on with
  the rest. `--apply` refuses to start (`NextReleaseMissing`) until the
  `turn_cursors` table exists: the next release's first migration creates
  it, and until that release serves, this release can still answer a turn
  paused on ADK, which a close would end.
- `rollback-turn-sessions`: for a rollback to the previous release, deletes
  each ADK turn session `{conversation_id}#t{n}` (app `dfo_analyst_graph`)
  numbered above its ledger session's `studio:turn_index`. This release
  creates such sessions for the turns it serves, and never writes the
  ledger session, so the previous release would claim the same numbers and
  answer 409 until its index passed them. Under `--apply` each
  conversation's ledger session is locked `FOR UPDATE` first, so a turn
  the previous release claims commits its index before or after the read,
  and the session it creates after its claim is never above the index
  read. ADK's `events` rows go with their session (`ON DELETE CASCADE`).
  A ledger session whose `studio:turn_index` is not a whole number is
  listed and left alone. The run fails while it finds any session to
  delete, and is repeated until it finds none.
- `gate`: the cutover predicate (`CUTOVER_PREDICATE`), read only. It fails
  while any ledger session is not cleared: its import record must be
  current (recorded at the session's `update_time`) and either imported
  (not diverged, with its rows) or kept (`--keep-tables`, `--discard`). It
  prints the count, the first ten, and the commands that clear them. It
  also fails while a ledger session's `studio:segment` lease is live on
  the database's clock (`_STREAMING_ON_ADK`): a stream of the previous
  revision still runs on it, and deleting that revision would cut it. It
  lists them, and is rerun once they end. It lists apart each cleared ledger session the
  operator discarded (`--discard`), so the count of conversations given up
  is in its output. Where no table named `sessions` carries ADK's columns
  it cannot check anything and stops (`AdkTablesMissing`), unless
  `--adk-archived` says the tables were archived on purpose.

The serving release reads conversations from the tables while the previous
release may finish turns on ADK. `reimport_on_access` is the
`AdkBridge` the server gives `StudioStore`: before each load it brings
the conversation up to date with its ledger session through `import`'s own
path, so a turn the previous release served after the last `--apply` is in
the tables before the next request reads them. It refuses a diverged
conversation with `harness.studio_store.ConversationHeld`. A conversation
whose ledger session the copy refuses is served as the tables hold it, and
started afresh by its next turn when they do not hold it. `open_turn_specs`
is the `turn_specs` reader the server gives `StudioStore`: it reads an open
turn's spec blocks as stored, and `StudioStore.load` converts them, as
`import` does, through the bridge's `convert_specs`, which logs a block that
does not convert.

Without `--apply` nothing is written to the conversation tables or to
ADK's. Only `import` migrates, its dry run included, so
`--verify` and `close-open-turns` also run from an image older than the
database's schema. They refuse to start when the tables do not exist.

The database comes from the settings the service uses
(`harness.settings.Settings`).

Exit status: 0 when every conversation imported, was unchanged, kept,
discarded or verified equal, every open turn eligible was closed, the gate
found every ledger session cleared, and `rollback-turn-sessions` found no
turn session to delete; 1 on a refusal, a conflict, a difference, a stream
live on ADK, a diverged conversation `close-open-turns` left alone, a
conversation it left open because ADK wrote since the import, a
conversation that changed during the run, a ledger session the gate did
not clear, or a turn session `rollback-turn-sessions` found; 2 when no
database is configured, ADK's schema is not the one the job reads, the
gate finds no ADK `sessions` table without `--adk-archived`,
`close-open-turns --apply` finds no `turn_cursors` table, the harness's
tables are missing, a migration file is refused, or `--prefer-adk`,
`--keep-tables` or `--discard` names a conversation no ledger session holds
or two of them name one.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import sys
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from common.logging import configure_logging
from common.telemetry import flush_telemetry_before_exit, init_telemetry
from compiler.sources import bound_rows, source_from_spec_block
from executor.ledger import RunStatus, stored_run
from pydantic import BaseModel, ValidationError
from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.approval import ApprovalRecord
from harness.db.engine import build_database
from harness.db.jsonb import to_json_param
from harness.db.migrate import MigrationChecksumError, apply_pending
from harness.graph.state import AGENT_NAME, ORCHESTRATOR_SPECS_KEY
from harness.settings import Settings
from harness.studio_state import ConversationState, TurnUsage
from harness.studio_store import (
    _APPROVAL_ITEM_PREFIX,
    _COMPONENT_ITEM_PREFIX,
    _KEY_COMPOSITION,
    _KEY_SEGMENT,
    _KEY_SEQ,
    _KEY_TURN_INDEX,
    _KEY_TURN_RECORDED,
    _SCALAR_KEYS,
    _TURN_ITEM_PREFIX,
    _WRITE_TOKEN_ITEM_PREFIX,
    DATABASE_NOW,
    AdkBridge,
    ConversationHeld,
    ConversationStale,
    SaveRefusedError,
    StudioStore,
    TurnSpecs,
    _delta_write,
    _DeltaWrite,
    _deserialize_pending,
    _deserialize_segment,
    _run_delta_write,
    _serialize_durable,
    _set_list_item,
    _turn_usage,
)
from harness.tool_contracts import HarnessCode
from harness.turn_records import PlanRecord, RecordedIdentity, SearchReading
from harness.wire import CompositionManifest

logger = logging.getLogger(__name__)

LEDGER_APP = f"{AGENT_NAME}.ledger"
TURN_APP = AGENT_NAME
ADK_SCHEMA_VERSION = "1"
# Frame numbers an imported conversation reserves past the last one ADK
# recorded. A stream the previous release lost may have sent frames past its
# last save, so the stream that resumes the turn starts past all of them.
IMPORTED_SEQ_MARGIN = 1000

_ADK_PREFIX = "studio:"
# Each scalar ledger key and the flat key `harness.studio_store` stores it
# under.
_SCALARS = {
    f"{_ADK_PREFIX}{key}": key
    for key in sorted({*_SCALAR_KEYS, _KEY_SEQ, _KEY_COMPOSITION, _KEY_SEGMENT})
}
# Each per-item ledger prefix and the flat prefix it maps to. Turns and
# approvals are keyed by a zero-padded position in the ledger and by the
# position as a number in the tables.
_ITEMS = {
    f"{_ADK_PREFIX}{prefix}": prefix
    for prefix in (
        _TURN_ITEM_PREFIX,
        _COMPONENT_ITEM_PREFIX,
        _WRITE_TOKEN_ITEM_PREFIX,
        _APPROVAL_ITEM_PREFIX,
    )
}
_POSITIONAL = frozenset({_TURN_ITEM_PREFIX, _APPROVAL_ITEM_PREFIX})
_RUN = "studio:run:"
_RUN_KEY = "studio:run_key:"
_RUN_SUBMISSION = "studio:run_submission:"
_RUN_RETRIEVALS = "studio:run_retrievals:"
_RUN_PLAN = "studio:run_plan:"
_RUN_PREFIXES = (_RUN_SUBMISSION, _RUN_RETRIEVALS, _RUN_PLAN, _RUN_KEY, _RUN)

# The tables holding a conversation's rows, each with the order its rows are
# digested in.
_CHILD_TABLES = (
    ("conversation_turns", "position"),
    ("conversation_components", "seq"),
    ("conversation_write_tokens", "component_id"),
    ("conversation_approvals", "position"),
    ("runs", "run_id"),
    ("run_keys", "idempotency_key"),
    ("run_plans", "revision"),
)


class AdkSchemaError(RuntimeError):
    """ADK's tables are missing or hold a schema version this job does not read."""


class TablesMissing(RuntimeError):  # noqa: N818 -- names the state it reports
    """The harness's tables do not exist: `import` has not run."""


class AdkTablesMissing(RuntimeError):  # noqa: N818 -- names the state it reports
    """No table named `sessions` carries ADK's columns, so the gate has
    nothing to check, and `--adk-archived` does not say that is intended."""


class NextReleaseMissing(RuntimeError):  # noqa: N818 -- names the state it reports
    """`close-open-turns --apply` found no `turn_cursors` table: the next
    release has not started, so this release may still answer a paused
    turn."""


class UnknownConversation(ValueError):  # noqa: N818 -- names the state it reports
    """`--prefer-adk`, `--keep-tables` or `--discard` names a conversation ADK
    holds no ledger session for."""


class NamedTwice(ValueError):  # noqa: N818 -- names the state it reports
    """Two of `--prefer-adk`, `--keep-tables` and `--discard` name the same
    conversation."""


@dataclass
class RunItems:
    """A conversation's run-ledger items: one record, submission and set of
    retrievals per run id, the run each idempotency key names, and the plan
    of each revision."""

    records: dict[str, Any] = field(default_factory=dict)
    submissions: dict[str, Any] = field(default_factory=dict)
    retrievals: dict[str, Any] = field(default_factory=dict)
    keys: dict[str, Any] = field(default_factory=dict)
    plans: dict[str, Any] = field(default_factory=dict)


@dataclass
class Blocks:
    """The spec blocks a conversion found: how many it rewrote, and why each
    one it could not rewrite failed."""

    converted: int = 0
    unconvertible: list[str] = field(default_factory=list)


def _convert_block(block: Mapping[str, Any]) -> Mapping[str, Any]:
    """`block` with a mapping-form `_bound_refs` rewritten as `BoundEntity`
    rows, `block` itself when it is already a list.

    Each entry `text: refs` becomes a row by `compiler.sources.bound_rows`:
    the mapping kept no row's pinned `dcid`, `name` or `role_text`, so the
    row's identity is its text and its role is unstated, and
    `compiler.evidence.bound_refs_of` finds it by its text, case-folded, with
    any role fitting, as it read the mapping. An entry with no text or no
    string ref is dropped. A mapping left with no entry drops the key, as a
    block whose part bound nothing is written.

    Raises:
        TypeError: `_bound_refs` is neither a mapping nor a list.
        Exception: whatever `source_from_spec_block` raises on the result,
            so a converted block is one the reader reads.
    """
    raw = block["_bound_refs"]
    if isinstance(raw, list):
        return block
    if not isinstance(raw, Mapping):
        raise TypeError(f"_bound_refs is of type {type(raw).__name__}")
    converted = {key: value for key, value in block.items() if key != "_bound_refs"}
    if rows := bound_rows(raw):
        converted["_bound_refs"] = [row.model_dump(mode="json") for row in rows]
    source_from_spec_block(converted)
    return converted


def _convert_blocks(value: Any, where: str, blocks: Blocks) -> Any:
    """`value` with every spec block in it, an object carrying `_bound_refs`
    at any depth, read by `_convert_block`. A block that does not convert is
    left as it is and named in `blocks`. `value` itself comes back when
    nothing changed."""
    if isinstance(value, Mapping) and "_bound_refs" in value:
        try:
            converted = _convert_block(value)
        # Any failure on one block is a reason to report, whatever it raised.
        except Exception as exc:
            blocks.unconvertible.append(
                f"{where}: the _bound_refs of spec block {value.get('spec_id')} does not "
                f"convert: {type(exc).__name__}: {exc}"
            )
            return value
        if converted is not value:
            blocks.converted += 1
        return converted
    if isinstance(value, Mapping):
        items = {key: _convert_blocks(item, where, blocks) for key, item in value.items()}
        return items if any(items[key] is not item for key, item in value.items()) else value
    if isinstance(value, list):
        listed = [_convert_blocks(item, where, blocks) for item in value]
        return (
            listed if any(new is not old for new, old in zip(listed, value, strict=True)) else value
        )
    return value


@dataclass
class Ledger:
    """One ledger session's state as the tables hold it.

    `flat` holds the conversation's flat keys (`harness.studio_store`) in the
    session state's own order, which is the order the canvas digest reads
    components in, with its spec blocks converted (`_convert_blocks`).
    `faults` are the structural faults that refuse the conversation.
    `findings` are the records this release's load skips or fails on, a
    block that does not convert among them. `blocks` counts the conversion.
    """

    flat: dict[str, Any]
    runs: RunItems
    faults: list[str]
    findings: list[str]
    blocks: Blocks


def _model_error(model: type[BaseModel], raw: Any) -> str | None:
    try:
        model.model_validate(raw)
    except ValidationError as exc:
        return f"not a {model.__name__}: {exc.errors()[0]['msg']}"
    return None


def _check_records(
    model: type[BaseModel], records: Any, where: str, effect: str, findings: list[str]
) -> None:
    for index, raw in enumerate(records or ()):
        if (error := _model_error(model, raw)) is not None:
            findings.append(f"{where} {index}: {effect}: {error}")


def _check_turn(position: str, record: Any, findings: list[str]) -> None:
    if not isinstance(record, dict):
        findings.append(f"turn {position}: not an object")
        return
    skipped = "load skips it"
    _check_records(PlanRecord, record.get("plans"), f"turn {position} plan", skipped, findings)
    _check_records(
        SearchReading, record.get("readings"), f"turn {position} reading", skipped, findings
    )
    for spec_id, identity in (record.get("identities") or {}).items():
        if (error := _model_error(RecordedIdentity, identity)) is not None:
            findings.append(f"turn {position} identity {spec_id}: load fails on it: {error}")


def _check_run(run_id: str, raw: Any, findings: list[str]) -> None:
    """A run record `stored_run` refuses is recorded failed by the run ledger
    when it finds it, unless that is already what it holds."""
    failure = raw.get("failure") if isinstance(raw, dict) else None
    if (
        isinstance(raw, dict)
        and raw.get("status") == RunStatus.FAILED.value
        and isinstance(failure, dict)
        and failure.get("code") == HarnessCode.RUN_RECORD_UNREADABLE.value
    ):
        return
    try:
        stored_run(raw)
    except ValidationError as exc:
        findings.append(
            f"run {run_id}: the run ledger records it failed as unreadable: "
            f"{exc.errors()[0]['msg']}"
        )


def _check_scalar(key: str, value: Any, findings: list[str]) -> None:
    if key == _KEY_COMPOSITION and value is not None:
        if (error := _model_error(CompositionManifest, value)) is not None:
            findings.append(f"composition: load fails on it: {error}")
    elif key == "pending":
        try:
            _deserialize_pending(value or {})
        except (KeyError, TypeError, AttributeError, ValidationError) as exc:
            findings.append(f"pending: load fails on it: {exc!r}")
    elif key == "turn_plans":
        _check_records(PlanRecord, value, "open turn plan", "load skips it", findings)
    elif key == "turn_readings":
        _check_records(SearchReading, value, "open turn reading", "load skips it", findings)


def read_ledger(state: dict[str, Any]) -> Ledger:
    """Map a ledger session's `state` onto the tables' flat keys and run
    items, converting its spec blocks and collecting its structural faults
    and the records this release's load skips or fails on. A key with a fault is
    left out of `flat`."""
    flat: dict[str, Any] = {}
    runs = RunItems()
    faults: list[str] = []
    findings: list[str] = []
    blocks = Blocks()
    positions: dict[str, list[int]] = {prefix: [] for prefix in _POSITIONAL}
    for key, stored in state.items():
        value = _convert_blocks(stored, key, blocks)
        if key in _SCALARS:
            flat[_SCALARS[key]] = value
            _check_scalar(_SCALARS[key], value, findings)
        elif ledger_prefix := next((p for p in _ITEMS if key.startswith(p)), None):
            prefix = _ITEMS[ledger_prefix]
            ident = key.removeprefix(ledger_prefix)
            if prefix in _POSITIONAL:
                if not ident.isdigit():
                    faults.append(f"{key}: the position is not a number")
                    continue
                ident = str(int(ident))
                positions[prefix].append(int(ident))
            if prefix == _TURN_ITEM_PREFIX:
                _check_turn(ident, value, findings)
            elif prefix == _APPROVAL_ITEM_PREFIX:
                if (error := _model_error(ApprovalRecord, value)) is not None:
                    findings.append(f"approval {ident}: load fails on it: {error}")
            elif prefix == _WRITE_TOKEN_ITEM_PREFIX and not isinstance(value, str):
                faults.append(f"{key}: the write token is not a string")
                continue
            flat[f"{prefix}{ident}"] = value
        elif run_prefix := next((p for p in _RUN_PREFIXES if key.startswith(p)), None):
            ident = key.removeprefix(run_prefix)
            if run_prefix == _RUN:
                _check_run(ident, value, findings)
                runs.records[ident] = value
            elif run_prefix == _RUN_KEY:
                runs.keys[ident] = value
            elif run_prefix == _RUN_SUBMISSION:
                runs.submissions[ident] = value
            elif run_prefix == _RUN_RETRIEVALS:
                runs.retrievals[ident] = value
            else:
                runs.plans[ident] = value
        else:
            faults.append(f"{key}: a key this job does not know")
    for prefix, found in positions.items():
        if sorted(found) != list(range(len(found))):
            faults.append(
                f"{prefix.rstrip(':')} positions are not 0 to {len(found) - 1}: {sorted(found)}"
            )
    named = (
        ("idempotency key", runs.keys.values()),
        ("submission", runs.submissions),
        ("retrievals", runs.retrievals),
    )
    for what, run_ids in named:
        faults += [
            f"the {what} of run {run_id} names a run the ledger does not hold"
            for run_id in sorted(set(run_ids) - set(runs.records), key=str)
        ]
    findings += blocks.unconvertible
    return Ledger(flat=flat, runs=runs, faults=faults, findings=findings, blocks=blocks)


@dataclass(frozen=True)
class _Session:
    """A ledger session's state and its `update_time`, read together."""

    principal: str
    conversation_id: str
    state: dict[str, Any]
    updated_at: datetime

    @property
    def name(self) -> str:
        return f"{self.principal}/{self.conversation_id}"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


async def _check_adk_schema(conn: AsyncConnection) -> None:
    if (await conn.execute(text("SELECT to_regclass('adk_internal_metadata')"))).scalar() is None:
        raise AdkSchemaError(
            "no adk_internal_metadata table: ADK's session tables are missing or use the "
            f"v0 schema; this job reads schema version {ADK_SCHEMA_VERSION}"
        )
    version = (
        await conn.execute(
            text("SELECT value FROM adk_internal_metadata WHERE \"key\" = 'schema_version'")
        )
    ).scalar()
    if version != ADK_SCHEMA_VERSION:
        raise AdkSchemaError(
            f"adk_internal_metadata names schema version {version!r}; "
            f"this job reads schema version {ADK_SCHEMA_VERSION}"
        )


async def _check_tables(engine: AsyncEngine) -> None:
    """Check ADK's schema and that the harness's tables exist, without
    migrating them."""
    async with engine.connect() as conn:
        await _check_adk_schema(conn)
        for table in ("conversations", "adk_ledger_imports"):
            if (await conn.execute(text(f"SELECT to_regclass('{table}')"))).scalar() is None:
                raise TablesMissing(f"no {table} table: run `import` first")


@dataclass(frozen=True)
class _LedgerKey:
    principal: str
    conversation_id: str
    size: int

    @property
    def name(self) -> str:
        return f"{self.principal}/{self.conversation_id}"


async def _ledger_keys(engine: AsyncEngine) -> list[_LedgerKey]:
    """Every ledger session's key and the size of its state as JSON text,
    without the states, which the job reads one at a time."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT user_id, id, coalesce(octet_length(state::text), 0) FROM sessions "
                "WHERE app_name = :app ORDER BY user_id, id"
            ),
            {"app": LEDGER_APP},
        )
        return [_LedgerKey(*row) for row in rows]


async def _read_session(
    conn: AsyncConnection, principal: str, conversation_id: str
) -> _Session | None:
    """The ledger session of the conversation, None when ADK holds none."""
    row = (
        await conn.execute(
            text(
                "SELECT state, update_time FROM sessions "
                "WHERE app_name = :app AND user_id = :principal AND id = :id"
            ),
            {"app": LEDGER_APP, "principal": principal, "id": conversation_id},
        )
    ).first()
    if row is None:
        return None
    return _Session(principal, conversation_id, dict(row.state or {}), row.update_time)


async def _ledger_sessions(engine: AsyncEngine, keys: list[_LedgerKey]) -> AsyncIterator[_Session]:
    """The ledger session of each of `keys`, its state read when the caller
    reaches it. One deleted since the keys were listed is skipped."""
    for key in keys:
        async with engine.connect() as conn:
            session = await _read_session(conn, key.principal, key.conversation_id)
        if session is not None:
            yield session


@dataclass(frozen=True)
class _TurnSpecs:
    """The spec blocks a turn's own ADK session holds, converted
    (`_convert_blocks`), and the size of what it stores, in bytes of JSON
    text."""

    specs: dict[str, Any]
    blocks: Blocks
    size: int


async def _read_turn_session(
    conn: AsyncConnection, principal: str, session_id: str
) -> _TurnSpecs | None:
    """The spec blocks the turn session `session_id` holds, read on `conn`
    and converted, empty when it holds none. None when the session does not
    exist."""
    return _converted_turn(session_id, await _stored_turn_specs(conn, principal, session_id))


def _converted_turn(
    session_id: str, stored: tuple[dict[str, Any], int] | None
) -> _TurnSpecs | None:
    """The turn session `session_id`'s spec blocks as stored
    (`_stored_turn_specs`), converted (`_converted_specs`)."""
    if stored is None:
        return None
    specs, size = stored
    blocks = Blocks()
    return _TurnSpecs(specs=_converted_specs(session_id, specs, blocks), blocks=blocks, size=size)


async def _turn_session_row(
    engine: AsyncEngine, principal: str, session_id: str
) -> tuple[dict[str, Any], int] | None:
    """`_stored_turn_specs` on a connection of its own, the one read of a
    turn session's row outside the import's transaction."""
    async with engine.connect() as conn:
        return await _stored_turn_specs(conn, principal, session_id)


async def _stored_turn_specs(
    conn: AsyncConnection, principal: str, session_id: str
) -> tuple[dict[str, Any], int] | None:
    """The spec blocks a turn's own ADK session holds, as stored, and their
    size in bytes of JSON text. None when the session does not exist."""
    row = (
        await conn.execute(
            text(
                "SELECT state -> :key, coalesce(octet_length((state -> :key)::text), 0) "
                "FROM sessions "
                "WHERE app_name = :app AND user_id = :principal AND id = :session_id"
            ),
            {
                "key": ORCHESTRATOR_SPECS_KEY,
                "app": TURN_APP,
                "principal": principal,
                "session_id": session_id,
            },
        )
    ).first()
    if row is None:
        return None
    return dict(row[0] or {}), row[1]


def open_turn_specs(engine: AsyncEngine, *, found: set[str] | None = None) -> TurnSpecs:
    """`StudioStore`'s `turn_specs` reader over `engine`: the spec blocks the
    turn session `session_id` (app `dfo_analyst_graph`) holds, read by SQL
    as stored. Empty when the session holds none or does not exist.
    `StudioStore.load` converts what it reads, once, through its bridge's
    `convert_specs`. The server serves Studio through this reader.
    `--verify` reads open turns through it and the job's bridge (`_NoAdk`),
    whose conversion is the serving bridge's (`_converted_specs`), so the
    gate checks the blocks the serving release loads. `close-open-turns`
    reads them through it too, with `found` gathering the ids of the
    sessions that exist."""

    async def turn_specs(principal: str, session_id: str) -> dict[str, Any]:
        stored = await _turn_session_row(engine, principal, session_id)
        if stored is None:
            return {}
        if found is not None:
            found.add(session_id)
        return stored[0]

    return turn_specs


def _converted_specs(session_id: str, specs: dict[str, Any], blocks: Blocks) -> dict[str, Any]:
    """`specs`, the spec blocks turn session `session_id` holds, converted
    (`_convert_blocks`), and `specs` itself when no block needed converting.
    A block that does not convert is kept as it is and named in `blocks`."""
    converted = _convert_blocks(specs, session_id, blocks)
    return specs if converted is specs else dict(converted)


# How many unconvertible spec blocks a bridge remembers having logged.
_LOGGED_BLOCKS_KEPT = 1024


async def _find_row(conn: AsyncConnection, names: dict[str, str], *, lock: bool) -> int | None:
    """The id of the row for the conversation `names` identifies by its
    `principal` and `conversation_id`, None when there is none. With `lock`,
    the row is locked for the transaction."""
    return (
        await conn.execute(
            text(
                "SELECT id FROM conversations "
                "WHERE principal = :principal AND conversation_id = :conversation_id"
                + (" FOR UPDATE" if lock else "")
            ),
            names,
        )
    ).scalar()


async def _insert_row(conn: AsyncConnection, names: dict[str, str]) -> int | None:
    """Insert the conversation's row and return its id. None when another
    writer inserted it first."""
    return (
        await conn.execute(
            text(
                "INSERT INTO conversations (principal, conversation_id) "
                "VALUES (:principal, :conversation_id) ON CONFLICT DO NOTHING RETURNING id"
            ),
            names,
        )
    ).scalar()


async def _import_record(conn: AsyncConnection, names: dict[str, str]) -> Any:
    """The conversation's row of `adk_ledger_imports`, None when the job has
    not met it."""
    return (
        await conn.execute(
            text(
                "SELECT source_digest, imported_digest, kept_tables FROM adk_ledger_imports "
                "WHERE principal = :principal AND conversation_id = :conversation_id"
            ),
            names,
        )
    ).first()


async def _lease_live(conn: AsyncConnection, pk: int) -> bool:
    """Whether a stream holds the conversation's lease, on the database's
    clock, as the store reads it."""
    return bool(
        (
            await conn.execute(
                text(
                    f"SELECT coalesce(lease_expires_at > {DATABASE_NOW}, false) "
                    "FROM conversations WHERE id = :pk"
                ),
                {"pk": pk},
            )
        ).scalar_one()
    )


# The lease field left out of the rows' digest and of `--verify`'s
# comparison. A Stop sent to this release on a stream the previous release
# runs sets it on the lease a copy brought across. The flag does not reach
# that stream and the next copy replaces it with ADK's, so ignoring it
# loses no write the tables hold.
_UNCOMPARED_LEASE_FIELD = "cancel"


# The columns of each table the digest covers: those the import writes,
# named so that a column a later migration adds leaves every digest as it was.
_DIGESTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "conversations": (
        "principal",
        "conversation_id",
        "version",
        "turn_state",
        "composition",
        "lease_segment_id",
        "lease_owner",
        "lease_expires_at",
        "seq_next",
        "seq_reserved",
    ),
    "conversation_turns": ("position", "record"),
    "conversation_components": ("component_id", "seq", "entry"),
    "conversation_write_tokens": ("component_id", "token"),
    "conversation_approvals": ("position", "record"),
    "runs": ("run_id", "record", "submission", "retrievals"),
    "run_keys": ("idempotency_key", "run_id"),
    "run_plans": ("revision", "plan"),
}


def _digested_row(table: str) -> str:
    """The SQL expression for one row of `table` as the digest reads it."""
    pairs = ", ".join(f"'{column}', {column}" for column in _DIGESTED_COLUMNS[table])
    return f"jsonb_build_object({pairs})"


async def _stored_digest(conn: AsyncConnection, pk: int) -> str:
    """A digest of every row the tables hold for the conversation, over the
    columns `_DIGESTED_COLUMNS` names. `lease_{_UNCOMPARED_LEASE_FIELD}` is
    not among them."""
    conversation = (
        await conn.execute(
            text(f"SELECT {_digested_row('conversations')} FROM conversations WHERE id = :pk"),
            {"pk": pk},
        )
    ).scalar_one()
    rows: dict[str, list[Any]] = {"conversations": [conversation]}
    for table, order in _CHILD_TABLES:
        result = await conn.execute(
            text(
                f"SELECT {_digested_row(table)} FROM {table} "
                f"WHERE conversation = :pk ORDER BY {order}"
            ),
            {"pk": pk},
        )
        rows[table] = list(result.scalars())
    return _digest(rows)


def _jsonb(value: Any) -> str | None:
    return None if value is None else to_json_param(value)


@dataclass(frozen=True)
class _ImportRows:
    """What `_write` writes for one ledger, built before any write: the
    canvas as JSON text, the statements of the flat keys, and the
    parameters of each run, idempotency key and run plan row, all without
    the row id."""

    composition: str | None
    delta: _DeltaWrite
    runs: list[dict[str, Any]]
    keys: list[dict[str, Any]]
    plans: list[dict[str, Any]]


def _import_rows(ledger: Ledger) -> _ImportRows:
    """The rows `_write` writes for `ledger`, built with no database, so a
    dry run raises what the write would raise (`_delta_write`)."""
    runs = ledger.runs
    return _ImportRows(
        composition=_jsonb(ledger.flat.get(_KEY_COMPOSITION)),
        delta=_delta_write(
            {key: value for key, value in ledger.flat.items() if key != _KEY_COMPOSITION}
        ),
        runs=[
            {
                "run_id": run_id,
                "record": to_json_param(record),
                "submission": _jsonb(runs.submissions.get(run_id)),
                "retrievals": _jsonb(runs.retrievals.get(run_id)),
            }
            for run_id, record in runs.records.items()
        ],
        keys=[{"key": key, "run_id": run_id} for key, run_id in runs.keys.items()],
        plans=[
            {"revision": revision, "plan": to_json_param(plan)}
            for revision, plan in runs.plans.items()
        ],
    )


async def _write(conn: AsyncConnection, pk: int, rows: _ImportRows) -> None:
    """Replace whatever the tables hold for the conversation with `rows`,
    keeping the row and its id. The caller holds the row lock."""
    for table, _ in _CHILD_TABLES:
        await conn.execute(text(f"DELETE FROM {table} WHERE conversation = :pk"), {"pk": pk})
    # The canvas has its own writer, and the flat keys write the lease only
    # when the ledger holds one, so both are set here.
    await conn.execute(
        text(
            "UPDATE conversations SET turn_state = '{}'::jsonb, "
            "composition = CAST(:composition AS jsonb), lease_segment_id = NULL, "
            "lease_owner = NULL, lease_expires_at = NULL, lease_cancel = false, seq_next = 0 "
            "WHERE id = :pk"
        ),
        {"pk": pk, "composition": rows.composition},
    )
    await _run_delta_write(conn, pk, rows.delta)
    await conn.execute(
        text("UPDATE conversations SET seq_reserved = seq_next + :margin WHERE id = :pk"),
        {"pk": pk, "margin": IMPORTED_SEQ_MARGIN},
    )
    if rows.runs:
        await conn.execute(
            text(
                "INSERT INTO runs (conversation, run_id, record, submission, retrievals) "
                "VALUES (:pk, :run_id, CAST(:record AS jsonb), CAST(:submission AS jsonb), "
                "CAST(:retrievals AS jsonb))"
            ),
            [{**row, "pk": pk} for row in rows.runs],
        )
    if rows.keys:
        await conn.execute(
            text(
                "INSERT INTO run_keys (conversation, idempotency_key, run_id) "
                "VALUES (:pk, :key, :run_id)"
            ),
            [{**row, "pk": pk} for row in rows.keys],
        )
    if rows.plans:
        await conn.execute(
            text(
                "INSERT INTO run_plans (conversation, revision, plan) "
                "VALUES (:pk, :revision, CAST(:plan AS jsonb))"
            ),
            [{**row, "pk": pk} for row in rows.plans],
        )


def _distribution(label: str, sizes: list[int]) -> str:
    """`label`'s p50, p95 and max of `sizes`, by nearest rank."""
    if not sizes:
        return f"{label}: none"
    ordered = sorted(sizes)

    def rank(q: float) -> int:
        return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]

    return (
        f"{label}: p50 {rank(0.5)}, p95 {rank(0.95)}, max {ordered[-1]} bytes over {len(ordered)}"
    )


@dataclass
class ImportReport:
    """What `import_conversations` did, or would do, to each conversation, by
    `principal/conversation_id`, with the spec blocks it converted in the
    ledger sessions and, apart, in the open turns' sessions, and the sizes
    it read (module docstring)."""

    apply: bool
    imported: list[str] = field(default_factory=list)
    reimported: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)
    refused: dict[str, list[str]] = field(default_factory=dict)
    findings: dict[str, list[str]] = field(default_factory=dict)
    open_turn_findings: dict[str, list[str]] = field(default_factory=dict)
    converted_blocks: int = 0
    unconvertible_blocks: int = 0
    open_turn_converted_blocks: int = 0
    open_turn_unconvertible_blocks: int = 0
    state_sizes: list[int] = field(default_factory=list)
    open_turn_spec_sizes: list[int] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.refused and not self.conflicts

    def render(self) -> str:
        would = "" if self.apply else "would be "
        lines = [
            f"{label}: {name}"
            for label, names in (
                (f"{would}imported", self.imported),
                (f"{would}re-imported", self.reimported),
                ("unchanged", self.unchanged),
                (
                    "conflict, changed on both sides (--prefer-adk re-imports, "
                    "--keep-tables keeps the tables' copy)",
                    self.conflicts,
                ),
                (f"tables' copy {would}kept (--keep-tables)", self.kept),
                (
                    f"{would}discarded (--discard): the tables hold none of it, and its next "
                    "turn starts it afresh",
                    self.discarded,
                ),
            )
            for name in names
        ]
        for name, reasons in self.refused.items():
            lines.append(f"refused: {name}")
            lines += [f"  {reason}" for reason in reasons]
        for name, found in self.findings.items():
            lines.append(f"copied as stored, which the turn loader does not read whole: {name}")
            lines += [f"  {finding}" for finding in found]
        for name, found in self.open_turn_findings.items():
            lines.append(f"open turn blocks the release reads as stored: {name}")
            lines += [f"  {finding}" for finding in found]
        lines += [
            f"spec blocks with mapping-form _bound_refs: {self.converted_blocks} converted, "
            f"{self.unconvertible_blocks} unconvertible",
            "open turn spec blocks with mapping-form _bound_refs: "
            f"{self.open_turn_converted_blocks} converted, "
            f"{self.open_turn_unconvertible_blocks} unconvertible",
            _distribution("ledger state", self.state_sizes),
            _distribution("open turn orchestrator:specs", self.open_turn_spec_sizes),
            f"{len(self.imported)} {would}imported, {len(self.reimported)} {would}re-imported, "
            f"{len(self.unchanged)} unchanged, {len(self.conflicts)} conflicts, "
            f"{len(self.kept)} kept, {len(self.discarded)} discarded, "
            f"{len(self.refused)} refused",
        ]
        return "\n".join(lines)


type _Outcome = Literal[
    "imported", "reimported", "unchanged", "conflict", "live", "kept", "discarded", "refused"
]

# Why `--keep-tables` or `--discard` refuses a conversation it names.
_KEEP_ABSENT = (
    "--keep-tables keeps the tables' copy, and the tables hold none of this conversation, "
    "so keeping it would discard it: --discard gives it up and the report and the gate count it"
)
_DISCARD_HELD = (
    "--discard gives up a refused conversation the tables do not hold, and the tables hold "
    "this one: --keep-tables keeps their copy"
)


@dataclass(frozen=True)
class _Copy:
    """What `_import_one` decided for one conversation: the ledger session
    it read and its mapping, the outcome, the reasons of a refusal, whether
    the tables held the conversation before the copy, and the open turn's
    session when it was asked to read one and the conversation has one."""

    session: _Session
    ledger: Ledger
    outcome: _Outcome
    reasons: list[str]
    held: bool
    open_turn: _TurnSpecs | None = None


class _Undo(Exception):  # noqa: N818 -- names what raising it does
    """Ends `_import_one`'s transaction without committing what it wrote."""


# Records that the operator keeps the tables' copy of a conversation at the
# ledger session's state as it stands (`--keep-tables`). `imported_digest`
# is cleared, so a later ADK move meets the rows as changed and is recorded
# as a conflict, which leaves the kept rows in place.
_KEEP = text(
    "INSERT INTO adk_ledger_imports "
    "(principal, conversation_id, source_digest, source_updated_at, kept_tables) "
    "VALUES (:principal, :conversation_id, :source, :updated, true) "
    "ON CONFLICT (principal, conversation_id) DO UPDATE SET "
    "source_digest = EXCLUDED.source_digest, "
    "source_updated_at = EXCLUDED.source_updated_at, imported_digest = NULL, "
    "diverged = false, kept_tables = true"
)


async def _keep(
    conn: AsyncConnection, pk: int | None, *, created: bool, params: dict[str, Any]
) -> None:
    """Record the kept copy (`_KEEP`) and drop the row this transaction
    inserted to lock, so keeping a conversation the tables lack adds none."""
    if created:
        await conn.execute(text("DELETE FROM conversations WHERE id = :pk"), {"pk": pk})
    await conn.execute(_KEEP, params)


async def _import_one(
    engine: AsyncEngine,
    principal: str,
    conversation_id: str,
    *,
    apply: bool,
    prefer_adk: frozenset[str] = frozenset(),
    keep_tables: frozenset[str] = frozenset(),
    discard: frozenset[str] = frozenset(),
    force: bool = False,
    open_turn: bool = False,
) -> _Copy | None:
    """Decide, and under `apply` write, one conversation. None when ADK holds
    no ledger session for it. With `open_turn`, the open turn's ADK session
    is read too, inside the same transaction, so a failure to read it
    raises before anything is committed.

    The outcome is `refused` when the ledger holds a structural fault or the
    rows do not build (`_import_rows`), `unchanged` when the import record
    holds this ADK state, `imported` when the tables lack the conversation,
    `reimported` when its rows are as last imported or `prefer_adk` names
    it, `live` when `prefer_adk` names it and a stream holds its lease
    (unless `force`), and `conflict` otherwise. It is `kept` when the import
    record keeps the tables' copy of this ADK state and `prefer_adk` does
    not name it, and when `keep_tables` names a conversation the tables
    hold that would be refused or in conflict: the record then keeps the
    tables' copy (`_KEEP`) and the rows are left as they are. It is
    `discarded` where `kept` would be but the tables do not hold the
    conversation, when `discard` names a refused conversation the tables do
    not hold, or the record kept one. `keep_tables` naming a conversation the
    tables do not hold or any other conversation, or `discard` naming any
    other conversation, refuses it, with nothing written.

    Under `apply` the conversation's row is locked, or inserted, before the
    ledger session, the import record and the rows' digest are read, so a
    concurrent writer, the serving release or another instance copying the
    same conversation, either commits first and is seen, or waits, and
    copies serialised on the lock read ADK in the order they commit. A
    conflict records the session's `update_time` and marks the record
    `diverged`, its digests left as they were, so the job and `--verify`
    decide by digest. A refusal writes nothing, a row inserted for it
    included. A dry run takes no lock, writes nothing, and builds the rows
    the write would write, so it refuses what the write would refuse.
    """
    names = {"principal": principal, "conversation_id": conversation_id}
    session: _Session | None = None
    ledger: Ledger | None = None
    outcome: _Outcome = "refused"
    reasons: list[str] = []
    held = False
    turn: _TurnSpecs | None = None
    try:
        async with engine.begin() as conn:
            pk = await _find_row(conn, names, lock=apply)
            created = False
            if pk is None and apply:
                pk = await _insert_row(conn, names)
                created = pk is not None
                if pk is None:
                    pk = await _find_row(conn, names, lock=True)
            held = pk is not None and not created
            # Read under the row lock, so copies the lock serialises read
            # ADK in the order they commit.
            session = await _read_session(conn, principal, conversation_id)
            if session is None:
                raise _Undo
            ledger = read_ledger(session.state)
            source = _digest(session.state)
            updated = {**names, "updated": session.updated_at}
            kept = {**updated, "source": source}
            keep = session.name in keep_tables
            drop = session.name in discard
            prefer = session.name in prefer_adk
            # The tables hold no row of the conversation but the one this
            # transaction inserted to lock.
            absent = pk is None or created
            previous = await _import_record(conn, names)
            kept_here = (
                previous is not None and previous.kept_tables and previous.source_digest == source
            )
            if kept_here and not prefer:
                if drop and not absent:
                    reasons = [_DISCARD_HELD]
                    raise _Undo
                # The kept copy stands while ADK's state is the one kept.
                outcome = "discarded" if absent else "kept"
                if apply:
                    await _keep(conn, pk, created=created, params=kept)
                return _Copy(session, ledger, outcome, reasons, held, turn)
            rows: _ImportRows | None = None
            if ledger.faults:
                reasons = ledger.faults
            else:
                try:
                    rows = _import_rows(ledger)
                # Whatever building the rows raises refuses the conversation,
                # before any outcome is decided, so a refusal is recorded as a
                # refusal.
                except Exception as exc:
                    reasons = [f"the import failed: {type(exc).__name__}: {exc}"]
            if rows is None:
                if keep and absent:
                    reasons = [*reasons, _KEEP_ABSENT]
                elif drop and not absent:
                    reasons = [*reasons, _DISCARD_HELD]
                if not (keep and not absent) and not (drop and absent):
                    raise _Undo
                outcome, reasons = ("discarded" if drop else "kept"), []
                if apply:
                    await _keep(conn, pk, created=created, params=kept)
                return _Copy(session, ledger, outcome, reasons, held, turn)
            if open_turn:
                turn = await _read_open_turn(conn, session, ledger)
            if (
                not created
                and previous is not None
                and previous.source_digest == source
                and not (kept_here and prefer)
            ):
                if apply:
                    await conn.execute(
                        text(
                            "UPDATE adk_ledger_imports SET source_updated_at = :updated "
                            "WHERE principal = :principal AND conversation_id = :conversation_id"
                        ),
                        updated,
                    )
                outcome = "unchanged"
            elif pk is None or created:
                outcome = "imported"
            elif (
                previous is not None and await _stored_digest(conn, pk) == previous.imported_digest
            ):
                outcome = "reimported"
            elif prefer:
                outcome = "reimported"
                if not force and await _lease_live(conn, pk):
                    outcome = "live"
            elif keep:
                outcome = "kept"
                if apply:
                    await _keep(conn, pk, created=created, params=kept)
            else:
                outcome = "conflict"
                if apply:
                    await conn.execute(
                        text(
                            "INSERT INTO adk_ledger_imports "
                            "(principal, conversation_id, source_updated_at, diverged) "
                            "VALUES (:principal, :conversation_id, :updated, true) "
                            "ON CONFLICT (principal, conversation_id) DO UPDATE SET "
                            "source_updated_at = EXCLUDED.source_updated_at, diverged = true, "
                            "kept_tables = false"
                        ),
                        updated,
                    )
            if keep and outcome != "kept":
                reasons = [
                    "--keep-tables keeps the tables' copy of a refused or diverged "
                    f"conversation, and this one is {outcome}"
                ]
                outcome = "refused"
                raise _Undo
            if drop:
                reasons = [
                    "--discard gives up a refused conversation the tables do not hold, "
                    f"and this one is {outcome}"
                ]
                outcome = "refused"
                raise _Undo
            if outcome not in ("imported", "reimported"):
                return _Copy(session, ledger, outcome, reasons, held, turn)
            if apply:
                assert pk is not None
                await _write(conn, pk, rows)
                await conn.execute(
                    text(
                        "INSERT INTO adk_ledger_imports (principal, conversation_id, "
                        "source_digest, source_updated_at, imported_digest) "
                        "VALUES (:principal, :conversation_id, :source, :updated, :imported) "
                        "ON CONFLICT (principal, conversation_id) DO UPDATE SET "
                        "source_digest = EXCLUDED.source_digest, "
                        "source_updated_at = EXCLUDED.source_updated_at, "
                        "imported_digest = EXCLUDED.imported_digest, diverged = false, "
                        "kept_tables = false, imported_at = now()"
                    ),
                    {**updated, "source": source, "imported": await _stored_digest(conn, pk)},
                )
    except _Undo:
        if session is None:
            return None
    assert session is not None and ledger is not None
    return _Copy(session, ledger, outcome, reasons, held, turn)


async def _read_open_turn(
    conn: AsyncConnection, session: _Session, ledger: Ledger
) -> _TurnSpecs | None:
    """The open turn's ADK session, None when the ledger has no open turn or
    its session does not exist."""
    turn_index = ledger.flat.get(_KEY_TURN_INDEX)
    if not turn_index or ledger.flat.get(_KEY_TURN_RECORDED):
        return None
    return await _read_turn_session(
        conn, session.principal, f"{session.conversation_id}#t{turn_index}"
    )


def _count_open_turn(copy: _Copy, report: ImportReport) -> None:
    """Add the open turn's session, if the conversation has one, to the
    report: its size, its converted blocks, and each block that does not
    convert, under the conversation's open turn findings."""
    turn = copy.open_turn
    if turn is None:
        return
    report.open_turn_spec_sizes.append(turn.size)
    report.open_turn_converted_blocks += turn.blocks.converted
    report.open_turn_unconvertible_blocks += len(turn.blocks.unconvertible)
    if turn.blocks.unconvertible:
        report.open_turn_findings[copy.session.name] = [
            f"open turn {finding}" for finding in turn.blocks.unconvertible
        ]


async def import_conversations(
    engine: AsyncEngine,
    *,
    apply: bool,
    prefer_adk: frozenset[str] = frozenset(),
    keep_tables: frozenset[str] = frozenset(),
    discard: frozenset[str] = frozenset(),
    force: bool = False,
) -> ImportReport:
    """Import every ledger session (module docstring). With `apply` false,
    read and decide only. `prefer_adk` holds `principal/conversation_id`
    names to re-import over rows changed on both sides. One a stream holds
    is refused unless `force`. `keep_tables` holds names of refused or
    diverged conversations whose tables' copy the operator keeps, and
    `discard` names of refused conversations the tables do not hold that the
    operator gives up.

    Raises:
        AdkSchemaError: ADK's schema is not the one the job reads.
        NamedTwice: two of `prefer_adk`, `keep_tables` and `discard` share a
            name.
        UnknownConversation: `prefer_adk`, `keep_tables` or `discard` names
            a conversation ADK holds no ledger session for. Nothing is
            written.
    """
    flags = (("--prefer-adk", prefer_adk), ("--keep-tables", keep_tables), ("--discard", discard))
    for index, (flag, names) in enumerate(flags):
        for other, other_names in flags[index + 1 :]:
            if twice := sorted(names & other_names):
                raise NamedTwice(f"{flag} and {other} both name: {', '.join(twice)}")
    async with engine.connect() as conn:
        await _check_adk_schema(conn)
    keys = await _ledger_keys(engine)
    known = {key.name for key in keys}
    for flag, names in flags:
        if unknown := sorted(names - known):
            raise UnknownConversation(f"{flag} names no ledger session: {', '.join(unknown)}")
    await apply_pending(engine)
    report = ImportReport(apply=apply)
    for key in keys:
        try:
            copy = await _import_one(
                engine,
                key.principal,
                key.conversation_id,
                apply=apply,
                prefer_adk=prefer_adk,
                keep_tables=keep_tables,
                discard=discard,
                force=force,
                open_turn=True,
            )
        # One conversation's failure refuses it, with nothing written. The
        # run goes on with the rest.
        except Exception as exc:
            report.refused[key.name] = [f"the import failed: {type(exc).__name__}: {exc}"]
            continue
        if copy is None:
            continue
        report.state_sizes.append(key.size)
        report.converted_blocks += copy.ledger.blocks.converted
        report.unconvertible_blocks += len(copy.ledger.blocks.unconvertible)
        if copy.ledger.findings:
            report.findings[key.name] = list(copy.ledger.findings)
        _count_open_turn(copy, report)
        if copy.outcome == "refused":
            report.refused[key.name] = copy.reasons
            report.findings.pop(key.name, None)
        elif copy.outcome == "live":
            report.refused[key.name] = [
                "--prefer-adk would end the stream that holds its lease: rerun once the "
                "stream has ended, or add --force"
            ]
        else:
            if copy.outcome in ("kept", "discarded"):
                # Nothing is copied, so nothing is copied as stored.
                report.findings.pop(key.name, None)
            {
                "imported": report.imported,
                "reimported": report.reimported,
                "unchanged": report.unchanged,
                "conflict": report.conflicts,
                "kept": report.kept,
                "discarded": report.discarded,
            }[copy.outcome].append(key.name)
    return report


# One conversation's ledger session against what was last copied of it. No
# row comes back when ADK holds no ledger session for it. `current` is false
# when the session's `update_time` is not the one last recorded, or nothing
# is recorded. `diverged` is true when the last copy found both sides
# changed. One lookup of `sessions` by its primary key, joined to
# `adk_ledger_imports` by its own.
_ADK_MOVED = text(
    "SELECT i.source_updated_at IS NOT DISTINCT FROM s.update_time AS current, "
    "coalesce(i.diverged, false) AS diverged, s.update_time "
    "FROM sessions s LEFT JOIN adk_ledger_imports i "
    "ON i.principal = s.user_id AND i.conversation_id = s.id "
    "WHERE s.app_name = :app AND s.user_id = :principal AND s.id = :conversation_id"
)


def reimport_on_access(engine: AsyncEngine) -> AdkBridge:
    """`StudioStore`'s `AdkBridge` for the release that serves Studio from
    the tables while the previous release finishes turns on ADK.

    `AdkBridge.moved` says whether ADK's ledger session for a conversation moved
    since it was last copied or refused, or the conversation is diverged
    (`_ADK_MOVED`), so a `/turn` can tell a per-turn session the previous
    release just created from a leftover one. `AdkBridge.convert_specs`
    converts the spec blocks a turn session holds as the import does
    (`_converted_specs`), for a load of an open turn and a stream that
    resumes a turn the previous release opened, and logs each block that
    does not convert once (`_AdkOnAccess._log_unconvertible`).

    Before each load it reads the conversation's ledger session against
    what was last copied of it (`_ADK_MOVED`). When ADK holds no ledger
    session for it (a conversation started on this release) or the session
    has not moved, that one read is all it does. Otherwise it copies the
    conversation through the job's own path (`_import_one`), under the row's
    lock. A conversation only ADK changed is re-imported in place and one
    never imported is imported.

    A conversation that changed on both sides is diverged: an error names
    it and the command that takes ADK's copy, and every load of it raises
    `ConversationHeld` until that command runs. The tables lack what ADK
    holds, so serving them would lose the previous release's turns. A
    conversation whose ledger session the copy refuses (`_Copy.reasons`)
    is logged as an error once per ADK revision in this process and served
    as the tables hold it. When they do not hold it, `load` finds no
    conversation and the next `/turn` starts it afresh. Its later reads
    find the row that turn created, refuse the copy again and serve the
    tables, so no conflict follows. Any other failure of the copy
    propagates, so the request fails before it writes and its retry copies
    again.
    """
    return _AdkOnAccess(engine)


class _AdkOnAccess:
    """`reimport_on_access`'s `AdkBridge`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        # Each refused conversation, with the ledger session's `update_time` the
        # refusal read.
        self._refused: dict[tuple[str, str], datetime] = {}
        # The unconvertible spec blocks this bridge has logged, newest last,
        # by the reason logged, which names the turn session and spec id.
        self._logged: OrderedDict[str, None] = OrderedDict()

    async def _lookup(self, principal: str, conversation_id: str) -> Row[Any] | None:
        async with self._engine.connect() as conn:
            return (
                await conn.execute(
                    _ADK_MOVED,
                    {"app": LEDGER_APP, "principal": principal, "conversation_id": conversation_id},
                )
            ).first()

    async def moved(self, principal: str, conversation_id: str) -> bool:
        row = await self._lookup(principal, conversation_id)
        if row is None:
            return False
        # A conflict another instance recorded leaves `current` true at ADK's
        # latest `update_time`, though ADK moved after this instance's load.
        return row.diverged or (
            not row.current and self._refused.get((principal, conversation_id)) != row.update_time
        )

    def convert_specs(self, session_id: str, specs: dict[str, Any]) -> dict[str, Any]:
        blocks = Blocks()
        converted = _converted_specs(session_id, specs, blocks)
        for reason in blocks.unconvertible:
            self._log_unconvertible(reason)
        return converted

    def _log_unconvertible(self, reason: str) -> None:
        """Log a spec block served as stored, since it does not convert, once
        while it stays among the `_LOGGED_BLOCKS_KEPT` this bridge last
        logged. `reason` names the turn session (conversation and turn)
        holding it, its spec id and why. A turn that closes with it loses
        its entity refs, since the reader refuses the mapping form."""
        if reason in self._logged:
            self._logged.move_to_end(reason)
            return
        self._logged[reason] = None
        if len(self._logged) > _LOGGED_BLOCKS_KEPT:
            self._logged.popitem(last=False)
        logger.warning("adk import: a spec block is served as stored: %s", reason)

    async def reimport(self, principal: str, conversation_id: str) -> None:
        row = await self._lookup(principal, conversation_id)
        if row is not None and row.diverged:
            raise _diverged(principal, conversation_id)
        if row is None or row.current:
            return
        key = (principal, conversation_id)
        if self._refused.get(key) == row.update_time:
            return
        copy = await _import_one(self._engine, principal, conversation_id, apply=True)
        if copy is None:
            return
        if copy.outcome == "conflict":
            raise _diverged(principal, conversation_id)
        if copy.outcome != "refused":
            self._refused.pop(key, None)
            if copy.outcome not in ("unchanged", "kept"):
                logger.info("adk import: %s %s on access", copy.session.name, copy.outcome)
            return
        self._refused[key] = copy.session.updated_at
        logger.error(
            "adk import: %s moved on ADK and is refused, so %s: %s",
            copy.session.name,
            "it is served as the tables hold it"
            if copy.held
            else "the tables do not hold it and its next turn starts it afresh",
            "; ".join(copy.reasons),
        )


def _diverged(principal: str, conversation_id: str) -> ConversationHeld:
    """Log that the conversation is diverged, with the command that repairs
    it, and return the refusal `reimport_on_access` raises for it."""
    name = f"{principal}/{conversation_id}"
    logger.error(
        "adk import: %s changed on ADK and in the tables since it was imported, so it is not "
        "served until `python -m harness.adk_import import --apply --prefer-adk %s` takes "
        "ADK's copy or `python -m harness.adk_import import --apply --keep-tables %s` keeps "
        "the tables' copy",
        name,
        name,
        name,
    )
    return ConversationHeld(conversation_id)


@dataclass
class _NoAdk:
    """The job's stores read the tables as they stand and serve no turn.
    `blocks` gathers what `convert_specs` converted and what it could not."""

    blocks: Blocks = field(default_factory=Blocks)

    async def reimport(self, principal: str, conversation_id: str) -> None:
        del principal, conversation_id

    async def moved(self, principal: str, conversation_id: str) -> bool:
        del principal, conversation_id
        return False

    def convert_specs(self, session_id: str, specs: dict[str, Any]) -> dict[str, Any]:
        return _converted_specs(session_id, specs, self.blocks)


def _store(engine: AsyncEngine) -> StudioStore:
    """A store over the tables that reads an open turn's spec blocks through
    the reader the server serves (`open_turn_specs`), and whose `load` reads
    the tables as they stand. A store keeps every conversation it loads, so
    the job builds one per conversation."""
    return StudioStore(engine, turn_specs=open_turn_specs(engine), adk=_NoAdk())


def _conversation_view(state: ConversationState) -> dict[str, Any]:
    """Every durable field of `state`, its lease bar
    `_UNCOMPARED_LEASE_FIELD`, its canvas, the order of its components, and
    the open turn's spec blocks, as JSON."""
    durable = _serialize_durable(state)
    if (segment := durable[_KEY_SEGMENT]) is not None:
        durable[_KEY_SEGMENT] = {
            key: value for key, value in segment.items() if key != _UNCOMPARED_LEASE_FIELD
        }
    return {
        **durable,
        "composition": state.composition.model_dump(mode="json") if state.composition else None,
        "component order": list(state.components),
        "open turn spec blocks": state.turn_specs,
    }


def _run_view(runs: RunItems) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for run_id, record in runs.records.items():
        view[f"run {run_id}"] = [
            record,
            runs.submissions.get(run_id),
            runs.retrievals.get(run_id),
        ]
    for key, run_id in runs.keys.items():
        view[f"idempotency key {key}"] = run_id
    for revision, plan in runs.plans.items():
        view[f"run plan {revision}"] = plan
    return view


async def _stored_runs(engine: AsyncEngine, principal: str, conversation_id: str) -> RunItems:
    runs = RunItems()
    scope = (
        "JOIN conversations c ON c.id = t.conversation "
        "WHERE c.principal = :principal AND c.conversation_id = :conversation_id"
    )
    params = {"principal": principal, "conversation_id": conversation_id}
    async with engine.connect() as conn:
        for run_id, record, submission, retrievals in await conn.execute(
            text(f"SELECT run_id, record, submission, retrievals FROM runs t {scope}"), params
        ):
            runs.records[run_id] = record
            if submission is not None:
                runs.submissions[run_id] = submission
            if retrievals is not None:
                runs.retrievals[run_id] = retrievals
        for key, run_id in await conn.execute(
            text(f"SELECT idempotency_key, run_id FROM run_keys t {scope}"), params
        ):
            runs.keys[key] = run_id
        for revision, plan in await conn.execute(
            text(f"SELECT revision, plan FROM run_plans t {scope}"), params
        ):
            runs.plans[revision] = plan
    return runs


async def _read_turn_specs(engine: AsyncEngine, principal: str, state: ConversationState) -> None:
    """Set the open turn's spec blocks on `state` from its turn session,
    through the reader and the bridge the tables side's `StudioStore.load`
    uses (`open_turn_specs`, `_NoAdk.convert_specs`)."""
    if state.turn_index and not state.turn_recorded:
        session_id = state.session_id
        stored = await open_turn_specs(engine)(principal, session_id)
        state.turn_specs = _NoAdk().convert_specs(session_id, stored)


# The previous release's run-ledger prefixes, each with the `RunItems` field
# its `StudioStore.run_items` read it into.
_PREVIOUS_RUN_PREFIXES = (
    ("studio:run:", "records"),
    ("studio:run_key:", "keys"),
    ("studio:run_submission:", "submissions"),
    ("studio:run_retrievals:", "retrievals"),
    ("studio:run_plan:", "plans"),
)


def _previous_release_load(conversation_id: str, state: dict[str, Any]) -> ConversationState:
    """The conversation as the previous release loaded its ledger session's
    `state` (its `studio_state._deserialize_durable` and `_apply_delta`),
    with each value's spec blocks converted as the import converts them.
    The key mapping is that release's own, kept here apart from
    `read_ledger`, so `verify` compares the import's mapping against it. A
    `studio:` key it does not name is ignored, as that release ignored one.
    The value parsers are this release's, which read the same shapes."""
    loaded = ConversationState(conversation_id=conversation_id)
    blocks = Blocks()
    delta = {
        key: _convert_blocks(value, key, blocks)
        for key, value in state.items()
        if key.startswith(_ADK_PREFIX)
    }
    for key, value in delta.items():
        if key == "studio:turn_index":
            loaded.turn_index = value
        elif key == "studio:seq":
            loaded.seq = value
        elif key == "studio:trace_id":
            loaded.trace_id = value
        elif key == "studio:pending":
            loaded.pending = _deserialize_pending(value)
        elif key == "studio:plan_version_ids":
            loaded.plan_version_ids = {int(version): v for version, v in value.items()}
        elif key == "studio:latest_plan_version":
            loaded.latest_plan_version = value
        elif key == "studio:composition":
            loaded.composition = (
                CompositionManifest.model_validate(value) if value is not None else None
            )
        elif key == "studio:segment":
            loaded.segment = _deserialize_segment(value)
        elif key == "studio:auto_add":
            loaded.auto_add = value
        elif key == "studio:timed_out_calls":
            loaded.timed_out_calls = set(value)
        elif key == "studio:turn_user_message":
            loaded.turn_user_message = value
        elif key == "studio:turn_user_words":
            loaded.turn_user_words = value
        elif key == "studio:turn_final_text":
            loaded.turn_final_text = value
        elif key == "studio:turn_components":
            loaded.turn_components = list(value)
        elif key == "studio:turn_recorded":
            loaded.turn_recorded = value
        elif key == "studio:turn_plans":
            loaded.turn_plans = list(value)
        elif key == "studio:turn_readings":
            loaded.turn_readings = list(value)
        elif key == "studio:turn_applied_choice":
            loaded.turn_applied_choice = dict(value) if value else None
        elif key == "studio:turn_choice_baseline":
            loaded.turn_choice_baseline = dict(value or {})
        elif key.startswith("studio:turn:"):
            _set_list_item(loaded.turns, int(key.removeprefix("studio:turn:")), value)
        elif key.startswith("studio:component:"):
            loaded.components[key.removeprefix("studio:component:")] = value
        elif key.startswith("studio:write_token:"):
            loaded.write_tokens[key.removeprefix("studio:write_token:")] = value
        elif key.startswith("studio:approval:"):
            index = int(key.removeprefix("studio:approval:"))
            _set_list_item(loaded.approvals, index, ApprovalRecord.model_validate(value))
    if "studio:turn_usage" in delta:
        loaded.turn_usage = _turn_usage(delta["studio:turn_usage"], trace_id=loaded.trace_id)
    elif loaded.turn_usage.trace_id != loaded.trace_id:
        loaded.turn_usage = TurnUsage(trace_id=loaded.trace_id)
    return loaded


def _previous_release_runs(state: dict[str, Any]) -> RunItems:
    """The run-ledger items of a ledger session's `state` as the previous
    release's `StudioStore.run_items` read them, by prefix."""
    runs = RunItems()
    for key, value in state.items():
        for prefix, items in _PREVIOUS_RUN_PREFIXES:
            if key.startswith(prefix):
                getattr(runs, items)[key.removeprefix(prefix)] = value
                break
    return runs


async def _adk_view(engine: AsyncEngine, session: _Session) -> dict[str, Any]:
    """The conversation and its run items as the previous release loaded the
    ledger session (`_previous_release_load`, `_previous_release_runs`), its
    spec blocks converted as the import converts them."""
    state = _previous_release_load(session.conversation_id, session.state)
    await _read_turn_specs(engine, session.principal, state)
    return _conversation_view(state) | _run_view(_previous_release_runs(session.state))


async def _table_view(engine: AsyncEngine, session: _Session) -> dict[str, Any] | None:
    """The conversation and its run rows as `StudioStore.load` reads them
    from the tables. None when the tables do not hold it."""
    state = await _store(engine).load(session.principal, session.conversation_id)
    if state is None:
        return None
    runs = await _stored_runs(engine, session.principal, session.conversation_id)
    return _conversation_view(state) | _run_view(runs)


async def _outcome(view: Any) -> tuple[str, Any]:
    """`("view", ...)` for the view `view` awaits, or `("raises", ...)`
    naming the exception that stopped it, so a load this release fails on
    compares equal to the same failure."""
    try:
        return "view", await view
    # A failed load is an outcome to compare, whatever it raised.
    except Exception as exc:
        return "raises", f"{type(exc).__name__}: {exc}"


@dataclass
class VerifyReport:
    """For every ledger session, by `principal/conversation_id`, the keys
    whose value differs between ADK's load and the tables' load. Three
    kinds are listed apart and not compared. One is a conversation whose
    ADK state moved since the import while its rows stayed as imported,
    written by the previous release, which serves until cutover. The serving
    release re-imports it on its next read. Another is a conversation whose
    rows the serving release has written since the import, so its tables
    hold turns ADK never saw. The third is a conflict, where both moved,
    and it fails the verify. A conversation whose tables' copy the operator
    kept (`--keep-tables`) at ADK's state as it stands is listed apart too.
    A compared, written or kept conversation counts as checked. The verify
    fails on a difference or a conflict, and when the ledger holds sessions
    and none of them was checked.

    `streaming_on_adk` names each ledger session whose `studio:segment`
    lease is live on the database's clock: a stream of the previous
    revision still runs on it and can write it again. It is reported and
    does not fail the verify; `close_open_turns` fails on a live lease of a
    turn on ADK."""

    total: int = 0
    differences: dict[str, list[str]] = field(default_factory=dict)
    moved_on_adk: list[str] = field(default_factory=list)
    written_since_import: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    streaming_on_adk: list[str] = field(default_factory=list)

    @property
    def compared(self) -> int:
        return (
            self.total
            - len(self.moved_on_adk)
            - len(self.written_since_import)
            - len(self.conflicts)
            - len(self.kept)
        )

    @property
    def checked(self) -> int:
        return self.compared + len(self.written_since_import) + len(self.kept)

    @property
    def clean(self) -> bool:
        return not self.differences and not self.conflicts and (self.checked > 0 or self.total == 0)

    def render(self) -> str:
        compared = self.compared
        lines = [f"differs: {name}: {', '.join(keys)}" for name, keys in self.differences.items()]
        lines += [
            f"moved on ADK since the import; re-imported on access: {name}"
            for name in self.moved_on_adk
        ]
        lines += [
            f"written by the tables' release since the import, not compared: {name}"
            for name in self.written_since_import
        ]
        lines += [
            "conflict, changed on both sides since the import (--prefer-adk re-imports, "
            f"--keep-tables keeps the tables' copy): {name}"
            for name in self.conflicts
        ]
        lines += [f"tables' copy kept (--keep-tables), not compared: {name}" for name in self.kept]
        lines += [f"a stream holds its lease on ADK: {name}" for name in self.streaming_on_adk]
        if self.streaming_on_adk:
            count = len(self.streaming_on_adk)
            lines.append(
                f"{count} {'stream holds its' if count == 1 else 'streams hold their'} lease on ADK"
            )
        if self.checked == 0 and self.total > 0:
            lines.append(f"none of the {self.total} ledger sessions compared, written or kept")
        lines.append(f"{compared - len(self.differences)} of {compared} equal")
        return "\n".join(lines)


async def _tables_side(
    engine: AsyncEngine, session: _Session
) -> Literal["as imported", "moved on adk", "written", "conflict", "kept"]:
    """How the conversation stands against the import: `as imported` when
    its rows are what the job wrote from ADK's state as it stands, or the
    job has not imported it; `moved on adk` when the rows are as imported
    and ADK's state moved since; `written` when the serving release wrote
    the rows since; `conflict` when it did and ADK's state also moved;
    `kept` when the operator kept the tables' copy at ADK's state as it
    stands."""
    adk_digest = _digest(session.state)
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT c.id, i.imported_digest, i.source_digest, i.kept_tables "
                    "FROM adk_ledger_imports i LEFT JOIN conversations c "
                    "USING (principal, conversation_id) "
                    "WHERE principal = :principal AND conversation_id = :conversation_id"
                ),
                {"principal": session.principal, "conversation_id": session.conversation_id},
            )
        ).first()
        if row is not None and row.kept_tables and row.source_digest == adk_digest:
            return "kept"
        if row is None or row.id is None:
            return "as imported"
        rows_moved = await _stored_digest(conn, row.id) != row.imported_digest
    adk_moved = adk_digest != row.source_digest
    if rows_moved:
        return "conflict" if adk_moved else "written"
    return "moved on adk" if adk_moved else "as imported"


async def verify(engine: AsyncEngine) -> VerifyReport:
    """Compare every ledger session's load with the tables' (module docstring)."""
    await _check_tables(engine)
    report = VerifyReport()
    async with engine.connect() as conn:
        streaming = await conn.execute(_STREAMING_ON_ADK, {"app": LEDGER_APP})
        report.streaming_on_adk = [f"{principal}/{cid}" for principal, cid in streaming]
    async for session in _ledger_sessions(engine, await _ledger_keys(engine)):
        report.total += 1
        side = await _tables_side(engine, session)
        if side == "as imported" and (differences := await _differences(engine, session)):
            # A write the serving release commits between the classification
            # and the compared reads shows as a difference. The rows' digest,
            # read again, tells the two apart.
            side = await _tables_side(engine, session)
            if side == "as imported":
                report.differences[session.name] = differences
        if side == "moved on adk":
            report.moved_on_adk.append(session.name)
        elif side == "written":
            report.written_since_import.append(session.name)
        elif side == "conflict":
            report.conflicts.append(session.name)
        elif side == "kept":
            report.kept.append(session.name)
    return report


# Each ledger session whose `studio:segment` lease (`_serialize_segment`:
# `expires_at` in epoch seconds) is live on the database's clock, as the
# previous revision's `claim_segment` reads a lease.
_STREAMING_ON_ADK = text(
    "SELECT user_id, id FROM sessions WHERE app_name = :app "
    "AND CASE WHEN jsonb_typeof(state -> 'studio:segment' -> 'expires_at') = 'number' "
    "THEN (state -> 'studio:segment' ->> 'expires_at')::double precision END "
    f"> {DATABASE_NOW} "
    "ORDER BY user_id, id"
)


async def _differences(engine: AsyncEngine, session: _Session) -> list[str]:
    """What differs between the conversation as ADK's state loads and as the
    tables load, as the keys whose values differ or how the loads failed."""
    old_kind, old = await _outcome(_adk_view(engine, session))
    new_kind, new = await _outcome(_table_view(engine, session))
    if new_kind == "view" and new is None:
        return ["not in the tables"]
    if old_kind != new_kind or old_kind == "raises":
        return [f"ADK load {old_kind}, tables load {new_kind}"] if old != new else []
    return sorted(key for key in old.keys() | new.keys() if old.get(key, ...) != new.get(key, ...))


@dataclass
class CloseReport:
    """What `close_open_turns` did, or would do, by
    `principal/conversation_id turn n`. A turn a stream holds the lease of
    is `live_on_adk` when it runs on ADK (its ADK turn session exists and
    `turn_cursors` does not hold it) and `live_not_adk` otherwise. A
    conversation whose ledger session holds a live `studio:segment` lease
    (`_STREAMING_ON_ADK`), which a stream of the previous release writes
    there and never to the tables, is `live_on_adk` by
    `principal/conversation_id`, whatever the tables' lease. A conversation
    whose import record is not current (`_STALE_IMPORTS`: ADK wrote its
    ledger session since the import) is `stale_import`, since the tables
    lack what ADK wrote; `import --apply` is run first. Each is left open.
    A turn or conversation live on ADK fails the run, since its stream can
    still end on a pause the tables' turns cannot answer; the run is
    repeated once it ends. A stale import fails it too."""

    apply: bool
    closed: list[str] = field(default_factory=list)
    live_on_adk: list[str] = field(default_factory=list)
    stale_import: list[str] = field(default_factory=list)
    live_not_adk: list[str] = field(default_factory=list)
    not_on_adk: list[str] = field(default_factory=list)
    in_turn_cursors: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    diverged: list[str] = field(default_factory=list)
    refused: dict[str, list[str]] = field(default_factory=dict)
    findings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (
            self.changed or self.diverged or self.refused or self.live_on_adk or self.stale_import
        )

    def render(self) -> str:
        would = "" if self.apply else "would be "
        lines = [
            f"{label}: {name}"
            for label, names in (
                (f"{would}closed as abandoned", self.closed),
                (
                    "left open, a stream on ADK holds its lease, run again once it ends",
                    self.live_on_adk,
                ),
                (
                    "left open, ADK wrote since the import, rerun import --apply first",
                    self.stale_import,
                ),
                ("left open, a stream holds its lease on a turn not on ADK", self.live_not_adk),
                ("left open, no ADK turn session", self.not_on_adk),
                ("left open, turn_cursors holds the turn", self.in_turn_cursors),
                ("changed during the run, run again", self.changed),
                ("left alone, diverged (--prefer-adk or --keep-tables first)", self.diverged),
            )
            for name in names
        ]
        for name, reasons in self.refused.items():
            lines.append(f"left open, refused: {name}")
            lines += [f"  {reason}" for reason in reasons]
        for name, found in self.findings.items():
            outcome = f"{would}closed" if name in self.closed else "left open"
            lines.append(f"{outcome} with blocks as stored, which the turn loader drops: {name}")
            lines += [f"  {finding}" for finding in found]
        lines.append(
            f"{len(self.closed)} {would}closed, {len(self.live_on_adk)} live on ADK, "
            f"{len(self.stale_import)} imported before ADK's last write, "
            f"{len(self.live_not_adk)} live not on ADK, "
            f"{len(self.not_on_adk)} without an ADK turn session, "
            f"{len(self.in_turn_cursors)} in turn_cursors, {len(self.changed)} changed, "
            f"{len(self.diverged)} diverged, {len(self.refused)} refused"
        )
        return "\n".join(lines)


async def close_open_turns(engine: AsyncEngine, *, apply: bool) -> CloseReport:
    """Close every open ADK turn the tables hold (module docstring). With
    `apply` false, decide only. A diverged conversation is left alone and
    listed, since its rows lack what ADK holds. A conversation whose close
    fails is refused, and the run goes on with the rest. A conversation
    a stream holds on ADK's ledger, or whose import is not current, is left
    open (`CloseReport`).

    Raises:
        NextReleaseMissing: `apply` while no `turn_cursors` table exists.
            The next release's first migration creates it, and until that
            release serves, this release answers a turn paused on ADK, which
            the close would end, so a user's answer to it would find no
            pause.
    """
    await _check_tables(engine)
    if apply:
        async with engine.connect() as conn:
            if (await conn.execute(text("SELECT to_regclass('turn_cursors')"))).scalar() is None:
                raise NextReleaseMissing(
                    "close-open-turns --apply runs once the next release serves: no "
                    "turn_cursors table, which that release's first migration creates. Until "
                    "then this release can still answer a turn paused on ADK, and closing it "
                    "would refuse the user's answer. Run the dry run instead."
                )
    report = CloseReport(apply=apply)
    streaming: set[tuple[str, str]] = set()
    stale: set[tuple[str, str]] = set()
    async with engine.connect() as conn:
        if (await conn.execute(text(CUTOVER_ADK_SESSIONS))).scalar_one():
            streaming = {
                (principal, cid)
                for principal, cid in await conn.execute(_STREAMING_ON_ADK, {"app": LEDGER_APP})
            }
            stale = {
                (principal, cid)
                for principal, cid in await conn.execute(_STALE_IMPORTS, {"app": LEDGER_APP})
            }
        conversations = (
            await conn.execute(
                text(
                    # Lease expiry is read on the database's clock, as the
                    # store reads it.
                    "SELECT c.id, c.principal, c.conversation_id, "
                    f"coalesce(c.lease_expires_at > {DATABASE_NOW}, false) AS leased, "
                    "coalesce(i.diverged, false) AS diverged "
                    "FROM conversations c LEFT JOIN adk_ledger_imports i "
                    "USING (principal, conversation_id) ORDER BY principal, conversation_id"
                )
            )
        ).all()
    for pk, principal, conversation_id, leased, diverged in conversations:
        name = f"{principal}/{conversation_id}"
        if diverged:
            report.diverged.append(name)
            continue
        if (principal, conversation_id) in streaming:
            report.live_on_adk.append(name)
            continue
        if (principal, conversation_id) in stale:
            report.stale_import.append(name)
            continue
        try:
            await _close_one(
                engine,
                principal,
                conversation_id,
                pk=pk,
                leased=leased,
                apply=apply,
                report=report,
            )
        # One conversation's failure refuses it. The run goes on with the rest.
        except Exception as exc:
            report.refused[name] = [f"the close failed: {type(exc).__name__}: {exc}"]
    return report


async def _close_one(
    engine: AsyncEngine,
    principal: str,
    conversation_id: str,
    *,
    pk: int,
    leased: bool,
    apply: bool,
    report: CloseReport,
) -> None:
    """Close one conversation's open ADK turn, if it has one, and note what
    became of it in `report`. The open turn's spec blocks are the ones the
    store's `load` read and converted, through the server's reader
    (`open_turn_specs`). `pk` is the conversation's row id. A turn the
    `turn_cursors` table holds is left open. A turn a stream holds is
    left open and sorted by whether it runs on ADK (`CloseReport`)."""
    sessions: set[str] = set()
    bridge = _NoAdk()
    store = StudioStore(engine, turn_specs=open_turn_specs(engine, found=sessions), adk=bridge)
    state = await store.load(principal, conversation_id)
    if state is None or not state.turn_index or state.turn_recorded:
        return
    name = f"{principal}/{conversation_id} turn {state.turn_index}"
    if leased:
        async with engine.connect() as conn:
            held = await _close_refusal(conn, pk, state.turn_index)
        if held == "moved":
            report.changed.append(name)
        elif held is None and state.session_id in sessions:
            report.live_on_adk.append(name)
        else:
            report.live_not_adk.append(name)
        return
    if state.session_id not in sessions:
        report.not_on_adk.append(name)
        return
    turn_index = state.turn_index
    refusal: _Refusal | None = None

    async def closing(conn: AsyncConnection, locked: int) -> bool:
        nonlocal refusal
        refusal = await _close_refusal(conn, locked, turn_index)
        if refusal is None:
            await conn.execute(_CLEAR_LEASE, {"pk": locked})
        return refusal is None

    if not apply:
        async with engine.connect() as conn:
            refusal = await _close_refusal(conn, pk, turn_index)
        if refusal == "moved":
            report.changed.append(name)
            return
        if refusal == "in_turn_cursors":
            report.in_turn_cursors.append(name)
            return
    # A block that does not convert is kept as stored, as `import` keeps one.
    if bridge.blocks.unconvertible:
        report.findings[name] = bridge.blocks.unconvertible
    if not apply:
        report.closed.append(name)
        return
    state.close_turn(final_text=state.turn_final_text, error="abandoned")
    state.pending.clear()
    # No stream holds the lease, so any lease on record has lapsed. Clearing
    # it fences out a stream running under it, whose checkpoint is fenced on
    # the lease id alone and would otherwise write over the turn this closes.
    state.segment = None
    try:
        await store.save(principal, conversation_id, guard=closing)
    except SaveRefusedError:
        if refusal == "moved":
            report.changed.append(name)
        else:
            # Left open, so none of its blocks is closed as stored.
            report.findings.pop(name, None)
            report.in_turn_cursors.append(name)
    except ConversationStale:
        report.changed.append(name)
    else:
        report.closed.append(name)


type _Refusal = Literal["moved", "in_turn_cursors"]

# The closing write leaves no lease on the row, whatever its load read.
_CLEAR_LEASE = text(
    "UPDATE conversations SET lease_segment_id = NULL, lease_owner = NULL, "
    "lease_expires_at = NULL, lease_cancel = false WHERE id = :pk"
)


async def _close_refusal(conn: AsyncConnection, pk: int, turn_index: int) -> _Refusal | None:
    """Why the job must leave turn `turn_index` of the conversation whose
    row id is `pk` open, or `None` when it may close it: `moved` when the
    row holds another turn index, `in_turn_cursors` when `turn_cursors`
    holds the turn. A statement that names a missing table fails, so one
    statement reads the row's turn index and probes `turn_cursors`, and a
    second reads it only when the table exists."""
    probe = (
        await conn.execute(
            text(
                "SELECT to_regclass('turn_cursors') IS NOT NULL AS present, "
                "(turn_state ->> 'turn_index')::int AS stored "
                "FROM conversations WHERE id = :pk"
            ),
            {"pk": pk},
        )
    ).one()
    if probe.stored != turn_index:
        return "moved"
    if not probe.present:
        return None
    found = await conn.execute(
        text("SELECT 1 FROM turn_cursors WHERE conversation = :pk AND turn_index = :turn_index"),
        {"pk": pk, "turn_index": turn_index},
    )
    return "in_turn_cursors" if found.first() is not None else None


# A ledger session's `studio:turn_index`, locked under `--apply`: the
# previous release's claim of a turn writes the index to this row before it
# creates the turn's session.
_LEDGER_TURN_INDEX = (
    "SELECT state -> 'studio:turn_index' AS turn_index FROM sessions "
    "WHERE app_name = :app AND user_id = :principal AND id = :id"
)

# Each of a conversation's ADK turn sessions `{conversation_id}#t{n}`
# numbered above `:turn_index`.
_TURN_SESSIONS_AHEAD = text(
    "SELECT id FROM sessions WHERE app_name = :app AND user_id = :principal "
    "AND left(id, length(:id) + 2) = :id || '#t' "
    "AND substr(id, length(:id) + 3) ~ '^[0-9]{1,9}$' "
    "AND substr(id, length(:id) + 3)::integer > :turn_index "
    "ORDER BY substr(id, length(:id) + 3)::integer"
)


@dataclass
class RollbackReport:
    """What `rollback_turn_sessions` deleted, or would delete: each
    conversation's ADK turn sessions numbered above its ledger session's
    `studio:turn_index`, by `principal/conversation_id`, and each ledger
    session whose index is not a whole number, left alone. Clean when it
    found nothing to delete and no index it cannot read."""

    apply: bool
    ahead: dict[str, list[str]] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.ahead and not self.unreadable

    def render(self) -> str:
        would = "" if self.apply else "would be "
        lines: list[str] = []
        for name, sessions in self.ahead.items():
            lines.append(f"{would}deleted, above the ledger's turn index: {name}")
            lines += [f"  {session}" for session in sessions]
        lines += [
            f"left alone, studio:turn_index is not a whole number: {name}"
            for name in self.unreadable
        ]
        count = sum(len(sessions) for sessions in self.ahead.values())
        lines.append(
            f"{count} turn sessions of {len(self.ahead)} conversations {would}deleted, "
            f"{len(self.unreadable)} ledger sessions left alone"
            + ("; run again until none is found" if count else "")
        )
        return "\n".join(lines)


async def rollback_turn_sessions(engine: AsyncEngine, *, apply: bool) -> RollbackReport:
    """Delete, for each ledger session, the conversation's ADK turn sessions
    numbered above its `studio:turn_index` (module docstring). With `apply`
    false, list them only, without locking.

    Raises:
        AdkSchemaError: ADK's schema is not the one the job reads.
    """
    async with engine.connect() as conn:
        await _check_adk_schema(conn)
    report = RollbackReport(apply=apply)
    lock = " FOR UPDATE" if apply else ""
    for key in await _ledger_keys(engine):
        names = {"principal": key.principal, "id": key.conversation_id}
        async with engine.begin() as conn:
            row = (
                await conn.execute(text(_LEDGER_TURN_INDEX + lock), {**names, "app": LEDGER_APP})
            ).first()
            if row is None:
                continue
            index = 0 if row.turn_index is None else row.turn_index
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                report.unreadable.append(key.name)
                continue
            ahead = (
                (
                    await conn.execute(
                        _TURN_SESSIONS_AHEAD, {**names, "app": TURN_APP, "turn_index": index}
                    )
                )
                .scalars()
                .all()
            )
            if not ahead:
                continue
            report.ahead[key.name] = list(ahead)
            if apply:
                await conn.execute(
                    text(
                        "DELETE FROM sessions WHERE app_name = :app AND user_id = :principal "
                        "AND id = ANY(:ids)"
                    ),
                    {"app": TURN_APP, "principal": key.principal, "ids": list(ahead)},
                )
    return report


# Whether a table named `sessions` resolves through `search_path` with the
# columns of ADK's `sessions` (v0 and v1), as 0005 reads ADK's tables. When
# it does not, there are no ledger sessions to check.
CUTOVER_ADK_SESSIONS = (
    "SELECT to_regclass('sessions') IS NOT NULL AND NOT EXISTS ("
    "SELECT 1 FROM unnest(ARRAY['app_name', 'user_id', 'id', 'state', 'create_time', "
    "'update_time']) AS c(col) "
    "WHERE NOT EXISTS (SELECT 1 FROM pg_attribute a "
    "WHERE a.attrelid = to_regclass('sessions') AND a.attname = c.col "
    "AND a.attnum > 0 AND NOT a.attisdropped))"
)

# A ledger session's import record is current when it was recorded at the
# session's `update_time`: ADK has not written the session since.
_IMPORT_CURRENT = "i.source_updated_at = s.update_time"

# Each ledger session whose import record is not current, or that has none.
_STALE_IMPORTS = text(
    "SELECT s.user_id, s.id FROM sessions s LEFT JOIN adk_ledger_imports i "
    "ON i.principal = s.user_id AND i.conversation_id = s.id "
    f"WHERE s.app_name = :app AND NOT coalesce({_IMPORT_CURRENT}, false) "
    "ORDER BY s.user_id, s.id"
)

# The cutover predicate: every ledger session not cleared, the first ten by
# key, each with the count of all of them. A session is cleared when its
# `adk_ledger_imports` row is current (`source_updated_at` is the session's
# `update_time`) and either imported (not diverged, `imported_digest` set,
# a `conversations` row present) or kept (`kept_tables`).
CUTOVER_PREDICATE = (
    "SELECT s.user_id AS principal, s.id AS conversation_id, count(*) OVER () AS uncleared "
    "FROM sessions s LEFT JOIN adk_ledger_imports i "
    "ON i.principal = s.user_id AND i.conversation_id = s.id "
    f"WHERE s.app_name = '{LEDGER_APP}' "
    f"AND NOT coalesce({_IMPORT_CURRENT} AND (i.kept_tables OR ("
    "NOT i.diverged AND i.imported_digest IS NOT NULL AND EXISTS ("
    "SELECT 1 FROM conversations c "
    "WHERE c.principal = s.user_id AND c.conversation_id = s.id))), false) "
    "ORDER BY s.user_id, s.id LIMIT 10"
)

# `CUTOVER_PREDICATE` where the import's tables do not exist yet.
_CUTOVER_NO_TABLES = (
    "SELECT user_id AS principal, id AS conversation_id, count(*) OVER () AS uncleared "
    f"FROM sessions WHERE app_name = '{LEDGER_APP}' ORDER BY user_id, id LIMIT 10"
)

_GATE_COMMANDS = (
    "python -m harness.adk_import import --apply",
    "python -m harness.adk_import import --apply --prefer-adk PRINCIPAL/CONVERSATION_ID ...",
    "python -m harness.adk_import import --apply --keep-tables PRINCIPAL/CONVERSATION_ID ...",
    "python -m harness.adk_import import --apply --discard PRINCIPAL/CONVERSATION_ID ...",
)


# Each cleared ledger session the operator discarded: its current import
# record keeps the tables' copy, and the tables hold none of it.
_DISCARDED = text(
    "SELECT s.user_id, s.id FROM sessions s JOIN adk_ledger_imports i "
    "ON i.principal = s.user_id AND i.conversation_id = s.id "
    f"WHERE s.app_name = :app AND {_IMPORT_CURRENT} AND i.kept_tables "
    "AND NOT EXISTS (SELECT 1 FROM conversations c "
    "WHERE c.principal = s.user_id AND c.conversation_id = s.id) "
    "ORDER BY s.user_id, s.id"
)


@dataclass
class GateReport:
    """What `cutover_gate` found: how many ledger sessions are not cleared
    and the first ten of them, each ledger session a stream of the previous
    revision holds a live lease on, and each cleared ledger session the
    operator discarded, by `principal/conversation_id`. `archived` is set
    when no ADK `sessions` table resolves and `--adk-archived` said that is
    intended."""

    uncleared: int = 0
    first: list[str] = field(default_factory=list)
    streaming_on_adk: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)
    archived: bool = False

    @property
    def clean(self) -> bool:
        return self.uncleared == 0 and not self.streaming_on_adk

    def render(self) -> str:
        if self.archived:
            return "cutover gate: no ADK sessions table, archived as --adk-archived says"
        if self.clean:
            lines = ["cutover gate: every ledger session is cleared"]
            if self.discarded:
                lines.append(
                    f"cutover gate: {len(self.discarded)} of them discarded (--discard): the "
                    "tables hold none of each, and its next turn starts it afresh:"
                )
                lines += [f"  {name}" for name in self.discarded]
            return "\n".join(lines)
        lines = []
        if self.streaming_on_adk:
            lines.append(
                f"cutover gate: {len(self.streaming_on_adk)} ledger sessions hold a live "
                "stream lease on ADK; wait for the streams to end and run the gate again:"
            )
            lines += [f"  {name}" for name in self.streaming_on_adk]
        if self.uncleared == 0:
            return "\n".join(lines)
        lines.append(f"cutover gate: {self.uncleared} ledger sessions not cleared")
        lines += [f"  {name}" for name in self.first]
        if self.uncleared > len(self.first):
            lines.append(f"  and {self.uncleared - len(self.first)} more")
        lines.append(
            "clear them with: an import (copies each one ADK moved since its copy), "
            "--prefer-adk (takes ADK's copy of a conflict), --keep-tables (keeps the "
            "tables' copy of a refused or diverged conversation they hold) or --discard "
            "(gives up a refused conversation they do not hold), then run the gate again:"
        )
        lines += [f"  {command}" for command in _GATE_COMMANDS]
        return "\n".join(lines)


async def cutover_gate(engine: AsyncEngine, *, adk_archived: bool = False) -> GateReport:
    """Run the cutover predicate (`CUTOVER_PREDICATE`), list the ledger
    sessions whose stream lease is live (`_STREAMING_ON_ADK`) and those the
    operator discarded (`_DISCARDED`): clean when every ledger session is
    cleared and none holds a live lease. Columns and tables a later release
    adds do not change what it reads. Before `import` has created the
    tables, no ledger session is cleared.

    Raises:
        AdkTablesMissing: no table named `sessions` carries ADK's columns
            (`CUTOVER_ADK_SESSIONS`), as on a database the job was not meant
            to read, unless `adk_archived` says the tables were archived on
            purpose, which returns a clean report.
    """
    async with engine.connect() as conn:
        if not (await conn.execute(text(CUTOVER_ADK_SESSIONS))).scalar_one():
            if adk_archived:
                return GateReport(archived=True)
            raise AdkTablesMissing(
                "no table named sessions carries ADK's columns, so the gate has nothing to "
                "check: the job may be reading the wrong database. Pass --adk-archived only "
                "when ADK's tables were archived on purpose."
            )
        streaming = [
            f"{principal}/{cid}"
            for principal, cid in await conn.execute(_STREAMING_ON_ADK, {"app": LEDGER_APP})
        ]
        imported = (
            await conn.execute(
                text(
                    "SELECT to_regclass('conversations') IS NOT NULL "
                    "AND to_regclass('adk_ledger_imports') IS NOT NULL"
                )
            )
        ).scalar_one()
        rows = (
            await conn.execute(text(CUTOVER_PREDICATE if imported else _CUTOVER_NO_TABLES))
        ).all()
        discarded = (
            [
                f"{principal}/{cid}"
                for principal, cid in await conn.execute(_DISCARDED, {"app": LEDGER_APP})
            ]
            if imported
            else []
        )
    if not rows:
        return GateReport(streaming_on_adk=streaming, discarded=discarded)
    return GateReport(
        uncleared=rows[0].uncleared,
        first=[f"{row.principal}/{row.conversation_id}" for row in rows],
        streaming_on_adk=streaming,
        discarded=discarded,
    )


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.adk_import",
        description="Copy Studio conversations from ADK ledger sessions into the harness's "
        "tables. Writes nothing to the conversations unless --apply is given; the import's "
        "dry run applies the harness's migrations.",
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    imports = verbs.add_parser("import", help="copy every ledger session into the tables")
    mode = imports.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write the copies")
    mode.add_argument(
        "--verify",
        action="store_true",
        help="compare each conversation as ADK loads it with the tables' load",
    )
    imports.add_argument(
        "--prefer-adk",
        nargs="+",
        default=[],
        metavar="PRINCIPAL/CONVERSATION_ID",
        action="extend",
        help="re-import these conversations, named as the report names them, over rows "
        "changed on both sides",
    )
    imports.add_argument(
        "--keep-tables",
        nargs="+",
        action="extend",
        default=[],
        metavar="PRINCIPAL/CONVERSATION_ID",
        help="keep the tables' copy of these refused or diverged conversations the tables "
        "hold and give up ADK's, until ADK's ledger session moves again",
    )
    imports.add_argument(
        "--discard",
        nargs="+",
        action="extend",
        default=[],
        metavar="PRINCIPAL/CONVERSATION_ID",
        help="give up these refused conversations the tables do not hold: each starts afresh "
        "on its next turn, and the report and the gate count it",
    )
    imports.add_argument(
        "--force",
        action="store_true",
        help="re-import a --prefer-adk conversation even while a stream holds its lease, "
        "which ends that stream",
    )
    closes = verbs.add_parser(
        "close-open-turns", help="close every open ADK turn with no live lease as abandoned"
    )
    closes.add_argument(
        "--apply",
        action="store_true",
        help="write the closed turns; refused until the next release's turn_cursors table exists",
    )
    rollback = verbs.add_parser(
        "rollback-turn-sessions",
        help="delete each ADK turn session numbered above its ledger session's turn index, "
        "for a rollback to the previous release",
    )
    rollback.add_argument("--apply", action="store_true", help="delete the turn sessions")
    gate = verbs.add_parser(
        "gate",
        help="check that every ledger session is imported or kept as ADK holds it now and "
        "that no stream holds a live lease on one; reads only",
    )
    gate.add_argument(
        "--adk-archived",
        action="store_true",
        help="pass when no ADK sessions table resolves, because ADK's tables were archived",
    )
    return parser.parse_args(argv)


type _Report = ImportReport | VerifyReport | CloseReport | RollbackReport | GateReport


async def _run(arguments: argparse.Namespace, engine: AsyncEngine) -> _Report:
    if arguments.verb == "gate":
        return await cutover_gate(engine, adk_archived=arguments.adk_archived)
    if arguments.verb == "close-open-turns":
        return await close_open_turns(engine, apply=arguments.apply)
    if arguments.verb == "rollback-turn-sessions":
        return await rollback_turn_sessions(engine, apply=arguments.apply)
    if arguments.verify:
        return await verify(engine)
    return await import_conversations(
        engine,
        apply=arguments.apply,
        prefer_adk=frozenset(arguments.prefer_adk),
        keep_tables=frozenset(arguments.keep_tables),
        discard=frozenset(arguments.discard),
        force=arguments.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verb and print its report. Returns the exit status.

    Logging and telemetry come up as the service's do
    (`harness.server._lifespan`), so the `studio.turn` line and `turn`
    outcome of each turn `close-open-turns` closes are written as the
    service writes them: a JSON log line in a Cloud Run job execution, and,
    when telemetry is configured, an outcome the job exports, blocking until
    the export ends, once its event loop has finished and before it exits
    (`flush_telemetry_before_exit`)."""
    arguments = _arguments(argv)
    configure_logging()
    init_telemetry(service_name="harness")
    database = build_database(settings=Settings())
    if database is None:
        print("adk_import: set DATABASE_URL or the SESSION_STORE_* variables", file=sys.stderr)
        return 2

    async def go() -> _Report:
        try:
            return await _run(arguments, database.engine)
        finally:
            await database.aclose()

    try:
        report = asyncio.run(go())
    except (
        AdkSchemaError,
        AdkTablesMissing,
        NextReleaseMissing,
        TablesMissing,
        MigrationChecksumError,
        UnknownConversation,
        NamedTwice,
    ) as exc:
        print(f"adk_import: {exc}", file=sys.stderr)
        return 2
    finally:
        flush_telemetry_before_exit()
    print(report.render())
    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
