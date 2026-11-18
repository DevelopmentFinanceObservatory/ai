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
  `pending` included, since this release still resumes a pause ADK holds.
  A conversation is refused whole, with every reason named, for a
  structural fault: a key the job does not know, a turn or approval
  position that is not a number or leaves a gap, a write token that is not
  a string, or a run-ledger item naming a run the ledger does not hold.
  Any other failure while importing one conversation refuses it too, and
  the run goes on with the rest. Records this release's load skips or fails on (a
  plan record, search reading, identity, approval, canvas, pending
  interrupt or run record its model refuses, or a spec block whose
  `_bound_refs` does not convert, which the reader drops) are reported and
  copied as they are.

  `adk_ledger_imports` records a digest of each imported ADK state, the
  ledger session's `update_time` read with it, and a digest of the rows
  written, so a rerun leaves alone a conversation whose ADK state has not
  moved, re-imports one whose ADK state moved while its rows are still as
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

  The report counts the spec blocks converted and those that do not
  convert, in the ledger sessions and, on a line of their own, in the open
  turns' sessions, read in the conversation's own transaction. It lists an
  open turn's blocks that do not convert under a heading of their own: the
  job does not copy them, and the release reads them from the turn session
  as stored. It gives the size distribution (p50, p95, max, in bytes of JSON
  text) of the ledger sessions' states and of the open turns'
  `orchestrator:specs`.

  `--verify` writes nothing and compares, for every ledger session, the
  conversation as ADK's state loads (`_apply_delta` over the mapped keys)
  with `StudioStore.load` from the tables, the canvas included, and the
  run-ledger items with the run rows, its lease's cancel flag left out.
  Both sides take the open turn's spec blocks from its turn session
  through the reader the server serves (`open_turn_specs`), converted as
  `StudioStore.load` converts them. A conversation
  whose ADK state moved since the job imported it while its rows did not
  (the previous release still serving), and one whose rows moved since,
  written by the release that reads the tables (one written while verify
  compares it included), are each listed and not compared; one where both
  moved is a conflict, listed, not compared, and fails the verify.
- `close-open-turns`: closes, in the tables, every open turn that holds no
  live lease and ran on ADK (its turn session `{conversation_id}#t{n}`
  under app `dfo_analyst_graph` exists), as `ConversationState.begin_turn`
  closes an open turn: the record keeps what was asked with the error
  `abandoned`, and the blocks of the sources its plans used, read from the
  turn session's `orchestrator:specs`. Its pending interrupts and its
  lapsed lease are dropped. A block that does not convert is kept as
  stored, as `import` keeps one, and reported. A diverged conversation is
  left alone and listed, and fails the run: its rows lack the turns ADK
  holds, so `--prefer-adk` must take ADK's copy first. A conversation
  whose close fails is refused and left open; the run goes on with the
  rest.

The serving release reads conversations from the tables while the previous
release may still finish turns on ADK. `reimport_on_access` is the
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

Without `--apply` nothing is written to the conversation tables. Only
`import` migrates, so `--verify` and `close-open-turns` also run from an
image older than the database's schema; they refuse to start when the
tables do not exist.

The database comes from the settings the service uses
(`harness.settings.Settings`).

Exit status: 0 when every conversation imported, was unchanged or verified
equal, and every open turn eligible was closed; 1 on a refusal, a conflict,
a difference, a diverged conversation `close-open-turns` left alone, or a
conversation that changed during the run; 2 when no database is configured,
ADK's schema is not the one the job reads, the harness's tables are missing,
a migration file is refused, or `--prefer-adk` names a conversation no ledger
session holds.
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

from compiler.sources import bound_rows, source_from_spec_block
from executor.ledger import RunStatus, stored_run
from pydantic import BaseModel, ValidationError
from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.approval import ApprovalRecord
from harness.db.engine import build_database
from harness.db.migrate import MigrationChecksumError, apply_pending
from harness.graph.state import AGENT_NAME, ORCHESTRATOR_SPECS_KEY
from harness.settings import Settings
from harness.studio_state import ConversationState
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
    StudioStore,
    TurnSpecs,
    _apply_delta,
    _delta_write,
    _DeltaWrite,
    _deserialize_pending,
    _run_delta_write,
    _serialize_durable,
)
from harness.tool_contracts import HarnessCode
from harness.turn_records import PlanRecord, RecordedIdentity, SearchReading
from harness.wire import CompositionManifest

logger = logging.getLogger(__name__)

LEDGER_APP = f"{AGENT_NAME}.ledger"
TURN_APP = AGENT_NAME
ADK_SCHEMA_VERSION = "1"
# Frame numbers an imported conversation reserves past the last one ADK
# recorded. A resume the previous release retried after its stream died
# started that many past the stored number, since the dead stream may have
# sent frames past its last save; a resumed pause starts past these.
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


class UnknownConversation(ValueError):  # noqa: N818 -- names the state it reports
    """`--prefer-adk` names a conversation ADK holds no ledger session for."""


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
    string ref is dropped; a mapping left with no entry drops the key, as a
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
    left as it is and named in `blocks`; `value` itself comes back when
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
    `faults` are the structural faults that refuse the conversation;
    `findings` the records this release's load skips or fails on, a block that does
    not convert among them. `blocks` counts the conversion.
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


async def _turn_session_specs(
    engine: AsyncEngine, principal: str, session_id: str
) -> _TurnSpecs | None:
    """The spec blocks a turn's own ADK session holds, converted, empty when
    it holds none. None when the session does not exist."""
    return _converted_turn(session_id, await _turn_session_row(engine, principal, session_id))


async def _read_turn_session(
    conn: AsyncConnection, principal: str, session_id: str
) -> _TurnSpecs | None:
    """`_turn_session_specs` on `conn`."""
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
    """`_stored_turn_specs` on a connection of its own: the one read of a
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


def open_turn_specs(engine: AsyncEngine) -> TurnSpecs:
    """`StudioStore`'s `turn_specs` reader over `engine`: the spec blocks the
    turn session `session_id` (app `dfo_analyst_graph`) holds, read by SQL
    as stored. Empty when the session holds none or does not exist.
    `StudioStore.load` converts what it reads, once, through its bridge's
    `convert_specs`. The server serves Studio through it, and `--verify`
    reads open turns through it and the job's bridge (`_NoAdk`), whose
    conversion is the serving bridge's (`_converted_specs`), so the gate
    checks the blocks the serving release loads."""

    async def turn_specs(principal: str, session_id: str) -> dict[str, Any]:
        stored = await _turn_session_row(engine, principal, session_id)
        return stored[0] if stored is not None else {}

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
    """The id of the conversation `names` holds the `principal` and
    `conversation_id` of, None when no row holds it. With `lock`, the row is
    locked for the transaction."""
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
                "SELECT source_digest, imported_digest FROM adk_ledger_imports "
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


# The lease field neither the rows' digest nor `--verify`'s comparison
# reads. A Stop sent to this release on a stream the previous release runs
# sets it on the lease a copy brought across; the flag never reaches that
# stream, and the next copy replaces it with ADK's, so it is no write the
# tables would lose.
_UNCOMPARED_LEASE_FIELD = "cancel"


async def _stored_digest(conn: AsyncConnection, pk: int) -> str:
    """A digest of every row the tables hold for the conversation, its lease
    included bar `_UNCOMPARED_LEASE_FIELD`."""
    conversation = (
        await conn.execute(
            text(
                "SELECT to_jsonb(c) - 'id' - 'created_at' - 'updated_at' "
                f"- 'lease_{_UNCOMPARED_LEASE_FIELD}' FROM conversations c WHERE id = :pk"
            ),
            {"pk": pk},
        )
    ).scalar_one()
    rows: dict[str, list[Any]] = {"conversations": [conversation]}
    for table, order in _CHILD_TABLES:
        result = await conn.execute(
            text(
                f"SELECT to_jsonb(t) - 'conversation' FROM {table} t "
                f"WHERE conversation = :pk ORDER BY {order}"
            ),
            {"pk": pk},
        )
        rows[table] = list(result.scalars())
    return _digest(rows)


def _jsonb(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


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
                "record": json.dumps(record),
                "submission": _jsonb(runs.submissions.get(run_id)),
                "retrievals": _jsonb(runs.retrievals.get(run_id)),
            }
            for run_id, record in runs.records.items()
        ],
        keys=[{"key": key, "run_id": run_id} for key, run_id in runs.keys.items()],
        plans=[
            {"revision": revision, "plan": json.dumps(plan)}
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
                ("conflict, changed on both sides (--prefer-adk re-imports)", self.conflicts),
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
            f"{len(self.refused)} refused",
        ]
        return "\n".join(lines)


type _Outcome = Literal["imported", "reimported", "unchanged", "conflict", "live", "refused"]


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


async def _import_one(
    engine: AsyncEngine,
    principal: str,
    conversation_id: str,
    *,
    apply: bool,
    prefer_adk: frozenset[str] = frozenset(),
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
    (unless `force`), and `conflict` otherwise.

    Under `apply` the conversation's row is locked, or inserted, before the
    ledger session, the import record and the rows' digest are read, so a
    concurrent writer, the serving release or another instance copying the
    same conversation, either commits first and is seen, or waits, and
    copies serialised on the lock read ADK in the order they commit. A
    conflict records the session's `update_time` and marks the record
    `diverged`, its digests left as they were, so the job and `--verify`
    still decide by digest. A refusal writes nothing, a row inserted for it
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
            if ledger.faults:
                reasons = ledger.faults
                raise _Undo
            try:
                rows = _import_rows(ledger)
            # Whatever building the rows raises refuses the conversation, before
            # any outcome is decided, so a refusal is recorded as a refusal.
            except Exception as exc:
                reasons = [f"the import failed: {type(exc).__name__}: {exc}"]
                raise _Undo from exc
            if open_turn:
                turn = await _read_open_turn(conn, session, ledger)
            source = _digest(session.state)
            updated = {**names, "updated": session.updated_at}
            previous = await _import_record(conn, names)
            if not created and previous is not None and previous.source_digest == source:
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
            elif session.name in prefer_adk:
                outcome = "reimported"
                if not force and await _lease_live(conn, pk):
                    outcome = "live"
            else:
                outcome = "conflict"
                if apply:
                    await conn.execute(
                        text(
                            "INSERT INTO adk_ledger_imports "
                            "(principal, conversation_id, source_updated_at, diverged) "
                            "VALUES (:principal, :conversation_id, :updated, true) "
                            "ON CONFLICT (principal, conversation_id) DO UPDATE SET "
                            "source_updated_at = EXCLUDED.source_updated_at, diverged = true"
                        ),
                        updated,
                    )
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
                        "imported_at = now()"
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
    force: bool = False,
) -> ImportReport:
    """Import every ledger session (module docstring). With `apply` false,
    read and decide only. `prefer_adk` holds `principal/conversation_id`
    names to re-import over rows changed on both sides; one a stream holds
    is refused unless `force`.

    Raises:
        AdkSchemaError: ADK's schema is not the one the job reads.
        UnknownConversation: `prefer_adk` names a conversation ADK holds no
            ledger session for. Nothing is written.
    """
    async with engine.connect() as conn:
        await _check_adk_schema(conn)
    keys = await _ledger_keys(engine)
    if unknown := sorted(prefer_adk - {key.name for key in keys}):
        raise UnknownConversation(f"--prefer-adk names no ledger session: {', '.join(unknown)}")
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
                force=force,
                open_turn=True,
            )
        # One conversation's failure refuses it, with nothing written; the
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
            {
                "imported": report.imported,
                "reimported": report.reimported,
                "unchanged": report.unchanged,
                "conflict": report.conflicts,
            }[copy.outcome].append(key.name)
    return report


# One conversation's ledger session against what was last copied of it: no
# row when ADK holds no ledger session for it; `current` false when the
# session's `update_time` is not the one last recorded, or nothing is
# recorded; `diverged` when the last copy found both sides changed. One
# lookup of `sessions` by its primary key, joined to `adk_ledger_imports` by
# its own.
_ADK_MOVED = text(
    "SELECT i.source_updated_at IS NOT DISTINCT FROM s.update_time AS current, "
    "coalesce(i.diverged, false) AS diverged, s.update_time "
    "FROM sessions s LEFT JOIN adk_ledger_imports i "
    "ON i.principal = s.user_id AND i.conversation_id = s.id "
    "WHERE s.app_name = :app AND s.user_id = :principal AND s.id = :conversation_id"
)


def reimport_on_access(engine: AsyncEngine) -> AdkBridge:
    """`StudioStore`'s `AdkBridge` for the release that
    serves Studio from the tables while the previous release still finishes
    turns on ADK.

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
    lock: a conversation only ADK changed is re-imported in place and one
    never imported is imported.

    A conversation that changed on both sides is diverged: an error names
    it and the command that takes ADK's copy, and every load of it raises
    `ConversationHeld` until that command runs. The tables lack what ADK
    holds, so serving them would lose the previous release's turns. A
    conversation whose ledger session the copy refuses (`_Copy.reasons`)
    is logged as an error once per ADK revision in this process and served
    as the tables hold it: when they do not hold it, `load` finds no
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
            if copy.outcome != "unchanged":
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
        "ADK's copy",
        name,
        name,
    )
    return ConversationHeld(conversation_id)


class _NoAdk:
    """The job's stores read the tables as they stand and serve no turn."""

    async def reimport(self, principal: str, conversation_id: str) -> None:
        del principal, conversation_id

    async def moved(self, principal: str, conversation_id: str) -> bool:
        del principal, conversation_id
        return False

    def convert_specs(self, session_id: str, specs: dict[str, Any]) -> dict[str, Any]:
        return _converted_specs(session_id, specs, Blocks())


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


async def _adk_view(engine: AsyncEngine, session: _Session) -> dict[str, Any]:
    """The conversation and its run items as ADK's ledger session loads, its
    spec blocks converted as the import converts them."""
    ledger = read_ledger(session.state)
    state = ConversationState(conversation_id=session.conversation_id)
    _apply_delta(state, ledger.flat)
    await _read_turn_specs(engine, session.principal, state)
    return _conversation_view(state) | _run_view(ledger.runs)


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
    kinds are listed apart and not compared. A conversation whose ADK state
    moved since the import while its rows stayed as imported, written by
    the previous release still serving, which the serving release
    re-imports on its next read. One whose rows the serving release has
    written since the import: its tables hold turns ADK never saw. One
    where both moved is a conflict, and fails the verify."""

    total: int = 0
    differences: dict[str, list[str]] = field(default_factory=dict)
    moved_on_adk: list[str] = field(default_factory=list)
    written_since_import: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.differences and not self.conflicts

    def render(self) -> str:
        compared = (
            self.total
            - len(self.moved_on_adk)
            - len(self.written_since_import)
            - len(self.conflicts)
        )
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
            f"conflict, changed on both sides since the import (--prefer-adk re-imports): {name}"
            for name in self.conflicts
        ]
        lines.append(f"{compared - len(self.differences)} of {compared} equal")
        return "\n".join(lines)


async def _tables_side(
    engine: AsyncEngine, session: _Session
) -> Literal["as imported", "moved on adk", "written", "conflict"]:
    """How the conversation stands against the import: `as imported` when
    its rows are what the job wrote from ADK's state as it stands, or the
    job has not imported it; `moved on adk` when the rows are as imported
    and ADK's state moved since; `written` when the serving release wrote
    the rows since; `conflict` when it did and ADK's state also moved."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT c.id, i.imported_digest, i.source_digest FROM conversations c "
                    "JOIN adk_ledger_imports i USING (principal, conversation_id) "
                    "WHERE principal = :principal AND conversation_id = :conversation_id"
                ),
                {"principal": session.principal, "conversation_id": session.conversation_id},
            )
        ).first()
        if row is None:
            return "as imported"
        rows_moved = await _stored_digest(conn, row.id) != row.imported_digest
    adk_moved = _digest(session.state) != row.source_digest
    if rows_moved:
        return "conflict" if adk_moved else "written"
    return "moved on adk" if adk_moved else "as imported"


async def verify(engine: AsyncEngine) -> VerifyReport:
    """Compare every ledger session's load with the tables' (module docstring)."""
    await _check_tables(engine)
    report = VerifyReport()
    async for session in _ledger_sessions(engine, await _ledger_keys(engine)):
        report.total += 1
        side = await _tables_side(engine, session)
        if side == "as imported" and (differences := await _differences(engine, session)):
            # A write the serving release commits between the classification
            # and the compared reads shows as a difference; the rows' digest,
            # read again, tells it from one.
            side = await _tables_side(engine, session)
            if side == "as imported":
                report.differences[session.name] = differences
        if side == "moved on adk":
            report.moved_on_adk.append(session.name)
        elif side == "written":
            report.written_since_import.append(session.name)
        elif side == "conflict":
            report.conflicts.append(session.name)
    return report


async def _differences(engine: AsyncEngine, session: _Session) -> list[str]:
    """What differs between the conversation as ADK's state loads and as the
    tables load: the keys whose values differ, or how the loads failed."""
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
    `principal/conversation_id turn n`."""

    apply: bool
    closed: list[str] = field(default_factory=list)
    live: list[str] = field(default_factory=list)
    not_on_adk: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    diverged: list[str] = field(default_factory=list)
    refused: dict[str, list[str]] = field(default_factory=dict)
    findings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.changed and not self.diverged and not self.refused

    def render(self) -> str:
        would = "" if self.apply else "would be "
        lines = [
            f"{label}: {name}"
            for label, names in (
                (f"{would}closed as abandoned", self.closed),
                ("left open, a stream holds its lease", self.live),
                ("left open, no ADK turn session", self.not_on_adk),
                ("changed during the run, run again", self.changed),
                ("left alone, diverged (--prefer-adk first)", self.diverged),
            )
            for name in names
        ]
        for name, reasons in self.refused.items():
            lines.append(f"left open, refused: {name}")
            lines += [f"  {reason}" for reason in reasons]
        for name, found in self.findings.items():
            lines.append(
                f"{would}closed with blocks as stored, which the turn loader drops: {name}"
            )
            lines += [f"  {finding}" for finding in found]
        lines.append(
            f"{len(self.closed)} {would}closed, {len(self.live)} live, "
            f"{len(self.not_on_adk)} without an ADK turn session, {len(self.changed)} changed, "
            f"{len(self.diverged)} diverged, {len(self.refused)} refused"
        )
        return "\n".join(lines)


async def close_open_turns(engine: AsyncEngine, *, apply: bool) -> CloseReport:
    """Close every open ADK turn the tables hold (module docstring). With
    `apply` false, decide only. A diverged conversation is left alone and
    listed, since its rows lack what ADK holds. A conversation whose close
    fails is refused, and the run goes on with the rest."""
    await _check_tables(engine)
    report = CloseReport(apply=apply)
    async with engine.connect() as conn:
        conversations = (
            await conn.execute(
                text(
                    # Lease expiry is read on the database's clock, as the
                    # store reads it.
                    "SELECT c.principal, c.conversation_id, "
                    f"coalesce(c.lease_expires_at > {DATABASE_NOW}, false) AS leased, "
                    "coalesce(i.diverged, false) AS diverged "
                    "FROM conversations c LEFT JOIN adk_ledger_imports i "
                    "USING (principal, conversation_id) ORDER BY principal, conversation_id"
                )
            )
        ).all()
    for principal, conversation_id, leased, diverged in conversations:
        if diverged:
            report.diverged.append(f"{principal}/{conversation_id}")
            continue
        try:
            await _close_one(
                engine, principal, conversation_id, leased=leased, apply=apply, report=report
            )
        # One conversation's failure refuses it; the run goes on with the rest.
        except Exception as exc:
            report.refused[f"{principal}/{conversation_id}"] = [
                f"the close failed: {type(exc).__name__}: {exc}"
            ]
    return report


async def _close_one(
    engine: AsyncEngine,
    principal: str,
    conversation_id: str,
    *,
    leased: bool,
    apply: bool,
    report: CloseReport,
) -> None:
    """Close one conversation's open ADK turn, if it has one, and note what
    became of it in `report`."""
    store = _store(engine)
    state = await store.load(principal, conversation_id)
    if state is None or not state.turn_index or state.turn_recorded:
        return
    name = f"{principal}/{conversation_id} turn {state.turn_index}"
    if leased:
        report.live.append(name)
        return
    turn = await _turn_session_specs(engine, principal, state.session_id)
    if turn is None:
        report.not_on_adk.append(name)
        return
    # A block that does not convert is kept as stored, as `import` keeps one.
    if turn.blocks.unconvertible:
        report.findings[name] = turn.blocks.unconvertible
    if not apply:
        report.closed.append(name)
        return
    state.turn_specs = turn.specs
    state.close_turn(final_text=state.turn_final_text, error="abandoned")
    state.pending.clear()
    # No stream holds the lease, so any lease on record has lapsed. Clearing
    # it fences out a stream still running under it: its checkpoint is
    # fenced on the lease id alone, and would otherwise write over the turn
    # this closes.
    state.segment = None
    try:
        await store.save(principal, conversation_id)
    except ConversationStale:
        report.changed.append(name)
        return
    report.closed.append(name)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.adk_import",
        description="Copy Studio conversations from ADK ledger sessions into the harness's "
        "tables. Reads only, unless --apply is given.",
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
        help="re-import these conversations, named as the report names them, over rows "
        "changed on both sides",
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
    closes.add_argument("--apply", action="store_true", help="write the closed turns")
    return parser.parse_args(argv)


async def _run(
    arguments: argparse.Namespace, engine: AsyncEngine
) -> ImportReport | VerifyReport | CloseReport:
    if arguments.verb == "close-open-turns":
        return await close_open_turns(engine, apply=arguments.apply)
    if arguments.verify:
        return await verify(engine)
    return await import_conversations(
        engine,
        apply=arguments.apply,
        prefer_adk=frozenset(arguments.prefer_adk),
        force=arguments.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verb and print its report. Returns the exit status."""
    arguments = _arguments(argv)
    database = build_database(settings=Settings())
    if database is None:
        print("adk_import: set DATABASE_URL or the SESSION_STORE_* variables", file=sys.stderr)
        return 2

    async def go() -> ImportReport | VerifyReport | CloseReport:
        try:
            return await _run(arguments, database.engine)
        finally:
            await database.aclose()

    try:
        report = asyncio.run(go())
    except (AdkSchemaError, TablesMissing, MigrationChecksumError, UnknownConversation) as exc:
        print(f"adk_import: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
