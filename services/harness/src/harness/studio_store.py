"""`StudioStore`: each conversation's durable state in the harness's own tables.

A conversation is one `conversations` row (defined in
`harness/db/migrations/0001_conversations.sql`) and its per-item rows: closed turns, the digest
entry and write token of each component, and approvals. Every durable field
of a `ConversationState`, everything but the identity field
`conversation_id` and the process-local fields `_PROCESS_LOCAL_FIELDS`
names, lives there, so any instance can load a conversation and continue it.
The store keeps only runtime objects and each instance's own live
`ConversationState` in process, like `harness.provenance.ProvenanceStore`.

The row's column groups each have their own writer, so writers that touch
different groups never conflict:

- `save` writes the turn state compare-and-swapped on the row's `version`
  (`ConversationStale` when another instance moved it, or holds a live
  lease).
- `checkpoint` and `release` write the turn state fenced on the segment
  (`SegmentLost` when another instance holds the lease).
- `change_canvas` is a checkpoint with a card change applied: each card
  the turn lands, holds, swaps, revises or removes commits with its canvas
  block and every other changed durable field in one transaction, fenced
  on the segment.
- `mutate_composition` writes the canvas column alone for each
  `/composition` edit, unfenced, and leaves `version` alone.
- `renew` and `request_cancel` each update the lease columns in one
  conditional statement and leave `version` alone.
- `reserve_frames` extends the frame numbers the live segment may send,
  fenced on the segment and its lease being live.

Every canvas writer holds the row lock, so each applies its change to the
canvas as stored. Lease expiry is compared against the database's clock in
SQL, so instances whose clocks disagree agree on when a lease lapses.

Each write is one transaction. The writes a live stream makes between its
renewals (`checkpoint`, `change_canvas`, `reserve_frames`) retry a
transient database fault until the lease's deadline
(`StudioStore.lease_deadline`), then raise `DatabaseUnavailable`; the rest
do not retry. The run ledger
(`harness.run_ledger`) writes inside `fenced`.

The open turn's spec blocks are the one thing read from elsewhere: the
turn's own ADK session holds them, and `load` reads them back through the
`turn_specs` reader the store is given
(`harness.adk_import.open_turn_specs`) and converts blocks stored before
`planner.workflow.BoundEntity` through `AdkBridge.convert_specs`. Before
it reads a conversation's rows, `load` runs the `AdkBridge.reimport` of the
`adk` it is given, which copies in what the previous release wrote to the
conversation's ADK ledger session since the tables last took it
(`harness.adk_import.reimport_on_access`).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import math
import ssl
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Iterator,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from executor.ledger import RunStatus
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.approval import ApprovalRecord
from harness.studio_state import (
    SEGMENT_TTL_S,
    ConversationState,
    DatabaseUnavailable,
    Segment,
    SegmentBusy,
    SegmentLost,
    _PendingChoice,
    _PendingInterrupt,
    _PendingOption,
    _PendingQuestion,
    _wire_id,
)
from harness.wire import CompositionManifest, PlanProposed

logger = logging.getLogger(__name__)

# This process's id, carried on every segment it claims for logs only.
# Fencing compares segment ids.
_OWNER_ID = uuid.uuid4().hex[:12]

# A conversation's durable fields as one flat map, the shape `save` and
# `checkpoint` diff against the last stored snapshot. The scalar keys live in
# `conversations.turn_state`, `composition` and `segment` in their own
# columns, and each per-item key in a row of its own table.
_KEY_TURN_INDEX = "turn_index"
_KEY_SEQ = "seq"
_KEY_TRACE_ID = "trace_id"
_KEY_PENDING = "pending"
_KEY_PLAN_VERSION_IDS = "plan_version_ids"
_KEY_LATEST_PLAN_VERSION = "latest_plan_version"
_KEY_AUTO_ADD = "auto_add"
_KEY_TIMED_OUT_CALLS = "timed_out_calls"
_KEY_TURN_USER_MESSAGE = "turn_user_message"
_KEY_TURN_USER_WORDS = "turn_user_words"
_KEY_TURN_FINAL_TEXT = "turn_final_text"
_KEY_TURN_COMPONENTS = "turn_components"
_KEY_TURN_RECORDED = "turn_recorded"
_KEY_TURN_PLANS = "turn_plans"
_KEY_TURN_READINGS = "turn_readings"
_KEY_TURN_APPLIED_CHOICE = "turn_applied_choice"
_KEY_TURN_CHOICE_BASELINE = "turn_choice_baseline"
_SCALAR_KEYS = frozenset(
    {
        _KEY_TURN_INDEX,
        _KEY_TRACE_ID,
        _KEY_PENDING,
        _KEY_PLAN_VERSION_IDS,
        _KEY_LATEST_PLAN_VERSION,
        _KEY_AUTO_ADD,
        _KEY_TIMED_OUT_CALLS,
        _KEY_TURN_USER_MESSAGE,
        _KEY_TURN_USER_WORDS,
        _KEY_TURN_FINAL_TEXT,
        _KEY_TURN_COMPONENTS,
        _KEY_TURN_RECORDED,
        _KEY_TURN_PLANS,
        _KEY_TURN_READINGS,
        _KEY_TURN_APPLIED_CHOICE,
        _KEY_TURN_CHOICE_BASELINE,
    }
)
_KEY_COMPOSITION = "composition"
_KEY_SEGMENT = "segment"
# Per-item keys: `turns` and `approvals` by their position, `components` and
# `write_tokens` by component id.
_TURN_ITEM_PREFIX = "turn:"
_COMPONENT_ITEM_PREFIX = "component:"
_WRITE_TOKEN_ITEM_PREFIX = "write_token:"
_APPROVAL_ITEM_PREFIX = "approval:"

# Reads the spec blocks an open turn's own session holds, called with
# (principal, session_id); empty when it holds none.
type TurnSpecs = Callable[[str, str], Awaitable[dict[str, Any]]]


class AdkBridge(Protocol):
    """How `StudioStore` reads conversations the previous release still
    writes to ADK (`harness.adk_import.reimport_on_access`)."""

    async def reimport(self, principal: str, conversation_id: str) -> None:
        """Bring one conversation's rows up to date with its ADK ledger
        session before `StudioStore.load` reads them. Raises
        `ConversationHeld` for a conversation that cannot be served, and lets
        any other failure of the copy propagate, so the request fails before
        it writes."""
        ...

    async def moved(self, principal: str, conversation_id: str) -> bool:
        """Whether ADK's ledger session for a conversation moved since it was
        last copied or refused, or the conversation is diverged: the previous
        release wrote to it after this instance's `load` read the rows."""
        ...

    def convert_specs(self, session_id: str, specs: dict[str, Any]) -> dict[str, Any]:
        """`specs`, the spec blocks turn session `session_id` holds, with each
        block a release before `planner.workflow.BoundEntity` stored read as
        `compiler.sources.source_from_spec_block` takes it, and `specs`
        itself when converting changes nothing."""
        ...


# Identity, the process-local runtime objects, and the turn's spec blocks
# (`turn_specs`, read back from the turn's own ADK session by
# `StudioStore.load`), which `_serialize_durable`/`_apply_delta` skip.
_PROCESS_LOCAL_FIELDS = frozenset(
    {
        "conversation_id",
        "lock",
        "renew_now",
        "_telemetry_span",
        "_removal_unreported",
        "turn_specs",
    }
)

_NO_BASELINE_VALUE = object()

# The time every lease statement reads, in epoch seconds: the database's
# clock, so instances whose clocks disagree still agree on when a lease
# lapses. `now()` is the transaction's start time.
DATABASE_NOW = "CAST(extract(epoch FROM now()) AS double precision)"

# Frame numbers a segment reserves at a time (`StudioStore.reserve_frames`).
SEQ_BLOCK = 256
# A segment reserves its next block once fewer numbers than this are left,
# so the frames sent after the stream's last relayed one (the closing
# `activity.step` and `error` frames `harness.studio._sse_stream` adds on a
# Stop) still fall inside the reservation.
_SEQ_HEADROOM = 64

# The backoff between retries of a write fenced on the live segment
# (`StudioStore._retrying`), doubling from the first delay to the last.
_RETRY_FIRST_DELAY_S = 0.25
_RETRY_MAX_DELAY_S = 4.0
# How long before its lease lapses an instance stops trusting it holds the
# conversation: the heartbeat gives up renewing and writes stop retrying
# (`StudioStore.lease_deadline`).
LEASE_MARGIN_S = 10


# The field values `_shallow_copy` copies, one level deep.
_CONTAINERS = (dict, list, set)


def _shallow_copy(state: ConversationState) -> ConversationState:
    """A copy of `state` whose dict, list and set fields are copies one
    level deep and whose other fields hold the same objects. A change that
    replaces the values it changes, as every canvas change does, leaves
    `state` itself untouched."""
    return dataclasses.replace(
        state,
        **{
            f.name: copy.copy(value)
            for f in dataclasses.fields(state)
            if isinstance(value := getattr(state, f.name), _CONTAINERS)
        },
    )


async def _run_to_completion[T](work: Coroutine[Any, Any, T]) -> T:
    """Run `work` in a task of its own and wait for it to finish even when
    the caller is cancelled, then re-raise the caller's cancellation.

    A write the stream makes from a `finally` (`release`) or from a task the
    stream cancels (`renew`, on the heartbeat) is cancelled with
    `Task.cancel`, which no anyio shield stops. Cancelled mid-statement, the
    write never lands.

    The caller's cancellation wins over whatever `work` ended with: a
    heartbeat cancelled while its renewal fails must stop, or the stream it
    renews never finishes unwinding. What `work` raised is then logged."""
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            cancelled = exc
    if cancelled is not None:
        if not task.cancelled() and (failure := task.exception()) is not None:
            logger.warning(
                "studio store: a write ended with %r after its caller was cancelled", failure
            )
        raise cancelled
    return task.result()


class _ConnectFailedError(Exception):
    """Acquiring a connection to the database failed (a pool checkout or a
    connect), raised from what failed, by `StudioStore._transaction`."""


# The faults no retry of a connect mends: a socket path or certificate file
# that is missing, is not a file or may not be read, and a server
# certificate that does not verify.
_PERMANENT_CONNECT_FAULTS = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    ssl.SSLCertVerificationError,
)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """`exc` and every error it was raised from, a driver error's original
    included."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if isinstance(current, DBAPIError) and isinstance(current.orig, BaseException):
            yield from _chain(current.orig)
        current = current.__cause__


def _transient(exc: BaseException) -> bool:
    """Whether `exc` is a fault a retry may outlast, decided by the phase it
    was raised in. A connect (`_ConnectFailedError`) that failed is
    retried whatever failed (a refused, reset or timed-out socket, an
    unreachable host, a name that does not resolve, a TLS failure, the pool
    timing out, too many open files), unless anything in its chain is a
    permanent fault (`_PERMANENT_CONNECT_FAULTS`): the network or the
    server may come back, a missing or unreadable file will not. A
    statement that failed is retried only when the connection was lost
    under it: SQLAlchemy invalidated the connection, or a `ConnectionError`
    is in the chain. A statement or lock timeout, a client-side
    `TimeoutError` and a misuse of the connection raise at once."""
    if isinstance(exc, _ConnectFailedError):
        return not any(isinstance(e, _PERMANENT_CONNECT_FAULTS) for e in _chain(exc))
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return True
    return any(isinstance(e, ConnectionError) for e in _chain(exc))


def _changed_since(full: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The subset of `full` that `baseline` either lacks or holds a
    different value for. `save` and `checkpoint` both diff a freshly
    serialised state against a baseline this way."""
    return {
        key: value for key, value in full.items() if baseline.get(key, _NO_BASELINE_VALUE) != value
    }


def _json_safe(value: Any) -> Any:
    """`value`, proven JSON-serialisable by round-tripping it through
    `json.dumps`/`json.loads`. Raises on a value `json.dumps` refuses."""
    return json.loads(json.dumps(value))


def _serialize_segment(segment: Segment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "owner": segment.owner,
        "expires_at": segment.expires_at,
        "cancel": segment.cancel,
    }


def _deserialize_segment(value: Any) -> Segment | None:
    if not value:
        return None
    return Segment(
        id=value["id"],
        owner=value["owner"],
        expires_at=value["expires_at"],
        cancel=bool(value.get("cancel", False)),
    )


def _serialize_pending(pending: dict[str, _PendingInterrupt]) -> dict[str, Any]:
    return {
        interrupt_id: {
            "adk_id": p.adk_id,
            "kind": p.kind,
            "plan_version": p.plan_version,
            "plan_revision": p.plan_revision,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "options": dict(q.options),
                    "skippable": q.skippable,
                }
                for q in p.questions
            ],
            "skippable": p.skippable,
            "legacy": p.legacy,
            "step_order": list(p.step_order),
            "step_labels": dict(p.step_labels),
            "resuming_segment": p.resuming_segment,
            "answer": p.answer,
            "choices": {
                choice_id: {
                    "active": choice.active,
                    "options": {
                        option_id: {"label": option.label, "status": option.status}
                        for option_id, option in choice.options.items()
                    },
                }
                for choice_id, choice in p.choices.items()
            },
            # `new_plan` is stored beside the card. An instance rolled back to a
            # model without it cannot load a stored card that holds it, and
            # ignores an unknown key of the entry.
            "card": (
                p.card.model_dump(mode="json", exclude_none=True, exclude={"new_plan"})
                if p.card
                else None
            ),
            "card_new_plan": bool(p.card and p.card.new_plan),
            "approved_sent": p.approved_sent,
            "pick_replan": p.pick_replan,
        }
        for interrupt_id, p in pending.items()
    }


def _deserialize_entry(entry: dict[str, Any]) -> _PendingInterrupt:
    pending = _PendingInterrupt(
        adk_id=entry["adk_id"],
        kind=entry["kind"],
        plan_version=entry.get("plan_version"),
        plan_revision=entry.get("plan_revision"),
        step_order=tuple(entry.get("step_order") or ()),
        step_labels=dict(entry.get("step_labels") or {}),
        resuming_segment=entry.get("resuming_segment"),
        answer=entry.get("answer"),
        choices={
            choice_id: _PendingChoice(
                active=choice["active"],
                options={
                    option_id: _PendingOption(label=option["label"], status=option["status"])
                    for option_id, option in choice["options"].items()
                },
            )
            for choice_id, choice in (entry.get("choices") or {}).items()
        },
        card=(
            PlanProposed.model_validate(
                {**entry["card"], "new_plan": True if entry.get("card_new_plan") else None}
            )
            if entry.get("card")
            else None
        ),
        approved_sent=bool(entry.get("approved_sent")),
        pick_replan=bool(entry.get("pick_replan")),
    )
    if "questions" in entry:
        pending.questions = tuple(
            _PendingQuestion(
                id=q["id"],
                question=q["question"],
                options=dict(q["options"]),
                skippable=q["skippable"],
            )
            for q in entry["questions"]
        )
        pending.skippable = entry["skippable"]
        pending.legacy = entry["legacy"]
    elif pending.kind == "elicitation":
        # Saved before questions were kept. The flat `option_ids` lift into
        # one question `q1` whose labels are unknown, and every such frame
        # was skippable as a whole.
        options: dict[str, str | None] = dict.fromkeys(entry.get("option_ids") or ())
        pending.questions = (_PendingQuestion(id="q1", question="", options=options),)
        pending.skippable = True
        pending.legacy = True
    return pending


def _deserialize_pending(data: dict[str, Any]) -> dict[str, _PendingInterrupt]:
    return {interrupt_id: _deserialize_entry(entry) for interrupt_id, entry in data.items()}


def _serialize_durable(state: ConversationState) -> dict[str, Any]:
    """Every durable field of `state` as the flat map the tables store,
    proven JSON-safe. Both a save's candidate delta (before diffing against
    a baseline) and the snapshot later saves diff against."""
    data: dict[str, Any] = {
        _KEY_TURN_INDEX: state.turn_index,
        _KEY_SEQ: state.seq,
        _KEY_TRACE_ID: state.trace_id,
        _KEY_PENDING: _serialize_pending(state.pending),
        # A JSON object's keys are always strings; `plan_version_ids`' int
        # keys are recovered on load.
        _KEY_PLAN_VERSION_IDS: {
            str(version): value for version, value in state.plan_version_ids.items()
        },
        _KEY_LATEST_PLAN_VERSION: state.latest_plan_version,
        _KEY_SEGMENT: _serialize_segment(state.segment) if state.segment is not None else None,
        _KEY_AUTO_ADD: state.auto_add,
        _KEY_TIMED_OUT_CALLS: sorted(state.timed_out_calls),
        _KEY_TURN_USER_MESSAGE: state.turn_user_message,
        _KEY_TURN_USER_WORDS: state.turn_user_words,
        _KEY_TURN_FINAL_TEXT: state.turn_final_text,
        _KEY_TURN_COMPONENTS: state.turn_components,
        _KEY_TURN_RECORDED: state.turn_recorded,
        _KEY_TURN_PLANS: state.turn_plans,
        _KEY_TURN_READINGS: state.turn_readings,
        _KEY_TURN_APPLIED_CHOICE: state.turn_applied_choice,
        _KEY_TURN_CHOICE_BASELINE: state.turn_choice_baseline,
    }
    for index, turn in enumerate(state.turns):
        data[f"{_TURN_ITEM_PREFIX}{index}"] = turn
    for component_id, component_entry in state.components.items():
        data[f"{_COMPONENT_ITEM_PREFIX}{component_id}"] = component_entry
    for component_id, token in state.write_tokens.items():
        data[f"{_WRITE_TOKEN_ITEM_PREFIX}{component_id}"] = token
    for index, approval in enumerate(state.approvals):
        data[f"{_APPROVAL_ITEM_PREFIX}{index}"] = approval.model_dump(mode="json")
    return {key: _json_safe(value) for key, value in data.items()}


def _set_list_item(items: list[Any], index: int, value: Any) -> None:
    """`items[index] = value`, extending `items` first when `index` is past
    its end, the shape every per-item key's index takes, since `turns` and
    `approvals` are append-only."""
    if index < len(items):
        items[index] = value
    else:
        items.extend([None] * (index - len(items)))
        items.append(value)


def _apply_delta(state: ConversationState, delta: dict[str, Any]) -> None:
    """Merge `delta` (a subset of `_serialize_durable`'s keys) onto `state`
    in place. A key this release does not know (one a later release wrote
    into `turn_state`) is logged and ignored."""
    for key, value in delta.items():
        if key == _KEY_TURN_INDEX:
            state.turn_index = value
        elif key == _KEY_SEQ:
            state.seq = value
        elif key == _KEY_TRACE_ID:
            state.trace_id = value
        elif key == _KEY_PENDING:
            state.pending = _deserialize_pending(value)
        elif key == _KEY_PLAN_VERSION_IDS:
            state.plan_version_ids = {int(version): v for version, v in value.items()}
        elif key == _KEY_LATEST_PLAN_VERSION:
            state.latest_plan_version = value
        elif key == _KEY_COMPOSITION:
            state.composition = (
                CompositionManifest.model_validate(value) if value is not None else None
            )
        elif key == _KEY_SEGMENT:
            state.segment = _deserialize_segment(value)
        elif key == _KEY_AUTO_ADD:
            state.auto_add = value
        elif key == _KEY_TIMED_OUT_CALLS:
            state.timed_out_calls = set(value)
        elif key == _KEY_TURN_USER_MESSAGE:
            state.turn_user_message = value
        elif key == _KEY_TURN_USER_WORDS:
            state.turn_user_words = value
        elif key == _KEY_TURN_FINAL_TEXT:
            state.turn_final_text = value
        elif key == _KEY_TURN_COMPONENTS:
            state.turn_components = list(value)
        elif key == _KEY_TURN_RECORDED:
            state.turn_recorded = value
        elif key == _KEY_TURN_PLANS:
            state.turn_plans = list(value)
        elif key == _KEY_TURN_READINGS:
            state.turn_readings = list(value)
        elif key == _KEY_TURN_APPLIED_CHOICE:
            state.turn_applied_choice = dict(value) if value else None
        elif key == _KEY_TURN_CHOICE_BASELINE:
            state.turn_choice_baseline = dict(value or {})
        elif key.startswith(_TURN_ITEM_PREFIX):
            _set_list_item(state.turns, int(key.removeprefix(_TURN_ITEM_PREFIX)), value)
        elif key.startswith(_COMPONENT_ITEM_PREFIX):
            state.components[key.removeprefix(_COMPONENT_ITEM_PREFIX)] = value
        elif key.startswith(_WRITE_TOKEN_ITEM_PREFIX):
            state.write_tokens[key.removeprefix(_WRITE_TOKEN_ITEM_PREFIX)] = value
        elif key.startswith(_APPROVAL_ITEM_PREFIX):
            index = int(key.removeprefix(_APPROVAL_ITEM_PREFIX))
            _set_list_item(state.approvals, index, ApprovalRecord.model_validate(value))
        else:
            logger.warning("studio store: ignoring unknown turn state key %r", key)


class ConversationStale(Exception):  # noqa: N818 -- named for the wire 409
    """`StudioStore.save` found the conversation's turn state moved since this
    instance loaded it: another instance claimed or saved a turn in between,
    or holds a live lease this save would claim over. The `/turn` claim and
    `/interrupt` answer 409."""


class ConversationExists(Exception):  # noqa: N818 -- named for the race it reports
    """`StudioStore.create` lost the race to create the conversation."""


class ConversationHeld(Exception):  # noqa: N818 -- names the state the routes refuse
    """`StudioStore.load` cannot serve the conversation, and no retry will
    until an operator acts, raised by `AdkBridge.reimport`: the
    conversation's ADK ledger session and its rows both changed since the
    last copy (the wire's `conversation_diverged`)."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"conversation {conversation_id!r} is diverged")


# ---------------------------------------------------------------------------
# Rows and columns
# ---------------------------------------------------------------------------


def _jsonb(value: Any) -> str | None:
    """`value` as a jsonb parameter: JSON text, or SQL NULL for `None`."""
    return None if value is None else json.dumps(value)


_CONVERSATION_COLUMNS = (
    "id, version, turn_state, composition, lease_segment_id, lease_owner, "
    "lease_expires_at, lease_cancel, seq_next, seq_reserved"
)


def _row_segment(row: Any) -> Segment | None:
    if row.lease_segment_id is None:
        return None
    return Segment(
        id=row.lease_segment_id,
        owner=row.lease_owner,
        expires_at=row.lease_expires_at,
        cancel=row.lease_cancel,
    )


def _row_flat(row: Any) -> dict[str, Any]:
    """The flat keys `conversations` holds in its own columns."""
    segment = _row_segment(row)
    return {
        **row.turn_state,
        _KEY_SEQ: row.seq_next,
        _KEY_COMPOSITION: row.composition,
        _KEY_SEGMENT: _serialize_segment(segment) if segment is not None else None,
    }


@dataclass(frozen=True)
class _ItemTable:
    """The table one per-item key prefix is stored in: `ident` is the column
    the key's suffix fills, `value` the column its value fills, `ordered_by`
    the order `load` reads its rows in."""

    table: str
    ident: str
    value: str
    ordered_by: str | None
    positional: bool
    json_value: bool = True

    def upsert(self) -> Any:
        value = "CAST(:value AS jsonb)" if self.json_value else ":value"
        return text(
            f"INSERT INTO {self.table} (conversation, {self.ident}, {self.value}) "
            f"VALUES (:pk, :ident, {value}) "
            f"ON CONFLICT (conversation, {self.ident}) "
            f"DO UPDATE SET {self.value} = EXCLUDED.{self.value}"
        )

    def select(self) -> Any:
        order = f" ORDER BY {self.ordered_by}" if self.ordered_by else ""
        return text(
            f"SELECT {self.ident}, {self.value} FROM {self.table} WHERE conversation = :pk{order}"
        )

    def select_idents(self) -> Any:
        """The rows of the idents bound to `:idents`."""
        return text(
            f"SELECT {self.ident}, {self.value} FROM {self.table} "
            f"WHERE conversation = :pk AND {self.ident} = ANY(:idents)"
        )


_ITEM_TABLES = {
    _TURN_ITEM_PREFIX: _ItemTable(
        "conversation_turns", "position", "record", "position", positional=True
    ),
    # `seq` keeps first-insertion order, which the canvas digest's
    # hidden-card list reads as "oldest first".
    _COMPONENT_ITEM_PREFIX: _ItemTable(
        "conversation_components", "component_id", "entry", "seq", positional=False
    ),
    _WRITE_TOKEN_ITEM_PREFIX: _ItemTable(
        "conversation_write_tokens",
        "component_id",
        "token",
        None,
        positional=False,
        json_value=False,
    ),
    _APPROVAL_ITEM_PREFIX: _ItemTable(
        "conversation_approvals", "position", "record", "position", positional=True
    ),
}


@dataclass(frozen=True)
class _DeltaWrite:
    """The statements `_write_delta` runs for one delta, without the row id:
    the `SET` clauses of the `conversations` update and their parameters,
    and each item table's rows by key prefix."""

    sets: tuple[str, ...]
    params: dict[str, Any]
    rows: dict[str, list[dict[str, Any]]]


def _check_column(key: str, value: Any, kind: type | tuple[type, ...]) -> None:
    """Raise `TypeError` when `value` is not of `kind`, the Python type the
    driver binds to the column `key` is written to. None of these columns
    is boolean, so a bool, an `int` to Python, is refused too."""
    if isinstance(value, bool) or not isinstance(value, kind):
        raise TypeError(f"{key} is of type {type(value).__name__}")


def _delta_write(delta: dict[str, Any]) -> _DeltaWrite:
    """The statements that write `delta`'s flat keys. Raises on a value the
    columns cannot hold (a `seq` that is not an integer, a segment without
    an id or whose id, owner or expiry is of another type, a position that
    is not a number, a write token that is not a string, a value
    `json.dumps` refuses) and on a key no column or row holds, so a caller
    that builds the statements without running them learns what running
    them would refuse."""
    sets = ["version = version + 1", "updated_at = now()"]
    params: dict[str, Any] = {}
    scalars = {key: value for key, value in delta.items() if key in _SCALAR_KEYS}
    if scalars:
        sets.append("turn_state = turn_state || CAST(:scalars AS jsonb)")
        params["scalars"] = json.dumps(scalars)
    if _KEY_SEQ in delta:
        _check_column(_KEY_SEQ, delta[_KEY_SEQ], int)
        sets.append("seq_next = :seq_next")
        params["seq_next"] = delta[_KEY_SEQ]
    if _KEY_SEGMENT in delta:
        segment = _deserialize_segment(delta[_KEY_SEGMENT])
        if segment is not None:
            _check_column("segment id", segment.id, str)
            _check_column("segment owner", segment.owner, str)
            _check_column("segment expiry", segment.expires_at, (int, float))
        sets.append(
            "lease_segment_id = :segment_id, lease_owner = :owner, "
            "lease_expires_at = :expires_at, lease_cancel = :cancel"
        )
        params |= {
            "segment_id": segment.id if segment else None,
            "owner": segment.owner if segment else None,
            "expires_at": segment.expires_at if segment else None,
            "cancel": segment.cancel if segment else False,
        }
    rows: dict[str, list[dict[str, Any]]] = {}
    for key, value in delta.items():
        if key in _SCALAR_KEYS or key in (_KEY_SEQ, _KEY_SEGMENT):
            continue
        prefix = next((prefix for prefix in _ITEM_TABLES if key.startswith(prefix)), None)
        if prefix is None:
            raise ValueError(f"studio store: no column or row holds key {key!r}")
        items = _ITEM_TABLES[prefix]
        ident = key.removeprefix(prefix)
        if not items.json_value:
            _check_column(key, value, str)
        rows.setdefault(prefix, []).append(
            {
                "ident": int(ident) if items.positional else ident,
                "value": json.dumps(value) if items.json_value else value,
            }
        )
    return _DeltaWrite(sets=tuple(sets), params=params, rows=rows)


async def _write_delta(conn: AsyncConnection, pk: int, delta: dict[str, Any]) -> int:
    """Write `delta`'s flat keys to their columns and rows, bump the turn
    state's version, and return the new version. The caller holds the row
    lock and decided the write is allowed."""
    return await _run_delta_write(conn, pk, _delta_write(delta))


async def _run_delta_write(conn: AsyncConnection, pk: int, write: _DeltaWrite) -> int:
    """Run `write`'s statements on row `pk` and return the row's new
    version (`_write_delta`)."""
    version = (
        await conn.execute(
            text(
                f"UPDATE conversations SET {', '.join(write.sets)} WHERE id = :pk RETURNING version"
            ),
            {**write.params, "pk": pk},
        )
    ).scalar_one()
    for prefix, rows in write.rows.items():
        await conn.execute(_ITEM_TABLES[prefix].upsert(), [{**row, "pk": pk} for row in rows])
    return version


@dataclass
class _RuntimeEntry:
    """The process-local anchor for one (principal, conversation_id).

    `lock` serialises same-instance requests before any stored data exists,
    including the pair racing to create a new conversation. `pk` and
    `version` are the conversation's row id and the turn-state version this
    instance last read or wrote; `save` compares `version`. `read_at` is the
    database clock's reading (`time_sql`) when the row was last read, which
    `claim_segment` compares the loaded lease against, and `read_stamp` this
    instance's own clock just before that read. `lease_stamp` is this
    instance's clock just before the read or renewal that set the held
    lease's expiry: the claim's `read_stamp`, then each renewal's start, the
    anchor of `StudioStore.lease_deadline`. `held` is the segment
    this instance claimed and has not yet released, the one lease its own
    saves may write under. `seq_reserved` is the stored end of the frame
    numbers reserved for the conversation's segments (`reserve_frames`),
    `claimed_reserved` the end a claim not yet saved reserves, and
    `claim_resumes` whether that claim resumes a turn rather than beginning
    one. `state` is the `ConversationState` this instance keeps live between
    requests; `load` refreshes its durable fields in place, so `lock`,
    `renew_now` and `_telemetry_span` keep their identity across a reload.
    `baseline` is the last durable snapshot known stored, the reference a
    save diffs against. `composition_lock` serialises same-instance
    `/composition` calls on the shared `state` object, separate from `lock`
    since a canvas edit must land while a turn streams. `write_lock`
    serialises this instance's own
    writes on the conversation, so two of them never interleave their
    `state`/`baseline` updates.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    composition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pk: int | None = None
    version: int = 0
    read_at: float = 0.0
    read_stamp: float = 0.0
    lease_stamp: float = 0.0
    held: str | None = None
    seq_reserved: int = 0
    claimed_reserved: int | None = None
    claim_resumes: bool = False
    state: ConversationState | None = None
    baseline: dict[str, Any] = field(default_factory=dict)


class StudioStore:
    """Conversation state, keyed (principal, conversation_id), stored in the
    harness's tables over `engine` (module docstring).

    `turn_specs` reads the spec blocks the open turn's own session
    (`ConversationState.session_id`) holds, which `load` reads back for
    `ConversationState.close_turn`. `adk.reimport` runs before `load` reads
    a conversation (`adk.moved` answers whether ADK has moved since), so a
    conversation the previous release still writes to ADK is read as ADK
    last stored it. `claim_segment`/`renew`/`checkpoint`/`request_cancel` are
    the segment lease API: together they keep at most one instance streaming
    a conversation at a time.

    Lease expiry is decided on the database's clock: `time_sql` is the SQL
    expression every lease statement reads the time from, epoch seconds,
    `extract(epoch FROM now())` by default. A test passes an expression it
    can move. `clock` is the instance's own `time.time`-shaped clock, which
    stamps run records (`harness.run_ledger`) and times the lease deadline
    (`lease_deadline`) the heartbeat and retried writes stop at, both local
    measurements.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        turn_specs: TurnSpecs,
        adk: AdkBridge,
        clock: Callable[[], float] = time.time,
        time_sql: str = DATABASE_NOW,
    ) -> None:
        self._engine = engine
        self._turn_specs = turn_specs
        self._adk = adk
        self._runtime: dict[tuple[str, str], _RuntimeEntry] = {}
        self._clock = clock
        self._time_sql = time_sql
        # Each read of the row carries the database clock's reading with it.
        self._select_conversation = text(
            f"SELECT {_CONVERSATION_COLUMNS}, {time_sql} AS db_now FROM conversations "
            "WHERE principal = :principal AND conversation_id = :conversation_id"
        )
        self._lock_conversation = text(
            f"SELECT {_CONVERSATION_COLUMNS}, {time_sql} AS db_now FROM conversations "
            "WHERE id = :pk FOR UPDATE"
        )

    @property
    def engine(self) -> AsyncEngine:
        """The engine the conversations are stored over; the run ledger
        reads its rows through it."""
        return self._engine

    def now(self) -> float:
        """The instance clock's current reading, for local measurements
        (run record stamps, the lease deadline). Lease expiry is never
        compared against it."""
        return self._clock()

    def current(self, principal: str, conversation_id: str) -> ConversationState | None:
        """This instance's live `ConversationState` for `(principal,
        conversation_id)`, if `create` or `load` has already read one, with
        no database access. The held-in-process accessor `make_recall_part`
        and the recorder read a conversation's in-flight durable state
        through."""
        entry = self._runtime.get((principal, conversation_id))
        return entry.state if entry is not None else None

    def discard_unsaved(self, principal: str, conversation_id: str) -> None:
        """Drop this instance's live state of the conversation, which holds
        changes no save stored (a turn begun, a segment claimed in memory, an
        answer marked), keeping its runtime entry and so its locks. The
        turn's telemetry span ends first, as abandoned
        (`ConversationState.abandon_turn_span`), since nothing left holds it.
        `current` returns None until the next `load`, which reads the row
        afresh. Called after any lease drop (`drop_lease`), whose
        bookkeeping reads the entry this clears."""
        entry = self._runtime.get((principal, conversation_id))
        if entry is not None:
            if entry.state is not None:
                entry.state.abandon_turn_span()
            entry.state = None
            entry.claimed_reserved = None
            entry.held = None

    def lease_deadline(self, principal: str, conversation_id: str) -> float:
        """When, on this instance's clock (`now`), it stops trusting that it
        still holds the lease it claimed on `conversation_id`:
        `LEASE_MARGIN_S` before the lease would lapse, counted from just
        before the read or renewal that set its expiry. The heartbeat ends
        the stream there when renewals keep failing, and a write that keeps
        failing stops retrying there (`_retrying`)."""
        return self._lease_deadline(self._entry(principal, conversation_id))

    def _lease_deadline(self, entry: _RuntimeEntry) -> float:
        return entry.lease_stamp + SEGMENT_TTL_S - LEASE_MARGIN_S

    def lock_for(self, principal: str, conversation_id: str) -> asyncio.Lock:
        """This (principal, conversation_id) pair's lock, creating its
        runtime entry on first use. Callers acquire it BEFORE `load` or
        `create` touch any durable state, so ownership settles while the
        request handler still controls the response."""
        return self._entry(principal, conversation_id).lock

    def composition_lock_for(self, principal: str, conversation_id: str) -> asyncio.Lock:
        """This (principal, conversation_id) pair's composition lock,
        creating its runtime entry on first use. `/composition` serialises
        same-instance calls on this lock, separate from `lock_for`'s turn
        lock, so a canvas edit can land while a turn streams."""
        return self._entry(principal, conversation_id).composition_lock

    def _entry(self, principal: str, conversation_id: str) -> _RuntimeEntry:
        return self._runtime.setdefault((principal, conversation_id), _RuntimeEntry())

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncConnection]:
        """One read-committed transaction. Every write goes through here.
        Acquiring its connection, which may check one out of the pool or
        connect, raises `_ConnectFailedError` from what failed, so a retried
        write can tell a failed connect from a failed statement
        (`_transient`)."""
        try:
            conn = await self._engine.connect()
        except Exception as exc:
            raise _ConnectFailedError(f"no connection to the database: {exc}") from exc
        try:
            async with conn.begin():
                yield conn
        finally:
            await conn.close()

    async def _write(
        self,
        conn: AsyncConnection,
        pk: int,
        delta: dict[str, Any],
        *,
        seq_reserved: int | None = None,
    ) -> int:
        """Write `delta` to the turn state, the canvas, the lease and the item
        rows, and `seq_reserved` when given, and return the row's new version
        (`_write_delta`)."""
        version = await _write_delta(conn, pk, delta)
        if seq_reserved is not None:
            await conn.execute(
                text("UPDATE conversations SET seq_reserved = :reserved WHERE id = :pk"),
                {"pk": pk, "reserved": seq_reserved},
            )
        return version

    async def _write_composition(
        self, conn: AsyncConnection, pk: int, composition: dict[str, Any] | None
    ) -> None:
        """Write the canvas column alone. It does not move the turn state's
        version."""
        await conn.execute(
            text(
                "UPDATE conversations SET composition = CAST(:composition AS jsonb), "
                "updated_at = now() WHERE id = :pk"
            ),
            {"pk": pk, "composition": _jsonb(composition)},
        )

    async def create(self, principal: str, conversation_id: str) -> ConversationState:
        """Insert `conversation_id`'s row. This is the only path that creates
        one (`/turn` on a conversation absent from `load`). Raises
        `ConversationExists` when another instance inserted it first, a race
        the caller must answer for."""
        entry = self._entry(principal, conversation_id)
        stamp = self._clock()
        async with self._transaction() as conn:
            row = (
                await conn.execute(
                    text(
                        "INSERT INTO conversations (principal, conversation_id) "
                        "VALUES (:principal, :conversation_id) ON CONFLICT DO NOTHING "
                        f"RETURNING id, version, seq_reserved, {self._time_sql} AS db_now"
                    ),
                    {"principal": principal, "conversation_id": conversation_id},
                )
            ).first()
        if row is None:
            raise ConversationExists(conversation_id)
        entry.pk, entry.version, entry.read_at = row.id, row.version, row.db_now
        entry.read_stamp = stamp
        entry.held = None
        entry.seq_reserved, entry.claimed_reserved = row.seq_reserved, None
        entry.baseline = {}
        entry.state = ConversationState(conversation_id=conversation_id)
        entry.state.lock = entry.lock
        return entry.state

    async def _read(
        self, principal: str, conversation_id: str
    ) -> tuple[Any, dict[str, Any]] | None:
        """The conversation's row, with the database clock's reading as
        `db_now`, and its flat durable keys, read in one snapshot. None when
        no row holds it."""
        async with self._engine.connect() as plain:
            conn = await plain.execution_options(isolation_level="REPEATABLE READ")
            async with conn.begin():
                row = (
                    await conn.execute(
                        self._select_conversation,
                        {"principal": principal, "conversation_id": conversation_id},
                    )
                ).first()
                if row is None:
                    return None
                flat = _row_flat(row)
                for prefix, items in _ITEM_TABLES.items():
                    for ident, value in await conn.execute(items.select(), {"pk": row.id}):
                        flat[f"{prefix}{ident}"] = value
        return row, flat

    async def adk_moved(self, principal: str, conversation_id: str) -> bool:
        """Whether ADK's ledger session for `conversation_id` moved since
        `load` last copied or refused it (`AdkBridge.moved`)."""
        return await self._adk.moved(principal, conversation_id)

    def adk_convert_specs(self, session_id: str, specs: dict[str, Any]) -> dict[str, Any]:
        """`specs` from turn session `session_id`, converted
        (`AdkBridge.convert_specs`)."""
        return self._adk.convert_specs(session_id, specs)

    async def load(self, principal: str, conversation_id: str) -> ConversationState | None:
        """Reload `conversation_id`'s durable fields, after `adk.reimport`
        has brought its rows up to date with ADK. `None` when the store does
        not hold the conversation. Raises what `adk.reimport` raises:
        `ConversationHeld`, or the failure of a copy, before anything is
        read or written. An open turn's spec blocks are read through
        `turn_specs` and passed through `adk.convert_specs`, which reports a
        block that does not convert."""
        entry = self._entry(principal, conversation_id)
        await self._adk.reimport(principal, conversation_id)
        stamp = self._clock()
        stored = await self._read(principal, conversation_id)
        if stored is None:
            return None
        row, flat = stored
        entry.pk, entry.version, entry.read_at = row.id, row.version, row.db_now
        entry.read_stamp = stamp
        entry.seq_reserved, entry.claimed_reserved = row.seq_reserved, None
        durable = ConversationState(conversation_id=conversation_id)
        _apply_delta(durable, flat)
        if entry.state is None:
            durable.lock = entry.lock
            entry.state = durable
        else:
            for f in dataclasses.fields(ConversationState):
                if f.name not in _PROCESS_LOCAL_FIELDS:
                    setattr(entry.state, f.name, getattr(durable, f.name))
        entry.baseline = _serialize_durable(entry.state)
        segment = entry.state.segment
        if segment is None or segment.id != entry.held:
            entry.held = None
        if entry.state.turn_index and not entry.state.turn_recorded:
            session_id = entry.state.session_id
            entry.state.turn_specs = self._adk.convert_specs(
                session_id, await self._turn_specs(principal, session_id)
            )
        return entry.state

    def claim_segment(self, principal: str, conversation_id: str) -> str:
        """Claim a fresh segment on the live state already loaded on this
        instance (`create` or `load` must have run first). Raises
        `SegmentBusy` when the loaded lease had not lapsed by the database
        clock's reading taken with the load. Otherwise writes a fresh segment
        into `entry.state`, expiring `SEGMENT_TTL_S` after that reading, and
        returns its id.

        The claim also takes the segment's first block of frame numbers and
        starts the live state's `seq` there: `0` up to `SEQ_BLOCK` for a turn
        `begin_turn` just opened, else the block after every number reserved
        before, so a stream resuming the turn never repeats a number an
        earlier segment sent.

        Does no I/O: the caller's own claim save (the `/turn` claim or the
        `/interrupt` save) persists it, and refuses it when the stored lease
        is live by then."""
        entry = self._runtime[(principal, conversation_id)]
        assert entry.state is not None
        live = entry.state.segment
        if live is not None and live.expires_at > entry.read_at:
            raise SegmentBusy(retry_after_seconds=math.ceil(live.expires_at - entry.read_at))
        segment = Segment(
            id=_wire_id("seg"), owner=_OWNER_ID, expires_at=entry.read_at + SEGMENT_TTL_S
        )
        entry.state.segment = segment
        entry.lease_stamp = entry.read_stamp
        entry.claim_resumes = entry.state.turn_index == entry.baseline.get(_KEY_TURN_INDEX)
        entry.state.seq = entry.seq_reserved if entry.claim_resumes else 0
        entry.claimed_reserved = entry.state.seq + SEQ_BLOCK
        return segment.id

    async def save(self, principal: str, conversation_id: str) -> None:
        """Write this instance's changed durable fields, compare-and-swapped
        on the turn state's version. Only the keys whose serialised value
        changed since the last load or save on this instance are written; a
        save with nothing changed writes nothing.

        Raises `ConversationStale` when another instance's claim, save or
        checkpoint moved the version since this instance last read or wrote
        it, or when the stored lease is live on the database clock and is
        neither the segment this save claims nor the one this instance
        holds: a claim over a live lease (a renewal does not move the
        version, so the lease is compared itself), and a lease-less save
        while another instance streams, whose turn state that stream alone
        writes. A renewal, a cancel, a canvas edit or a run-ledger write does
        not move the version. A save that persists `claim_segment`'s claim
        also stores the frame numbers it reserved. A claim that resumes a
        turn starts past the stored reserved end when that moved since the
        load: the lapsed stream the claim takes over reserved more numbers
        before its lease lapsed, and may have sent them.
        """
        entry = self._runtime[(principal, conversation_id)]
        assert entry.pk is not None and entry.state is not None
        async with entry.write_lock:
            delta = _changed_since(_serialize_durable(entry.state), entry.baseline)
            if not delta:
                return
            claimed = _deserialize_segment(delta.get(_KEY_SEGMENT))
            reserved = entry.claimed_reserved if claimed is not None else None
            async with self._transaction() as conn:
                row = (await conn.execute(self._lock_conversation, {"pk": entry.pk})).first()
                if row is None or row.version != entry.version:
                    raise ConversationStale(conversation_id)
                stored_lease = _row_segment(row)
                if (
                    stored_lease is not None
                    and stored_lease.expires_at > row.db_now
                    and stored_lease.id not in {entry.held, claimed.id if claimed else None}
                ):
                    raise ConversationStale(conversation_id)
                if (
                    claimed is not None
                    and entry.claim_resumes
                    and row.seq_reserved > entry.state.seq
                ):
                    entry.state.seq = row.seq_reserved
                    reserved = row.seq_reserved + SEQ_BLOCK
                    delta = _changed_since(_serialize_durable(entry.state), entry.baseline)
                version = await self._write(conn, entry.pk, delta, seq_reserved=reserved)
            entry.version = version
            entry.baseline.update(delta)
            if claimed is not None:
                entry.held = claimed.id
            if reserved is not None:
                entry.seq_reserved = reserved
                entry.claimed_reserved = None

    async def reserve_frames(self, principal: str, conversation_id: str, segment_id: str) -> None:
        """Keep the frame numbers the live stream sends inside its
        reservation. With fewer than `_SEQ_HEADROOM` numbers left before the
        reserved end, push the end to `SEQ_BLOCK` past the next number,
        fenced on `segment_id`; otherwise do nothing and touch no storage.
        `harness.studio` calls it before relaying each frame, so a stream
        that takes the turn over after this one dies starts past every
        number this one could have sent. Raises `SegmentLost` when
        `segment_id` is no longer the stored lease, or its lease has lapsed by
        the database clock: another instance may have loaded the reserved end
        to claim the turn from, so a lapsed stream extending it would hand
        out numbers that claim also uses. A transient database fault is
        retried until the lease's deadline (`_retrying`); the write takes the
        greater of the stored end and its own, so a retry whose first commit
        landed writes the same end."""
        entry = self._runtime[(principal, conversation_id)]
        assert entry.pk is not None and entry.state is not None
        if entry.state.seq + _SEQ_HEADROOM < entry.seq_reserved:
            return
        await self._retrying(
            entry,
            conversation_id,
            lambda _retried: self._reserve(entry, conversation_id, segment_id),
        )

    async def _reserve(self, entry: _RuntimeEntry, conversation_id: str, segment_id: str) -> None:
        """`reserve_frames`' write."""
        assert entry.pk is not None and entry.state is not None
        async with entry.write_lock:
            reserved = entry.state.seq + SEQ_BLOCK
            async with self._transaction() as conn:
                row = (
                    await conn.execute(
                        text(
                            "UPDATE conversations SET seq_reserved = "
                            "GREATEST(seq_reserved, :reserved) "
                            "WHERE id = :pk AND lease_segment_id = :segment_id "
                            f"AND lease_expires_at > {self._time_sql} "
                            "RETURNING seq_reserved"
                        ),
                        {"pk": entry.pk, "segment_id": segment_id, "reserved": reserved},
                    )
                ).first()
            if row is None:
                raise SegmentLost(
                    f"conversation {conversation_id!r} segment {segment_id!r} is not a live lease"
                )
            entry.seq_reserved = row.seq_reserved

    async def mutate_composition(
        self,
        principal: str,
        conversation_id: str,
        mutate: Callable[
            [CompositionManifest | None, dict[str, dict[str, Any]]],
            CompositionManifest | JSONResponse,
        ],
        *,
        component_ids: Collection[str] = (),
    ) -> CompositionManifest | JSONResponse | None:
        """`/composition`'s canvas write, unfenced, since a user's edit lands
        whoever streams the conversation. After `adk.reimport` has brought
        the rows up to date, one transaction locks the conversation's row,
        reads its stored canvas and the digest entries
        (`conversation_components`) of the components `component_ids`
        names, none when it names none, hands both to `mutate`, and writes
        what `mutate` returns. Every other canvas write, from this instance or
        another, waits on the row lock, so each change applies to the canvas
        as stored and none is lost. The canvas has its own column and this
        write does not move the turn state's version. It reads nothing of
        this instance's live state, which a stream may be changing, so it
        needs no `load`.

        `mutate` returning the manifest it was handed (a no-op) or a
        `JSONResponse` (a refusal) writes nothing and returns it. Any other
        manifest is written and returned. Returns None, and calls nothing,
        when the store does not hold the conversation. Raises what
        `adk.reimport` raises. This instance's live state, when it holds
        one, takes the canvas read or written unless it already holds a
        later revision.
        """
        await self._adk.reimport(principal, conversation_id)
        entry = self._entry(principal, conversation_id)
        components_table = _ITEM_TABLES[_COMPONENT_ITEM_PREFIX]
        async with entry.write_lock:
            async with self._transaction() as conn:
                row = (
                    await conn.execute(
                        text(
                            "SELECT id, composition FROM conversations "
                            "WHERE principal = :principal AND conversation_id = :conversation_id "
                            "FOR UPDATE"
                        ),
                        {"principal": principal, "conversation_id": conversation_id},
                    )
                ).first()
                if row is None:
                    return None
                manifest = (
                    CompositionManifest.model_validate(row.composition) if row.composition else None
                )
                components: dict[str, dict[str, Any]] = {}
                if component_ids:
                    rows = await conn.execute(
                        components_table.select_idents(),
                        {"pk": row.id, "idents": list(component_ids)},
                    )
                    components = dict(rows.tuples().all())
                result = mutate(manifest, components)
                canvas = manifest
                if not isinstance(result, JSONResponse) and result is not manifest:
                    await self._write_composition(conn, row.id, result.model_dump(mode="json"))
                    canvas = result
            live = entry.state
            if (
                live is not None
                and canvas is not None
                and (live.composition is None or live.composition.revision <= canvas.revision)
            ):
                live.composition = canvas
            return result

    async def change_canvas[T](
        self,
        principal: str,
        conversation_id: str,
        change: Callable[[ConversationState], T],
    ) -> T:
        """The write of a card change the turn makes: each card it lands,
        holds, swaps, revises or removes. A checkpoint of the whole
        conversation, fenced on the segment the live state holds, with
        `change` applied: the canvas `change` produces and every durable
        field that differs from the last stored snapshot commit in one
        transaction (`_fenced_save`), so a stored card always has its digest
        entry, write token and turn line beside it.

        Raises `SegmentLost` and writes nothing when the stored lease is not
        that segment, so a stream another instance took over lands no card.
        Otherwise `change` runs on a copy of the live state whose canvas is
        the canvas as stored. It must replace what it changes (a digest
        entry included) rather than mutate it in place, since the copy is
        one level deep. The fields `change` changed are assigned to the live
        state only once the write has committed, so a write that fails
        leaves the live state as it was. A transient database fault is
        retried until the lease's deadline (`_retrying`). Returns what
        `change` returns.
        """
        entry = self._runtime[(principal, conversation_id)]
        state = entry.state
        assert entry.pk is not None and state is not None
        segment_id = state.segment.id if state.segment is not None else None
        if segment_id is None:
            raise SegmentLost(f"conversation {conversation_id!r} holds no segment")
        return await self._retrying(
            entry,
            conversation_id,
            lambda retried: self._fenced_save(
                entry, conversation_id, segment_id, change, retried=retried
            ),
        )

    async def _retrying[T](
        self,
        entry: _RuntimeEntry,
        conversation_id: str,
        attempt: Callable[[bool], Awaitable[T]],
    ) -> T:
        """Run `attempt`, a write fenced on the live segment, retrying a
        transient database fault (`_transient`) with backoff until the
        lease's deadline (`lease_deadline`), then raising
        `DatabaseUnavailable` from the last fault: by then another instance
        may claim the conversation, though none has. Each retry sleeps
        with no lock held, so the heartbeat's renewal is never queued behind
        it. An attempt whose commit landed but whose acknowledgement was
        lost is retried as well, so every attempt must leave the stored row
        as one attempt would. `attempt` is told whether an earlier attempt
        met a fault, so it may have committed."""
        delay = _RETRY_FIRST_DELAY_S
        retried = False
        while True:
            try:
                return await attempt(retried)
            except Exception as exc:
                if not _transient(exc):
                    raise
                if self._clock() + delay >= self._lease_deadline(entry):
                    raise DatabaseUnavailable(
                        f"conversation {conversation_id!r} could not be written before its "
                        f"lease's deadline: {exc}"
                    ) from exc
                logger.warning(
                    "studio store: conversation %s write failed, retrying in %ss: %s",
                    conversation_id,
                    delay,
                    exc,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RETRY_MAX_DELAY_S)
            retried = True

    async def _fenced_save[T](
        self,
        entry: _RuntimeEntry,
        conversation_id: str,
        segment_id: str,
        change: Callable[[ConversationState], T],
        *,
        retried: bool = False,
    ) -> T:
        """`checkpoint` and `change_canvas`'s one write. Under the row lock,
        fenced on `segment_id`: run `change` on a one-level copy of the live
        state holding the canvas as stored, then write the copy's canvas when
        `change` replaced it, and every durable key whose value differs from
        `entry.baseline`, in one transaction. Once it commits, record the
        written keys in the baseline and the version, and assign the canvas
        read back and each field `change` changed to the live state.

        The copy is taken under the row lock with no await between it and
        the write, so the write holds the live state as it then stood. A
        field the live stream changes during the write keeps its new value,
        which differs from the baseline and goes out with the next write.

        A `retried` write that clears the lease (`release`'s) and finds the
        row holding no lease at the version the earlier attempt's commit
        would have written (`entry.version` + 1) was saved by that attempt,
        whose acknowledgement was lost. It writes nothing and records the
        write as made. Any other row, another instance's claim and release
        since included, fences the write out."""
        state = entry.state
        assert entry.pk is not None and state is not None
        async with entry.write_lock:
            async with self._transaction() as conn:
                row = (await conn.execute(self._lock_conversation, {"pk": entry.pk})).first()
                if row is None:
                    raise SegmentLost(f"conversation {conversation_id!r} is no longer stored")
                fenced = _row_segment(row)
                landed = (
                    retried
                    and state.segment is None
                    and fenced is None
                    and row.version == entry.version + 1
                )
                if not landed and (fenced is None or fenced.id != segment_id):
                    raise SegmentLost(
                        f"conversation {conversation_id!r} segment {segment_id!r} is not the lease"
                    )
                stored = (
                    CompositionManifest.model_validate(row.composition) if row.composition else None
                )
                base = _shallow_copy(state)
                work = _shallow_copy(state)
                work.composition = stored
                result = change(work)
                changed = _changed_since(_serialize_durable(work), entry.baseline)
                if landed:
                    logger.info(
                        "studio store: conversation %s segment %s was saved by an attempt "
                        "whose acknowledgement was lost",
                        conversation_id,
                        segment_id,
                    )
                    version = row.version
                else:
                    if work.composition is not stored:
                        await self._write_composition(
                            conn,
                            entry.pk,
                            work.composition.model_dump(mode="json")
                            if work.composition is not None
                            else None,
                        )
                    version = await self._write(conn, entry.pk, changed) if changed else row.version
            entry.version = version
            entry.baseline.update(changed)
            if work.segment is None:
                entry.held = None
            for f in dataclasses.fields(ConversationState):
                value = getattr(work, f.name)
                before = getattr(base, f.name)
                if value != before if isinstance(value, _CONTAINERS) else value is not before:
                    setattr(state, f.name, value)
            return result

    async def renew(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        *,
        timeout: float = 5.0,
        consume_cancel: bool = False,
    ) -> Literal["ok", "cancelled", "lost"]:
        """Push `segment_id`'s expiry to `SEGMENT_TTL_S` past the database
        clock's reading in one conditional update of the lease columns, which
        never disturbs whatever the live stream is doing to `entry.state` and
        never writes the rest of the conversation. That is `checkpoint`'s
        job.

        `"lost"` when the stored segment is no longer `segment_id` (another
        instance took over after this one's TTL lapsed); the caller ends the
        stream fatally. `"cancelled"` when it is still `segment_id` but its
        `cancel` flag is set, still renewed, so the flag keeps being observed
        until the caller reacts to it. With `consume_cancel` the same update
        clears the flag, so a later `request_cancel` sets it again and the
        next renewal reports that Stop on its own. `"ok"` otherwise. A
        database error propagates, as does a `TimeoutError` when the renewal
        takes longer than `timeout`; the caller's own heartbeat decides how
        long to keep trying.

        Runs to the end even when the caller is cancelled
        (`_run_to_completion`), since the heartbeat that calls it is
        cancelled whenever its stream ends.
        """
        return await _run_to_completion(
            self._renew(
                principal,
                conversation_id,
                segment_id,
                timeout=timeout,
                consume_cancel=consume_cancel,
            )
        )

    async def _renew(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        *,
        timeout: float,
        consume_cancel: bool,
    ) -> Literal["ok", "cancelled", "lost"]:
        """`renew`'s write, run as its own task."""
        stamp = self._clock()
        async with asyncio.timeout(timeout):
            entry = self._entry(principal, conversation_id)
            async with entry.write_lock:
                async with self._transaction() as conn:
                    row = (
                        await conn.execute(
                            text(
                                "WITH held AS ("
                                " SELECT id, lease_cancel FROM conversations"
                                " WHERE principal = :principal"
                                " AND conversation_id = :conversation_id"
                                " AND lease_segment_id = :segment_id FOR UPDATE)"
                                " UPDATE conversations AS c"
                                f" SET lease_expires_at = {self._time_sql} + :ttl,"
                                " lease_cancel = held.lease_cancel AND NOT :consume_cancel"
                                " FROM held WHERE c.id = held.id"
                                " RETURNING held.lease_cancel AS was_cancelled, c.lease_owner,"
                                " c.lease_cancel, c.lease_expires_at"
                            ),
                            {
                                "principal": principal,
                                "conversation_id": conversation_id,
                                "segment_id": segment_id,
                                "ttl": SEGMENT_TTL_S,
                                "consume_cancel": consume_cancel,
                            },
                        )
                    ).first()
                if row is None:
                    return "lost"
                entry.lease_stamp = max(entry.lease_stamp, stamp)
                self._note_segment(
                    entry,
                    Segment(
                        id=segment_id,
                        owner=row.lease_owner,
                        expires_at=row.lease_expires_at,
                        cancel=row.lease_cancel,
                    ),
                )
                return "cancelled" if row.was_cancelled else "ok"

    def _note_segment(self, entry: _RuntimeEntry, segment: Segment) -> None:
        """Record a lease write in this instance's baseline, and in its live
        state when that still holds the same segment."""
        entry.baseline[_KEY_SEGMENT] = _serialize_segment(segment)
        live = entry.state
        if live is not None and live.segment is not None and live.segment.id == segment.id:
            live.segment = segment

    async def checkpoint(self, principal: str, conversation_id: str, segment_id: str) -> None:
        """The save of a streaming segment: this instance's changed keys,
        fenced on `segment_id`. Runs before every actionable pause frame and
        before `done` (never `message.complete`), and once more, with the
        segment cleared, at the end of the segment (`release`).

        Under the row lock: raises `SegmentLost` and writes nothing when the
        stored lease is not `segment_id` (another instance's claim outlived
        this one's TTL). The diff is computed once the row is locked, with
        no await between it and the write (`_fenced_save`, the write
        `change_canvas` shares). While the lease is held, this stream is the
        turn state's one writer (`save` refuses every other), so the diff is
        written as it stands. The live canvas is read back from the row, so
        a `/composition` edit another instance made since this one last
        wrote the canvas reaches the stream's next frame. A transient
        database fault is retried until the lease's deadline (`_retrying`).
        """
        entry = self._runtime[(principal, conversation_id)]
        assert entry.pk is not None and entry.state is not None
        await self._retrying(
            entry,
            conversation_id,
            lambda retried: self._fenced_save(
                entry, conversation_id, segment_id, lambda _work: None, retried=retried
            ),
        )

    async def release(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float = 5.0
    ) -> None:
        """The end-of-segment save: drop this segment's still-resuming
        pending entries, unless the stream stopped reading it while it ran
        (`ConversationState.unfinished_segment`), so a retry of the same
        answer carries the turn on; clear the segment if it is still ours; then
        `checkpoint`. Runs to the end even when the caller is cancelled
        (`_run_to_completion`), since it runs in a `finally` on
        client-disconnect and generator-finaliser paths, where a disconnect
        cancels the task running it (`_unwind`'s `Task.cancel`). Cancelled
        mid-checkpoint, the segment would stay live until its TTL. The
        timeout cancels the checkpoint once (`asyncio.timeout`); anyio's
        `move_on_after` would re-cancel at every await until its scope
        exits, cutting short the rollback and connection close.

        Never raises anything but the caller's own cancellation: a
        `checkpoint` that cannot land (fenced out by a takeover, a database
        unreachable until the lease's deadline, another database error, or
        the timeout above) is logged, each as what it is, and swallowed. The segment
        then expires by its own TTL, which is what makes it safe to give up
        here.
        """
        entry = self._runtime.get((principal, conversation_id))
        if entry is None or entry.state is None:
            return
        await _run_to_completion(
            self._save_release(entry.state, principal, conversation_id, segment_id, timeout=timeout)
        )

    async def _save_release(
        self,
        state: ConversationState,
        principal: str,
        conversation_id: str,
        segment_id: str,
        *,
        timeout: float,
    ) -> None:
        """`release`'s save, run as its own task."""
        try:
            async with asyncio.timeout(timeout):
                if state.unfinished_segment != segment_id:
                    state.drop_resuming(segment_id)
                if state.segment is not None and state.segment.id == segment_id:
                    state.segment = None
                try:
                    await self.checkpoint(principal, conversation_id, segment_id)
                except SegmentLost:
                    logger.warning(
                        "studio store release: conversation %s segment %s was already "
                        "taken over; nothing saved",
                        conversation_id,
                        segment_id,
                    )
                except DatabaseUnavailable as exc:
                    logger.error(
                        "studio store release: conversation %s segment %s could not reach the "
                        "database before the lease's deadline; nothing saved, and the lease "
                        "expires by its TTL: %s",
                        conversation_id,
                        segment_id,
                        exc.__cause__,
                    )
                except Exception:
                    logger.exception(
                        "studio store release: conversation %s failed to persist",
                        conversation_id,
                    )
        except TimeoutError:
            logger.warning(
                "studio store release on conversation %s did not complete within %ss",
                conversation_id,
                timeout,
            )

    async def drop_lease(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float = 5.0
    ) -> None:
        """Clear the stored lease when it is still `segment_id`, and nothing
        else: one conditional update of the lease columns, which leaves the
        turn state and its version as the claim save stored them. The
        release of a claim whose stream never started (a route failing
        between its claim save and its response), so a retry of the same
        request claims at once and finds what the claim save stored, a
        resuming pending entry included. Whatever the failed request changed
        in this instance's live state after its claim save stays unsaved;
        the retry's `load` reads the row afresh.

        Runs to the end even when the caller is cancelled, and never raises
        anything but that cancellation, as `release`: a clear that cannot
        land is logged, and the lease then expires by its own TTL.
        """
        await _run_to_completion(
            self._drop_lease(principal, conversation_id, segment_id, timeout=timeout)
        )

    async def _drop_lease(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float
    ) -> None:
        """`drop_lease`'s write, run as its own task."""
        entry = self._entry(principal, conversation_id)
        try:
            async with asyncio.timeout(timeout), entry.write_lock:
                async with self._transaction() as conn:
                    await conn.execute(
                        text(
                            "UPDATE conversations SET lease_segment_id = NULL, "
                            "lease_owner = NULL, lease_expires_at = NULL, lease_cancel = false "
                            "WHERE principal = :principal "
                            "AND conversation_id = :conversation_id "
                            "AND lease_segment_id = :segment_id"
                        ),
                        {
                            "principal": principal,
                            "conversation_id": conversation_id,
                            "segment_id": segment_id,
                        },
                    )
                if entry.held == segment_id:
                    entry.held = None
                    entry.baseline[_KEY_SEGMENT] = None
                live = entry.state
                if live is not None and live.segment is not None and live.segment.id == segment_id:
                    live.segment = None
        except Exception:
            logger.exception(
                "studio store: conversation %s failed to drop lease %s; it expires by its TTL",
                conversation_id,
                segment_id,
            )

    async def request_cancel(self, principal: str, conversation_id: str) -> bool:
        """Set the cancel flag on `conversation_id`'s live segment, whoever
        owns it, this instance included; the segment's heartbeat reads it at
        its next renewal. One conditional update of the lease columns.

        Returns whether a live segment was found and signalled: `False`
        when there is none, or the one on record has already expired by the
        database clock. The caller falls back to whatever cancellation means
        with no active segment.
        """
        entry = self._entry(principal, conversation_id)
        async with entry.write_lock:
            async with self._transaction() as conn:
                row = (
                    await conn.execute(
                        text(
                            "UPDATE conversations SET lease_cancel = true "
                            "WHERE principal = :principal AND conversation_id = :conversation_id "
                            "AND lease_segment_id IS NOT NULL "
                            f"AND lease_expires_at > {self._time_sql} "
                            "RETURNING lease_segment_id, lease_owner, lease_expires_at"
                        ),
                        {"principal": principal, "conversation_id": conversation_id},
                    )
                ).first()
            if row is None:
                return False
            self._note_segment(
                entry,
                Segment(
                    id=row.lease_segment_id,
                    owner=row.lease_owner,
                    expires_at=row.lease_expires_at,
                    cancel=True,
                ),
            )
            return True

    @asynccontextmanager
    async def fenced(
        self, principal: str, conversation_id: str, segment_id: str
    ) -> AsyncIterator[tuple[AsyncConnection, int]]:
        """A transaction that holds `segment_id`'s lease row `FOR SHARE` for
        its whole length, and the conversation's row id. A takeover's claim
        waits for it to commit. Raises `SegmentLost` when `segment_id` is not
        the stored lease, before anything is written."""
        async with self._transaction() as conn:
            pk = (
                await conn.execute(
                    text(
                        "SELECT id FROM conversations WHERE principal = :principal "
                        "AND conversation_id = :conversation_id "
                        "AND lease_segment_id = :segment_id FOR SHARE"
                    ),
                    {
                        "principal": principal,
                        "conversation_id": conversation_id,
                        "segment_id": segment_id,
                    },
                )
            ).scalar_one_or_none()
            if pk is None:
                raise SegmentLost(
                    f"conversation {conversation_id!r} segment {segment_id!r} is not the lease"
                )
            yield conn, pk

    async def run_statuses(
        self, principal: str, conversation_id: str, *, conversation: ConversationState
    ) -> dict[str, RunStatus]:
        """The run ledger's status of each run `conversation`'s plan records
        await (`ConversationState.awaited_runs`), by run id, read off the
        stored records without validating the rest. Not read at all when no
        record awaits a run."""
        awaited = conversation.awaited_runs()
        if not awaited:
            return {}
        stored = await self._stored_statuses(principal, conversation_id, awaited)
        return {
            run_id: RunStatus(status) for run_id, status in stored.items() if status in RunStatus
        }

    async def _stored_statuses(
        self, principal: str, conversation_id: str, run_ids: Iterable[str]
    ) -> dict[str, Any]:
        """The status each named run's stored record holds, by run id."""
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT r.run_id, r.record ->> 'status' AS status FROM runs r "
                    "JOIN conversations c ON c.id = r.conversation "
                    "WHERE c.principal = :principal AND c.conversation_id = :conversation_id "
                    "AND r.run_id = ANY(:run_ids)"
                ),
                {
                    "principal": principal,
                    "conversation_id": conversation_id,
                    "run_ids": list(run_ids),
                },
            )
            return {row.run_id: row.status for row in rows}
