"""`StudioStore`: each conversation's durable state in the harness's own tables.

A conversation is one `conversations` row (defined in
`harness/db/migrations/0001_conversations.sql`) and its per-item rows: closed turns, the digest
entry and write token of each component, and approvals. Every durable field of a
`ConversationState` lives there, which is everything but the identity field `conversation_id`
and the process-local fields `_PROCESS_LOCAL_FIELDS` names, so any instance can load a
conversation and continue it. The store keeps only runtime objects and each instance's own live
`ConversationState` in process, like `harness.provenance.ProvenanceStore`.

The row's column groups each have their own writer, so writers that touch
different groups never conflict:

- `save` writes the turn state compare-and-swapped on the row's `version`
  (`ConversationStale` when another instance moved it, or holds a live
  lease).
- `close_segment` is a stream's closing write: fenced on the segment, it
  stores the turn state and clears the lease, and closes the turn
  `cancelled`, record and cursor, when a Stop is live for it, in one
  transaction.
- `load` records a turn whose cursor closed while its record stayed open,
  with the error its stream told the client (`closed_turn_error`),
  compare-and-swapped on `version` and only while no lease is live.
- `change_canvas` is a write fenced on the segment (`SegmentLost` when
  another instance holds the lease) with a card change applied: each card
  the turn lands, holds, swaps, revises or removes commits with its canvas
  block, every other changed durable field and a drawn card's frame lineage
  (`conversation_card_lineage`, read back by `card_lineage`) in one
  transaction, fenced on the segment.
- `mutate_composition` writes the canvas column alone for each
  `/composition` edit, unfenced, and leaves `version` alone.
- `renew` and `request_cancel` each update the lease columns in one
  conditional statement and leave `version` alone. `request_cancel` sets
  the boolean `lease_cancel`, as does any writer that knows only that flag.

A Stop names a turn. The next write of the lease columns after
`request_cancel` (a renewal, a claim save, a closing write, a dropped lease)
clears `lease_cancel` and records the Stop as a fresh `stop_id` naming the
row's turn index as stored before that write (`stop_turn`), in the same
statement. Whether a Stop is live for the row's turn is one predicate
(`StopState.live`). Every reader locks the row first and reads the turn's
cursor in a later statement, so it sees every write committed before its
lock. Closing the turn ends its Stop, whatever the status, and no write marks
a Stop read. A claim save that starts or resumes work refuses while a Stop is
live (`StopPending`), so the turn index does not move past a live Stop.

Every canvas writer holds the row lock, so each applies its change to the
canvas as stored. Lease expiry is compared against the database's clock in
SQL, so instances whose clocks disagree agree on when a lease lapses.

Each write is one transaction. The writes a live stream makes between its
renewals (`change_canvas`) retry a transient database fault
until the lease's deadline (`StudioStore.lease_deadline`), then raise
`DatabaseUnavailable`. The other writes do not retry. The run ledger
(`harness.run_ledger`) writes inside `fenced`.

Each write that commits a turn's close writes the turn's `studio.turn` log
line (`harness.studio_state.TURN_LOG_EVENT`) and its `turn` outcome
observation (`common.telemetry.record_turn_outcome`), in the task that
commits it, once the commit returns (`_committed_lines`): `close_segment`'s
closing write, a `/turn` claim save that closes the turn left open as
`abandoned`, a `/cancel` save that closes a paused turn `cancelled`, and
`load`'s record of a turn whose cursor closed. A close is a turn record the
row did not hold before the write, read against this instance's baseline,
the row as its last read or write left it, so a turn writes one close line.
A write that carries a line and whose COMMIT raises is confirmed by its
own transaction's status (`StudioStore._committing_lines`): one that
committed writes its lines and returns as one that was acknowledged, and
any other writes none. A canvas write's retry that finds it landed writes
its lines (`_resolve_canvas_writes`). A pause is the turn's cursor write, so
its line comes from that commit (`harness.turn.store.TurnStore.pause`).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import math
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, NamedTuple

from common.logging import JSON_FIELDS
from common.telemetry import deploy_release, record_turn_outcome, start_telemetry_flush
from compiler.surface import transient
from executor.ledger import RunStatus
from fastapi.responses import JSONResponse
from sqlalchemy import ARRAY, Text, bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from harness.approval import ApprovalRecord
from harness.db.best_effort import execute_in_savepoint
from harness.db.commit import (
    committing,
    run_to_completion,
    transaction_committed,
)
from harness.db.jsonb import json_or_none, to_json_param
from harness.fatal_codes import recorded_code
from harness.graph.state import (
    ASK_FOR_QUESTION_KEY,
    NEW_QUESTION_KEY,
    ORCHESTRATOR_APPLIED_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
)
from harness.provenance import CardLineage, CardLineageDecodeError, lineage_from_json
from harness.run_ledger import end_unfinished
from harness.studio_state import (
    SEGMENT_TTL_S,
    TURN_LOG_EVENT,
    ConversationState,
    DatabaseUnavailable,
    Segment,
    SegmentBusy,
    SegmentLost,
    TurnUsage,
    _PendingChoice,
    _PendingInterrupt,
    _PendingOption,
    _PendingQuestion,
    _recorded_reply,
    _wire_id,
)
from harness.tool_contracts import HarnessCode
from harness.turn.store import (
    CLOSED,
    Cursor,
    CursorStale,
    TurnStatus,
    close_unleased_on,
    cursor_at,
    finish_on,
)
from harness.wire import STUDIO_FRAME_ADAPTER, CompositionManifest, ErrorCode, PlanProposed

logger = logging.getLogger(__name__)

# This process's id, carried on every segment it claims for logs only.
# Fencing compares segment ids.
_OWNER_ID = uuid.uuid4().hex[:12]

# A conversation's durable fields as one flat map, the shape `save` and
# `change_canvas` diff against the last stored snapshot. The scalar keys live in
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
_KEY_TURN_APPROVED = "turn_approved"
_KEY_TURN_USAGE = "turn_usage"
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
        _KEY_TURN_APPROVED,
        _KEY_TURN_USAGE,
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

# Identity, the process-local runtime objects, and the turn's spec blocks
# (`turn_specs`, held by the open turn's `turn_specs` rows,
# `harness.turn.store.TurnStore.specs`), which `_serialize_durable` and
# `_apply_delta` skip.
_PROCESS_LOCAL_FIELDS = frozenset(
    {
        "conversation_id",
        "lock",
        "renew_now",
        "_removal_unreported",
        "turn_specs",
    }
)

_NO_BASELINE_VALUE = object()

# The time every lease statement reads, in epoch seconds: the database's
# clock, so instances whose clocks disagree still agree on when a lease
# lapses. `now()` is the transaction's start time.
DATABASE_NOW = "CAST(extract(epoch FROM now()) AS double precision)"

# Frame numbers a segment's claim reserves (`StudioStore.claim_segment`),
# more than any segment sends, so a stream numbers its frames with no write
# of its own and a takeover starts past every number the stream it takes
# over could have sent. A segment that died unsaved leaves a gap of up to
# this many in `seq`, which no client reads.
SEQ_BLOCK = 2**20

# The backoff between retries (`retry_delays`), doubling from the first
# delay to the last.
_RETRY_FIRST_DELAY_S = 0.25
_RETRY_MAX_DELAY_S = 4.0
# How long before its lease lapses an instance stops trusting it holds the
# conversation. The heartbeat gives up renewing and writes stop retrying then
# (`StudioStore.lease_deadline`).
LEASE_MARGIN_S = 10
# The longest a write that carries a turn line reads the status of a
# transaction whose COMMIT raised (`_committing_lines`).
COMMIT_READ_BACK_LIMIT_S = 5.0


# The field values `_shallow_copy` copies, one level deep.
_CONTAINERS = (dict, list, set)


def _assign_changed(
    live: ConversationState, work: ConversationState, base: ConversationState
) -> None:
    """Assign to `live` each field of `work`, a copy taken from it
    (`_shallow_copy`), that changed since `base`, the snapshot taken with
    the copy. A container counts as changed when it compares unequal, any
    other value when it is not the same object. A field `live` changed
    meanwhile and `work` did not keeps `live`'s value. A `TurnUsage`, which
    `_shallow_copy` copies, counts as changed when it compares unequal too."""
    for f in dataclasses.fields(ConversationState):
        value = getattr(work, f.name)
        before = getattr(base, f.name)
        if value != before if isinstance(value, (*_CONTAINERS, TurnUsage)) else value is not before:
            setattr(live, f.name, value)


def _shallow_copy(state: ConversationState) -> ConversationState:
    """A copy of `state` whose dict, list and set fields are copies one
    level deep, whose `turn_usage` is a copy of its own (`TurnUsage.copy`),
    and whose other fields hold the same objects. A change that replaces the
    values it changes, as every canvas change does, and a spec use noted on
    the copy (`ConversationState.note_spec_use`) leave `state` itself
    untouched."""
    return dataclasses.replace(
        state,
        **{
            f.name: copy.copy(value)
            for f in dataclasses.fields(state)
            if isinstance(value := getattr(state, f.name), _CONTAINERS)
        },
        turn_usage=state.turn_usage.copy(),
    )


def _refold_turn_cards(state: ConversationState, baseline: dict[str, Any]) -> None:
    """Fold `state.turn_components` into the turn's record again, as
    `ConversationState.close_turn` folds them, when this instance recorded
    the turn since its last stored write (`baseline`). A card write whose
    acknowledgement was lost, settled only by the closing write
    (`_resolve_canvas_writes`), adds its turn line after a failure close
    built the record without it."""
    if not state.turn_recorded or baseline.get(_KEY_TURN_RECORDED) or not state.turns:
        return
    record = {key: value for key, value in state.turns[-1].items() if key != "cards"}
    if state.turn_components:
        record["cards"] = list(state.turn_components)
    state.turns[-1] = record


def _changed_since(full: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The subset of `full` that `baseline` either lacks or holds a
    different value for. `save` and `change_canvas` both diff a freshly
    serialised state against a baseline this way."""
    return {
        key: value for key, value in full.items() if baseline.get(key, _NO_BASELINE_VALUE) != value
    }


def _json_safe(value: Any) -> Any:
    """`value`, proven JSON-serialisable by round-tripping it through
    `to_json_param`/`json.loads`. Raises on a value the write would refuse."""
    return json.loads(to_json_param(value))


def _serialize_segment(segment: Segment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "owner": segment.owner,
        "expires_at": segment.expires_at,
    }


def _deserialize_segment(value: Any) -> Segment | None:
    if not value:
        return None
    return Segment(
        id=value["id"],
        owner=value["owner"],
        expires_at=value["expires_at"],
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
            # Written only when set: an entry with no admitted run has no key.
            **({"admitted_run": p.admitted_run} if p.admitted_run is not None else {}),
            # Written only when set, as the wire sends each frame.
            **(
                {"frames": [frame.model_dump(mode="json", exclude_none=True) for frame in p.frames]}
                if p.frames
                else {}
            ),
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
        admitted_run=entry.get("admitted_run"),
        frames=tuple(
            STUDIO_FRAME_ADAPTER.validate_python(frame) for frame in entry.get("frames") or ()
        ),
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


# Durable fields as `_serialize_durable` serialises them, by key.
type Snapshot = dict[str, Any]


def _serialize_durable(state: ConversationState) -> Snapshot:
    """Every durable field of `state` as the flat map the tables store,
    proven JSON-safe. Both a save's candidate delta (before diffing against
    a baseline) and the snapshot later saves diff against."""
    data: dict[str, Any] = {
        _KEY_TURN_INDEX: state.turn_index,
        _KEY_SEQ: state.seq,
        _KEY_TRACE_ID: state.trace_id,
        _KEY_PENDING: _serialize_pending(state.pending),
        # A JSON object's keys are always strings. `plan_version_ids`' int
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
        _KEY_TURN_APPROVED: state.turn_approved,
        _KEY_TURN_USAGE: dataclasses.asdict(state.turn_usage),
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
        elif key == _KEY_TURN_APPROVED:
            state.turn_approved = list(value or ())
        elif key == _KEY_TURN_USAGE:
            pass  # after the loop, once `trace_id` holds its value from this delta
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
    if _KEY_TURN_USAGE in delta:
        state.turn_usage = _turn_usage(delta[_KEY_TURN_USAGE], trace_id=state.trace_id)
    elif state.turn_usage.trace_id != (state.trace_id or None):
        # A row written before turns kept a usage, or a delta that moved the
        # trace id alone, leaves the turn an empty usage of its own, which
        # `take_identity` and `note_spec_use` can fill and a save keeps.
        state.turn_usage = TurnUsage(trace_id=state.trace_id or None)


def _turn_usage(value: Any, *, trace_id: str | None) -> TurnUsage:
    """The `TurnUsage` of the turn `trace_id` a stored `turn_usage` holds. A
    value that is not an object, or names another turn's trace id, reads as
    unset, as a member of the wrong type does. A conversation that has
    begun no turn has the empty trace id, and its usage names none."""
    trace_id = trace_id or None
    stored = value if isinstance(value, dict) else {}
    if stored.get("trace_id") != trace_id:
        stored = {}

    def text(key: str) -> str | None:
        item = stored.get(key)
        return item if isinstance(item, str) else None

    def specs(key: str) -> list[str]:
        items = stored.get(key)
        return [item for item in items if isinstance(item, str)] if isinstance(items, list) else []

    return TurnUsage(
        trace_id=trace_id,
        started_at=text("started_at"),
        caller=text("caller"),
        account=text("account"),
        visitor=text("visitor"),
        surface=text("surface"),
        fetched_specs=specs("fetched_specs"),
        drawn_specs=specs("drawn_specs"),
    )


# ---------------------------------------------------------------------------
# Turn lines
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TurnLine:
    """One `TURN_LOG_EVENT` line and its outcome observation: a turn's close
    (`status` `closed`, a `turn` observation) or a pause of it for the user
    (`paused`, a `turn.pause` observation)."""

    status: Literal["closed", "paused"]
    trace_id: str
    usage: TurnUsage
    request: str
    response: str
    error: str | None = None


def _closed_line(record: Mapping[str, Any]) -> _TurnLine:
    """The line of the turn whose record is `record`
    (`ConversationState.close_turn`): its reply is the answer with the code's
    match notes and basis line after it, each part present
    (`harness.studio_state._recorded_reply`)."""
    stored = record.get("usage")
    stored = stored if isinstance(stored, dict) else {}
    return _TurnLine(
        status="closed",
        trace_id=str(record.get("trace_id")),
        usage=_turn_usage(stored, trace_id=stored.get("trace_id")),
        request=str(record.get("user") or ""),
        response=_recorded_reply(record),
        error=record.get("error"),
    )


def write_pause_line(conversation_id: str, state: ConversationState) -> None:
    """Write the line and `turn.pause` outcome of `state`'s open turn, paused
    for the user, with the text it streamed so far as its reply. The turn's
    pause write calls it once that write commits
    (`harness.turn.store.TurnStore.pause`), so a turn writes one paused line
    for each pause its cursor stores. Never raises."""
    _write_turn_lines(
        conversation_id,
        [
            _TurnLine(
                status="paused",
                trace_id=state.trace_id,
                usage=state.turn_usage.copy(),
                request=state.turn_user_message,
                response=state.turn_final_text,
            )
        ],
    )


def _committed_lines(delta: Mapping[str, Any], *, stored: Mapping[str, Any]) -> list[_TurnLine]:
    """The close lines a write of `delta` over `stored`, the flat keys the
    row held before it (this instance's baseline), commits: one for each
    turn record it adds. A turn record the row already held is no close, so
    a turn writes one closed line. A pause is no Studio write's: its line
    comes from the cursor's pause write (`write_pause_line`)."""
    return [
        _closed_line(delta[key])
        for key in sorted(
            (key for key in delta if key.startswith(_TURN_ITEM_PREFIX) and key not in stored),
            key=lambda key: int(key.removeprefix(_TURN_ITEM_PREFIX)),
        )
    ]


def _write_turn_lines(conversation_id: str, lines: Sequence[_TurnLine]) -> bool:
    """Write each of `lines`: its `TURN_LOG_EVENT` log line, with the turn's
    identity, status, when it began, the deployed revision, its error code,
    and the specs it fetched and drew, as JSON fields
    (`common.logging.JSON_FIELDS`) and in the message, then its outcome
    observation, a `turn` for a close and a `turn.pause` for a pause
    (`common.telemetry.record_turn_outcome`). Then start their export
    without waiting for it (`start_telemetry_flush`), so a write outside any
    stream, whose last flush would carry them, still sends them. Returns
    whether there were lines. Never raises: a line that fails is logged and
    the rest are written."""
    for line in lines:
        try:
            _write_turn_line(conversation_id, line)
        except Exception:
            logger.exception(
                "studio store: the %s line of turn %s could not be written",
                line.status,
                line.trace_id,
            )
    if lines:
        start_telemetry_flush()
    return bool(lines)


def _write_turn_line(conversation_id: str, line: _TurnLine) -> None:
    usage = line.usage
    release = deploy_release()
    logger.info(
        "%s trace_id=%s status=%s caller=%s account=%s visitor=%s release=%s error=%s "
        "fetched_specs=%d drawn_specs=%d",
        TURN_LOG_EVENT,
        line.trace_id,
        line.status,
        usage.caller,
        usage.account,
        usage.visitor,
        release,
        line.error,
        len(usage.fetched_specs),
        len(usage.drawn_specs),
        extra={
            JSON_FIELDS: {
                "event": TURN_LOG_EVENT,
                "trace_id": line.trace_id,
                "status": line.status,
                "started_at": usage.started_at,
                "caller": usage.caller,
                "account": usage.account,
                "visitor": usage.visitor,
                "surface": usage.surface,
                "release": release,
                "error": line.error,
                "fetched_specs": list(usage.fetched_specs),
                "drawn_specs": list(usage.drawn_specs),
            }
        },
    )
    record_turn_outcome(
        line.trace_id,
        session_id=conversation_id,
        user_id=usage.user_id,
        tags=usage.tags,
        request=line.request,
        response=line.response,
        error=line.error,
        paused=line.status == "paused",
    )


def retry_delays() -> Iterator[float]:
    """The backoff between attempts at a write that met a transient fault or
    lost to another writer. It starts at `_RETRY_FIRST_DELAY_S`, doubles up
    to `_RETRY_MAX_DELAY_S`, and has no end. A write fenced on the live
    segment waits these (`StudioStore._retrying`), as does `/cancel` between
    claims it loses (`harness.studio`). Each caller bounds the total by its
    own deadline."""
    delay = _RETRY_FIRST_DELAY_S
    while True:
        yield delay
        delay = min(delay * 2, _RETRY_MAX_DELAY_S)


class ConversationStale(Exception):  # noqa: N818 -- named for the wire 409
    """`StudioStore.save` found the conversation's turn state moved since this
    instance loaded it: another instance claimed or saved a turn in between,
    or holds a live lease this save would claim over. The `/turn` claim and
    `/interrupt` answer 409."""


class StopPending(ConversationStale):
    """`StudioStore.save` refused a claim that starts or resumes work while a
    Stop is live for the conversation's open turn (`StopState.live`). The
    turn is closed `cancelled` under a claim that acts on the Stop first
    (`save(stopping=True)`)."""


class TurnMoved(Exception):  # noqa: N818 -- named for the wire 409
    """`StudioStore.claim_turn_lease` found the conversation's latest turn
    is not the one the request names as the turn it follows (`after_turn`):
    another request began a turn since. `latest` is the row's turn index."""

    def __init__(self, conversation_id: str, *, latest: int) -> None:
        super().__init__(f"conversation {conversation_id}'s latest turn is {latest}")
        self.latest = latest


class ConversationExists(Exception):  # noqa: N818 -- named for the race it reports
    """`StudioStore.create` lost the race to create the conversation."""


# ---------------------------------------------------------------------------
# Rows and columns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Renewal:
    """What a lease renewal found: whether the segment still `held` the
    lease, and the id of the Stop live for the lease's turn
    (`StopState.live`), else `None`."""

    held: bool
    stop_id: str | None


@dataclass(frozen=True)
class StoredConversation:
    """A conversation as its row stands, read with no lock and no lease
    (`StudioStore.peek`). `state` holds the durable fields, built apart from
    this instance's live state. `turn_status` is the status of the cursor of
    the row's turn, `None` when that turn has none (an Edit Card turn, or a
    turn claimed and not yet opened). `pause_id` is the wire interrupt id of
    the pause a paused cursor holds, `None` otherwise. `lease_seconds` is how
    long the lease on the row stays live by the database clock, `None` when
    none is."""

    state: ConversationState
    turn_status: TurnStatus | None
    lease_seconds: float | None
    pause_id: str | None = None


def record_status(record: Mapping[str, Any]) -> TurnStatus:
    """The status a closed turn's record states: `done` with no error,
    `cancelled` or `abandoned` for those errors, and `failed` for any other
    error code. The record is what the client was told the turn ended on,
    so it outranks the turn's cursor wherever the two differ."""
    match record.get("error"):
        case None:
            return TurnStatus.DONE
        case TurnStatus.CANCELLED.value:
            return TurnStatus.CANCELLED
        case TurnStatus.ABANDONED.value:
            return TurnStatus.ABANDONED
        case _:
            return TurnStatus.FAILED


def recorded_close_error(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The error a cursor closes on when its turn's record closed first
    (`record_status`): the record's code alone for `failed` or `abandoned`,
    and `None` for the statuses that take no error. The record holds what
    the client was told, so the cursor keeps none of the failure's own
    payload, which may be what JSON could not encode."""
    error = record.get("error")
    return {"code": error} if record_status(record) in _ERRORED_STATUSES else None


_ERRORED_STATUSES = frozenset({TurnStatus.FAILED, TurnStatus.ABANDONED})


class Standing(StrEnum):
    """Where a conversation's latest turn stands, as `turn_standing` derives
    it. The state read and `/resume` (`harness.studio`) each map a standing
    to what they answer, so they agree on every turn the store can hold.

    - `CLOSED`: the turn ended, on the status `TurnStanding.closed` names.
    - `HELD`: a live lease holds the turn, on some instance.
    - `RESUMABLE`: the turn is running and no segment holds it.
    - `AWAITING_ANSWER`: the turn is paused on a question whose stored entry
      holds the frames the pause was sent as and awaits the user's answer on
      `/interrupt`.
    - `CARRY`: the turn is paused on a question whose entry holds an answer
      a client left before the turn took it; `/resume` carries it on as a
      retry of that answer.
    - `REPLAY`: the turn is paused on a question no entry holds, since the
      write that stores the entry never landed, or whose entry holds no
      frames to show, since it was stored before entries kept them;
      `/resume` sends the pause again from the cursor.
    """

    CLOSED = "closed"
    HELD = "held"
    RESUMABLE = "resumable"
    AWAITING_ANSWER = "awaiting_answer"
    CARRY = "carry"
    REPLAY = "replay"


@dataclass(frozen=True)
class TurnStanding:
    """A turn's `standing`, with the status it `closed` on when `CLOSED`."""

    standing: Standing
    closed: TurnStatus | None = None


def pause_entry(
    pending: Mapping[str, _PendingInterrupt], pause_id: str | None
) -> tuple[str, _PendingInterrupt] | None:
    """The conversation's entry for the pause `pause_id` (a paused cursor's
    wire interrupt id), with its key, or `None` when no entry holds it.
    Entries for other pauses, such as an earlier one whose answer the turn
    took, play no part."""
    if pause_id is None:
        return None
    for interrupt_id, entry in pending.items():
        if entry.adk_id == pause_id:
            return interrupt_id, entry
    return None


def turn_standing(
    *,
    cursor_status: TurnStatus | None,
    pause_id: str | None,
    lease_live: bool,
    state: ConversationState,
) -> TurnStanding:
    """Where `state`'s latest turn stands, from its cursor's status
    (`None` when the turn has none: an Edit Card turn, or one claimed and
    not yet opened), the wire interrupt id of the pause a paused cursor
    holds, and whether a lease on the conversation is live.

    A live lease holds the turn whatever the cursor and the record say, so
    a reader waits for its holder. Otherwise a recorded turn is closed as
    its record states (`record_status`), since the record is what the
    client was told, whatever its cursor says: a write that commits a
    record closes the cursor with it, and `StudioStore.load` closes a
    cursor it finds open under a record. An unrecorded turn with no cursor
    was left by a segment that never opened or recorded it, and reads
    `abandoned`, as the next turn records it and as `StudioStore.peek`
    records it on its copy. A cursor closed under an open record is
    recorded as it closed (`record_closed_turn`) by `load` and on `peek`'s
    copy, so it reads as the cursor closed. A running cursor is resumable, and a paused one
    stands as the pause's entry does (`pause_entry`)."""
    if lease_live:
        return TurnStanding(Standing.HELD)
    if state.turn_recorded:
        return TurnStanding(Standing.CLOSED, record_status(state.turns[-1]))
    match cursor_status:
        case None:
            return TurnStanding(Standing.CLOSED, TurnStatus.ABANDONED)
        case TurnStatus.RUNNING:
            return TurnStanding(Standing.RESUMABLE)
        case TurnStatus.PAUSED:
            return TurnStanding(_paused_standing(state.pending, pause_id))
        case _:
            return TurnStanding(Standing.CLOSED, cursor_status)


def _paused_standing(pending: Mapping[str, _PendingInterrupt], pause_id: str | None) -> Standing:
    """Where a turn paused on `pause_id` stands, by the pause's entry."""
    found = pause_entry(pending, pause_id)
    if found is None:
        return Standing.REPLAY
    _, entry = found
    if entry.resuming_segment is not None and entry.answer is not None:
        return Standing.CARRY
    if not entry.frames:
        # An entry stored before entries kept their frames gives the page
        # no card to draw, so the pause is sent again from the cursor, whose
        # translation stores the frames.
        return Standing.REPLAY
    return Standing.AWAITING_ANSWER


@dataclass(frozen=True)
class TurnLease:
    """A lease `StudioStore.claim_turn_lease` claimed: the segment's id and
    the `conversations` row's id."""

    segment_id: str
    conversation: int


class SegmentClose(StrEnum):
    """How a segment's closing write (`StudioStore.close_segment`) ended.

    - `SAVED`: the live state is stored and the lease cleared. The turn
      stands as the live state holds it, unless the bookkeeping after the
      commit failed: the store then dropped its live state, and the turn
      stands as the row holds it, the canvas the write committed included.
    - `STOPPED`: as `SAVED`, and a Stop live for the open turn closed it
      `cancelled`, record and cursor, in the same transaction.
    - `LOST`: the stored lease is not the segment's, so nothing was
      written. The turn stands as the lease's holder leaves it.
    - `FAILED`: the write did not commit as far as this instance knows (a
      database error, or the timeout). It may have committed with its
      acknowledgement lost. The lease lapses by its TTL.
    """

    SAVED = "saved"
    STOPPED = "stopped"
    LOST = "lost"
    FAILED = "failed"


def closed_turn_error(cursor: Cursor) -> str:
    """The error a turn record closes on when its cursor closed and the
    stream's closing write failed (`StudioStore.load`). It is what the stream
    told the client (`harness.studio_stream.ending`'s `failed` column). A
    Stop's close is `cancelled`. A turn closed `failed` or `abandoned` on a node's
    failure ended on the error frame its error gives, and records that
    frame's code (`harness.fatal_codes.recorded_code`). Any other was told
    its state could not be saved, `state_not_saved`."""
    match cursor.status:
        case TurnStatus.CANCELLED:
            return ErrorCode.CANCELLED.value
        case TurnStatus.FAILED | TurnStatus.ABANDONED:
            return recorded_code(cursor.error or {}).value
        case _:
            return HarnessCode.STATE_NOT_SAVED.value


_OPEN_STATUSES = frozenset({"running", "paused"})


@dataclass(frozen=True)
class StopState:
    """A conversation row's Stop columns and the turn its index names, read
    after the row lock. `turn_index` is the row's turn index; `flagged`
    whether `lease_cancel` is set, a Stop no lease write has stamped yet;
    `stop_id` and `stop_turn` the latest Stop and the turn index it names;
    `cursor_status` the status of the turn's cursor, `None` when the turn
    has none yet; `taken_id` the Stop that cursor's `StopScope` answered;
    and `lease_live` whether a lease on the row is live after the read."""

    turn_index: int
    flagged: bool
    stop_id: str | None
    stop_turn: int | None
    cursor_status: str | None
    taken_id: str | None
    lease_live: bool

    @property
    def live(self) -> bool:
        """Whether a Stop is live for turn `turn_index`: the turn is open, or
        claimed and awaiting its cursor (a live lease and no cursor yet), and
        either `lease_cancel` is set or the latest Stop names the turn and is
        not the one its scope answered."""
        if self.cursor_status is None:
            if not self.lease_live:
                return False
        elif self.cursor_status not in _OPEN_STATUSES:
            return False
        return self.flagged or (
            self.stop_turn == self.turn_index
            and self.stop_id is not None
            and self.stop_id != self.taken_id
        )


# The row's turn index, as stored in its turn state.
_TURN_INDEX_SQL = f"COALESCE((turn_state ->> '{_KEY_TURN_INDEX}')::int, 0)"
# Turns a set `lease_cancel` into a fresh Stop id naming the row's turn index
# as stored before the write, and clears the flag, in the statement that
# writes the lease columns.
_STAMP_STOP = (
    "stop_id = CASE WHEN lease_cancel THEN :minted ELSE stop_id END, "
    f"stop_turn = CASE WHEN lease_cancel THEN {_TURN_INDEX_SQL} ELSE stop_turn END, "
    "lease_cancel = false"
)
# The status of a conversation's turn's cursor, and its pause's interrupt id.
_TURN_PAUSE = text(
    "SELECT status, interrupt_id FROM turn_cursors "
    "WHERE conversation = :pk AND turn_index = :turn_index"
)
_TURN_AT = text(
    "SELECT status, stop_taken_id FROM turn_cursors "
    "WHERE conversation = :pk AND turn_index = :turn_index"
)


async def _stop_state(conn: AsyncConnection, row: Any, *, lease_live: bool) -> StopState:
    """The Stop state of `row`, a `conversations` row the caller locked,
    with the cursor of its turn read in this later statement so that it
    sees every cursor write committed before the lock."""
    cursor = (await conn.execute(_TURN_AT, {"pk": row.id, "turn_index": row.turn_index})).first()
    return StopState(
        turn_index=row.turn_index,
        flagged=row.lease_cancel,
        stop_id=row.stop_id,
        stop_turn=row.stop_turn,
        cursor_status=cursor.status if cursor is not None else None,
        taken_id=cursor.stop_taken_id if cursor is not None else None,
        lease_live=lease_live,
    )


def _jsonb(value: Any) -> str | None:
    """`value` as a jsonb parameter: JSON text, or SQL NULL for `None`."""
    return None if value is None else to_json_param(value)


_CONVERSATION_COLUMNS = (
    "id, version, turn_state, composition, lease_segment_id, lease_owner, "
    "lease_expires_at, stop_id, stop_turn, lease_cancel, seq_next, seq_reserved, canvas_writes, "
    f"{_TURN_INDEX_SQL} AS turn_index"
)


def _row_segment(row: Any) -> Segment | None:
    if row.lease_segment_id is None:
        return None
    return Segment(
        id=row.lease_segment_id,
        owner=row.lease_owner,
        expires_at=row.lease_expires_at,
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


def _delta_write(delta: dict[str, Any], *, lease_clock: str | None = None) -> _DeltaWrite:
    """The statements that write `delta`'s flat keys. A write of the lease
    columns stamps a pending Stop (`_STAMP_STOP`). Raises on a value the
    columns cannot hold (a `seq` that is not an integer, a segment without
    an id or whose id, owner or expiry is of another type, a position that
    is not a number, a write token that is not a string, a value
    `json.dumps` refuses) and on a key no column or row holds, so a caller
    that builds the statements without running them learns what running
    them would refuse. Given `lease_clock`, an SQL expression for the
    database clock, a segment's expiry is written as `SEGMENT_TTL_S` past
    that clock's reading at the write, in place of the segment's own."""
    sets = ["version = version + 1", "updated_at = now()"]
    params: dict[str, Any] = {}
    scalars = {key: value for key, value in delta.items() if key in _SCALAR_KEYS}
    if scalars:
        sets.append("turn_state = turn_state || CAST(:scalars AS jsonb)")
        params["scalars"] = to_json_param(scalars)
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
        clocked = segment is not None and lease_clock is not None
        expires = f"{lease_clock} + :ttl" if clocked else ":expires_at"
        sets.append(
            "lease_segment_id = :segment_id, lease_owner = :owner, "
            f"lease_expires_at = {expires}, {_STAMP_STOP}"
        )
        params |= {
            "minted": _wire_id("stop"),
            "segment_id": segment.id if segment else None,
            "owner": segment.owner if segment else None,
        }
        if clocked:
            params["ttl"] = SEGMENT_TTL_S
        else:
            params["expires_at"] = segment.expires_at if segment else None
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
                "value": to_json_param(value) if items.json_value else value,
            }
        )
    return _DeltaWrite(sets=tuple(sets), params=params, rows=rows)


class _Written(NamedTuple):
    """The row's turn-state version and lease expiry after a delta write."""

    version: int
    lease_expires_at: float | None


@dataclass(eq=False)
class _CanvasWrite:
    """One `change_canvas` call's write id and what its latest attempt to
    reach the write computed (`_fenced_save`): the working copy and the live
    snapshot taken with it, the changed durable keys and `change`'s result.
    The id is minted once per call and kept across its retries. An attempt
    that raised after it queued its write leaves the write pending on the
    runtime entry until the next fenced write resolves it from the row's
    `canvas_writes`: `landed` once the row holds the id, so the call's retry
    returns `result` and runs `change` no second time."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    work: ConversationState | None = None
    base: ConversationState | None = None
    changed: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    landed: bool = False


async def _write_delta(
    conn: AsyncConnection, pk: int, delta: dict[str, Any], *, lease_clock: str | None = None
) -> _Written:
    """Write `delta`'s flat keys to their columns and rows (`_delta_write`),
    bump the turn state's version, and return the new version and the
    stored lease expiry. The caller holds the row lock and decided the
    write is allowed."""
    return await _run_delta_write(conn, pk, _delta_write(delta, lease_clock=lease_clock))


async def _run_delta_write(conn: AsyncConnection, pk: int, write: _DeltaWrite) -> _Written:
    """Run `write`'s statements on row `pk` and return the row's new
    version and lease expiry (`_write_delta`)."""
    written = (
        await conn.execute(
            text(
                f"UPDATE conversations SET {', '.join(write.sets)} WHERE id = :pk "
                "RETURNING version, lease_expires_at"
            ),
            {**write.params, "pk": pk},
        )
    ).one()
    for prefix, rows in write.rows.items():
        await conn.execute(_ITEM_TABLES[prefix].upsert(), [{**row, "pk": pk} for row in rows])
    return _Written(written.version, written.lease_expires_at)


_INSERT_CARD_LINEAGE = text(
    "INSERT INTO conversation_card_lineage (conversation, component_id, spec_id, "
    "content_version, requested_spec_id, requested_content_version, members, stat_var_count) "
    "VALUES (:pk, :component_id, :spec_id, :content_version, :requested_spec_id, "
    ":requested_content_version, CAST(:members AS json), :stat_var_count) "
    "ON CONFLICT (conversation, component_id) DO NOTHING"
)

# A card's stored lineage in conversation `:pk`: the row of card
# `:component_id`, else one drawn from the frame `:spec_id`/`:content_version`,
# else one whose write named that frame, the latest written first among equals,
# leaving out the cards in `:skipped`, whose rows did not decode.
# Cards on one frame share one lineage, so any of them serves.
_SELECT_CARD_LINEAGE = text(
    "SELECT component_id, spec_id, content_version, requested_spec_id, "
    "requested_content_version, members, stat_var_count "
    "FROM conversation_card_lineage WHERE conversation = :pk AND ("
    "component_id = :component_id "
    "OR (spec_id = :spec_id AND content_version = :content_version) "
    "OR (requested_spec_id = :spec_id AND requested_content_version = :content_version)) "
    "AND component_id <> ALL(:skipped) "
    "ORDER BY COALESCE(component_id = :component_id, false) DESC, "
    "COALESCE(spec_id = :spec_id AND content_version = :content_version, false) DESC, "
    "created_at DESC, component_id "
    "LIMIT 1"
).bindparams(bindparam("skipped", type_=ARRAY(Text)))


def _card_lineage_row(lineage: CardLineage) -> dict[str, Any] | None:
    """`lineage` as the parameters of its row, its members as JSON text, or
    `None`, logged at error, when JSON cannot hold them (a NaN or an
    infinity among a manifest's values, which `json` would write as a bare
    token Postgres refuses). The card then lands without its row."""
    component_id = lineage.component_id
    members = json_or_none(
        lineage.members_json(),
        logger=logger,
        failure=lambda: (
            f"studio store: the lineage of component_id={component_id!r} cannot be "
            "stored as JSON; the card lands without it"
        ),
    )
    if members is None:
        return None
    requested = lineage.requested or (None, None)
    return {
        "component_id": lineage.component_id,
        "spec_id": lineage.frame[0],
        "content_version": lineage.frame[1],
        "requested_spec_id": requested[0],
        "requested_content_version": requested[1],
        "members": members,
        "stat_var_count": lineage.stat_var_count,
    }


async def _insert_card_lineage(conn: AsyncConnection, pk: int, row: Mapping[str, Any]) -> None:
    """Store a card's lineage `row` (`_card_lineage_row`) on conversation `pk`,
    in a savepoint, so a failed insert leaves the card's write to commit. A
    card's lineage is written once: its frame never changes, and a write
    retried after a commit whose acknowledgement was lost finds the row there.
    A transient fault raises, for the whole write to be retried; any other
    is logged at error and the card lands without its row
    (`harness.db.best_effort.execute_in_savepoint`)."""
    component_id = row["component_id"]
    await execute_in_savepoint(
        conn,
        _INSERT_CARD_LINEAGE,
        {**row, "pk": pk},
        logger=logger,
        failure=lambda: (
            f"studio store: the lineage of component_id={component_id!r} was not "
            "stored; the card lands without it"
        ),
    )


def keep_turn_records(conversation: ConversationState, state: Mapping[str, Any]) -> None:
    """Keep the plan records, search readings, spec blocks and applied chip
    `state`, a turn's state, holds, for `ConversationState.close_turn` to
    store with the turn. Each key holds the turn's whole value, so the latest
    write replaces. The blocks stay in process (`turn_specs`) until the turn
    closes."""
    if ORCHESTRATOR_PLAN_RECORDS_KEY in state:
        conversation.turn_plans = list(state[ORCHESTRATOR_PLAN_RECORDS_KEY] or [])
    if ORCHESTRATOR_SEARCHES_KEY in state:
        conversation.turn_readings = list(state[ORCHESTRATOR_SEARCHES_KEY] or [])
    if ORCHESTRATOR_SPECS_KEY in state:
        conversation.turn_specs = dict(state[ORCHESTRATOR_SPECS_KEY] or {})
    if ORCHESTRATOR_APPLIED_KEY in state:
        conversation.turn_applied_choice = dict(state[ORCHESTRATOR_APPLIED_KEY] or {})


def set_aside_question(cursor: Cursor) -> str | None:
    """The user's words turn `cursor` ended on when it closed `done` with
    its plan card set aside for them, a different question
    (`harness.graph.state.NEW_QUESTION_KEY`), else `None`."""
    typed = cursor.state.get(NEW_QUESTION_KEY)
    if cursor.status is TurnStatus.DONE and isinstance(typed, str) and typed:
        return typed
    return None


def asked_for_question(cursor: Cursor) -> bool:
    """Whether turn `cursor` closed `done` with its plan card set aside for a
    new question no message of the user's states, asking them to send it
    (`harness.graph.state.ASK_FOR_QUESTION_KEY`)."""
    return cursor.status is TurnStatus.DONE and cursor.state.get(ASK_FOR_QUESTION_KEY) is True


def record_closed_turn(conversation: ConversationState, cursor: Cursor) -> str:
    """Record turn `cursor`, closed, on `conversation` as the client was told
    it ended, for a turn whose closing write never stored its record: keep
    its records (`keep_turn_records`) and close it with no reply and
    `closed_turn_error`'s error. A turn that ended on a question set aside
    for the next turn (`set_aside_question`) keeps that question too, so the
    state read still hands it back to the page that never got `done`.
    Returns the error recorded."""
    keep_turn_records(conversation, cursor.state)
    error = closed_turn_error(cursor)
    conversation.close_turn(
        final_text="",
        error=error,
        new_question=set_aside_question(cursor),
        asked_for_question=asked_for_question(cursor),
    )
    return error


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
    lease's expiry: the start of the save that wrote the claim, then each
    renewal's start, the anchor of `StudioStore.lease_deadline`. `held` is the segment
    this instance claimed and has not yet released, the one lease its own
    saves may write under. `seq_reserved` is the stored end of the frame
    numbers reserved for the conversation's segments (`claim_segment`),
    and `claimed_reserved` the end a claim not yet saved reserves. `state`
    is the `ConversationState` this instance keeps live between requests;
    `load` refreshes its durable fields in place, so `lock` and `renew_now`
    keep their identity across a reload.
    `draft` is the working copy a request's claim is built on (`draft`) and
    the live state as it stood when the copy was taken. `baseline` is the
    last durable snapshot known stored, the reference a save diffs against.
    `composition_lock` serialises same-instance `/composition` calls on the
    shared `state` object, separate from `lock` since a canvas edit must
    land while a turn streams. `write_lock` serialises this instance's own
    writes on the conversation, so two of them never interleave their
    `state`/`baseline` updates. `canvas_pending` holds the canvas writes
    whose attempt raised once it had queued its write, so their commit is
    unknown; the next fenced write (`_fenced_save`) or closing write
    (`close_segment`) resolves them.
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
    state: ConversationState | None = None
    baseline: dict[str, Any] = field(default_factory=dict)
    draft: tuple[ConversationState, ConversationState] | None = None
    claimed: asyncio.Event = field(default_factory=asyncio.Event)
    canvas_pending: list[_CanvasWrite] = field(default_factory=list)


class StudioStore:
    """Conversation state, keyed (principal, conversation_id), stored in the
    harness's tables over `engine` (module docstring).

    `claim_segment`/`renew`/`request_cancel` are the segment
    lease API: together they keep at most one instance streaming a
    conversation at a time.

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
        lineage_engine: AsyncEngine | None = None,
        clock: Callable[[], float] = time.time,
        time_sql: str = DATABASE_NOW,
    ) -> None:
        self._engine = engine
        # Card lineage reads run over their own pool
        # (`harness.db.engine.DatabaseHandle.lineage_engine`); a test store
        # passes none and reads over `engine`.
        self._lineage_engine = lineage_engine or engine
        self._runtime: dict[tuple[str, str], _RuntimeEntry] = {}
        self._clock = clock
        self._time_sql = time_sql
        # Each read of the row carries the database clock's reading with it.
        self._select_conversation = text(
            f"SELECT {_CONVERSATION_COLUMNS}, {time_sql} AS db_now FROM conversations "
            "WHERE principal = :principal AND conversation_id = :conversation_id"
        )
        self._lock_conversation = text(
            f"SELECT {_CONVERSATION_COLUMNS}, {time_sql} AS db_now, "
            "pg_current_xact_id()::text AS xact FROM conversations "
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

    def draft(self, principal: str, conversation_id: str) -> ConversationState:
        """A working copy of this instance's live state, for the changes a
        request makes before its claim save (a turn begun, an answer marked,
        a segment claimed). `save` given the copy swaps what it changed into
        the live state once the write commits; until then, and on any exit
        that saves nothing (`drop_draft`), the live state stays as loaded.
        The copy is one level deep (`_shallow_copy`), with each pending
        interrupt copied too, since an answer is marked on its entry."""
        entry = self._runtime[(principal, conversation_id)]
        assert entry.state is not None
        work = _shallow_copy(entry.state)
        work.pending = {key: copy.copy(pending) for key, pending in work.pending.items()}
        entry.draft = (work, _shallow_copy(entry.state))
        return work

    def drop_draft(self, principal: str, conversation_id: str) -> None:
        """Give up the working copy `draft` handed out, its claim unsaved."""
        entry = self._runtime.get((principal, conversation_id))
        if entry is None or entry.draft is None:
            return
        entry.draft = None
        entry.claimed_reserved = None

    def lease_deadline(self, principal: str, conversation_id: str) -> float:
        """When, on this instance's clock (`now`), it stops trusting that it
        still holds the lease it claimed on `conversation_id`. That is
        `LEASE_MARGIN_S` before the lease would lapse, counted from just
        before the read or renewal that set its expiry. The heartbeat ends
        the stream there when renewals keep failing, and a write that keeps
        failing stops retrying there (`_retrying`)."""
        return self._lease_deadline(self._entry(principal, conversation_id))

    def _lease_deadline(self, entry: _RuntimeEntry) -> float:
        return entry.lease_stamp + SEGMENT_TTL_S - LEASE_MARGIN_S

    def lock_for(self, principal: str, conversation_id: str) -> asyncio.Lock:
        """This (principal, conversation_id) pair's lock, creating its
        runtime entry on first use. Callers acquire it before `load` or
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

    def _transaction(self) -> AbstractAsyncContextManager[AsyncConnection]:
        """One read-committed transaction. Every write goes through here."""
        return self._engine.begin()

    async def _write(
        self,
        conn: AsyncConnection,
        pk: int,
        delta: dict[str, Any],
        *,
        seq_reserved: int | None = None,
        claim: bool = False,
    ) -> _Written:
        """Write `delta` to the turn state, the canvas, the lease and the item
        rows, and `seq_reserved` when given, and return the row's new version
        and lease expiry (`_write_delta`). With `claim`, the segment `delta`
        carries expires `SEGMENT_TTL_S` past the database clock's reading at
        this write, and the canvas write ids recorded under earlier leases
        (`_CanvasWrite`) are cleared."""
        written = await _write_delta(conn, pk, delta, lease_clock=self._time_sql if claim else None)
        if claim:
            await conn.execute(
                text("UPDATE conversations SET canvas_writes = '[]'::jsonb WHERE id = :pk"),
                {"pk": pk},
            )
        if seq_reserved is not None:
            await conn.execute(
                text("UPDATE conversations SET seq_reserved = :reserved WHERE id = :pk"),
                {"pk": pk, "reserved": seq_reserved},
            )
        return written

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
    ) -> tuple[Any, dict[str, Any], Cursor | None, tuple[TurnStatus, str | None] | None] | None:
        """The conversation's row, with the database clock's reading as
        `db_now`, its flat durable keys, the cursor of its turn when that
        cursor and the turn's record disagree on whether the turn is closed,
        and the status of the turn's cursor with the interrupt id of the
        pause it holds, `None` when the turn has no cursor, read in one
        snapshot. None when no row holds it."""
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
                if not row.turn_index:
                    return row, flat, None, None
                at = (
                    await conn.execute(_TURN_PAUSE, {"pk": row.id, "turn_index": row.turn_index})
                ).first()
                status = (TurnStatus(at.status), at.interrupt_id) if at is not None else None
                disagrees = status is not None and (status[0] in CLOSED) != bool(
                    flat.get(_KEY_TURN_RECORDED)
                )
                cursor = await cursor_at(conn, row.id, row.turn_index) if disagrees else None
        return row, flat, cursor, status

    async def load(self, principal: str, conversation_id: str) -> ConversationState | None:
        """Reload `conversation_id`'s durable fields. `None` when the store
        does not hold the conversation. An open turn's spec blocks are left
        empty: its cursor holds them (`harness.turn.store.TurnStore.specs`),
        and whoever carries the turn on reads them there.

        The turn's cursor is the authority on whether the turn is closed. A
        load that finds the cursor closed and the conversation's record of
        the turn open, with no live lease on the row, records the turn
        (`_record_closed_turn`) before it returns. A stream sends its ending
        frame only once its closing write (`close_segment`) commits, so that
        state means the closing write failed, and the client was told what
        `harness.studio_stream.ending` sends for a failed close, or the
        instance died and the client was told nothing. Without the repair
        the next turn would record the turn `abandoned`. While a lease is
        live its holder records the turn.

        The read and the copy into the live state run under this instance's
        write lock, so no write of this instance, a lease claim say, commits
        between them and is then overwritten by the row as read before it."""
        entry = self._entry(principal, conversation_id)
        async with entry.write_lock:
            return await self._load(entry, principal, conversation_id)

    async def peek(self, principal: str, conversation_id: str) -> StoredConversation | None:
        """`conversation_id` as its row stands, for a read that changes
        nothing: no lock, no lease, and this instance's live state left as
        it is, so a stream running here keeps its own. `None` when the store
        does not hold the conversation.

        A turn whose cursor closed with its record left open and no lease
        live is shown recorded as `load` would record it
        (`_record_closed_turn`), and an unrecorded turn with no cursor and no
        lease live, an Edit Card turn whose closing write failed or a turn
        claimed and never opened, is shown recorded `abandoned`, as the next
        turn records it (`ConversationState.begin_turn`). Both are on the
        copy alone: nothing is written."""
        stored = await self._read(principal, conversation_id)
        if stored is None:
            return None
        row, flat, disagreeing, at = stored
        status, pause_id = at if at is not None else (None, None)
        state = ConversationState(conversation_id=conversation_id)
        _apply_delta(state, flat)
        lease = _row_segment(row)
        live = lease is not None and lease.expires_at > row.db_now
        if not live and state.turn_index and not state.turn_recorded:
            if disagreeing is not None:
                record_closed_turn(state, disagreeing)
            elif status is None:
                state.close_turn(final_text=state.turn_final_text, error="abandoned")
        return StoredConversation(
            state=state,
            turn_status=status,
            lease_seconds=lease.expires_at - row.db_now if lease is not None and live else None,
            pause_id=pause_id if status is TurnStatus.PAUSED else None,
        )

    async def _load(
        self, entry: _RuntimeEntry, principal: str, conversation_id: str
    ) -> ConversationState | None:
        """`load`'s read and copy. The caller holds the write lock."""
        stamp = self._clock()
        stored = await self._read(principal, conversation_id)
        if stored is None:
            return None
        row, flat, disagreeing, _ = stored
        durable = ConversationState(conversation_id=conversation_id)
        _apply_delta(durable, flat)
        version = row.version
        lease = _row_segment(row)
        if disagreeing is not None and (lease is None or lease.expires_at <= row.db_now):
            if durable.turn_recorded:
                await self._close_recorded_cursor(row.id, row.version, durable, disagreeing)
            else:
                recorded = await self._record_closed_turn(row.id, row.version, durable, disagreeing)
                if recorded is not None:
                    durable, version = recorded
        entry.pk, entry.version, entry.read_at = row.id, version, row.db_now
        entry.read_stamp = stamp
        entry.seq_reserved, entry.claimed_reserved = row.seq_reserved, None
        held = entry.state.turn_usage if entry.state is not None else None
        if entry.state is None:
            durable.lock = entry.lock
            entry.state = durable
        else:
            for f in dataclasses.fields(ConversationState):
                if f.name not in _PROCESS_LOCAL_FIELDS:
                    setattr(entry.state, f.name, getattr(durable, f.name))
        entry.baseline = _serialize_durable(entry.state)
        if held is not None and held.trace_id == entry.state.trace_id:
            # Specs the turn noted after its last save, during a pause, say,
            # are this process's alone. A turn this instance holds keeps them,
            # and its next save stores them.
            for spec_id in held.fetched_specs:
                entry.state.note_spec_use("fetched", spec_id)
            for spec_id in held.drawn_specs:
                entry.state.note_spec_use("drawn", spec_id)
        segment = entry.state.segment
        if segment is None or segment.id != entry.held:
            entry.held = None
        if entry.state.turn_index and not entry.state.turn_recorded:
            entry.state.turn_specs = {}
        return entry.state

    async def _record_closed_turn(
        self, pk: int, version: int, durable: ConversationState, cursor: Cursor
    ) -> tuple[ConversationState, int] | None:
        """Record the turn `cursor`, closed, as the client was told it ended:
        on a copy of `durable`, the conversation as read at `version`, keep
        the records the cursor's state holds (`keep_turn_records`) and close
        the record with `closed_turn_error`'s error and no reply, then store
        the change. One transaction under the row lock, compare-and-swapped
        on `version` and refused while a lease is live, so a writer that
        moved the row since the read wins and this records nothing. The
        write that stores the record writes the turn's line and `turn`
        outcome once it commits (`_committed_lines`), a COMMIT that raised
        and whose transaction committed included (`_committing_lines`).
        Returns the copy and the row's new version, `None` when it recorded
        nothing."""
        recorded = _shallow_copy(durable)
        error = record_closed_turn(recorded, cursor)
        before = _serialize_durable(durable)
        delta = _changed_since(_serialize_durable(recorded), before)
        lines = _committed_lines(delta, stored=before)

        async def write(conn: AsyncConnection) -> tuple[str | None, Callable[[], int | None]]:
            row = (await conn.execute(self._lock_conversation, {"pk": pk})).first()
            lease = _row_segment(row) if row is not None else None
            if (
                row is None
                or row.version != version
                or (lease is not None and lease.expires_at > row.db_now)
            ):
                return None, lambda: None
            written = (await self._write(conn, pk, delta)).version

            def committed() -> int:
                _write_turn_lines(durable.conversation_id, lines)
                return written

            return row.xact, committed

        stored = await self._committing_lines(
            write, what=f"conversation {durable.conversation_id}'s record of a closed turn"
        )
        if stored is None:
            return None
        logger.info(
            "studio store: conversation %s's turn %s was closed %s with its record open; "
            "recorded it %s",
            durable.conversation_id,
            cursor.turn_index,
            cursor.status.value,
            error,
        )
        return recorded, stored

    async def _close_recorded_cursor(
        self, pk: int, version: int, durable: ConversationState, cursor: Cursor
    ) -> None:
        """Close the cursor of a turn whose record committed while the
        cursor stayed open, with the status the record states and the
        record's code as its error (`record_status`, `recorded_close_error`).
        Every write that commits a record closes the cursor in the same
        transaction; this closes any cursor still found open under a record.
        One transaction under the row lock, refused when the row moved past
        `version` or a lease on it is live, so a writer since the read wins. A cursor that moved
        since the read is left as it stands."""
        record = durable.turns[-1]
        status = record_status(record)

        async def write(conn: AsyncConnection) -> bool:
            row = (await conn.execute(self._lock_conversation, {"pk": pk})).first()
            lease = _row_segment(row) if row is not None else None
            if (
                row is None
                or row.version != version
                or (lease is not None and lease.expires_at > row.db_now)
            ):
                return False
            try:
                await close_unleased_on(
                    conn,
                    cursor,
                    status=status,
                    now=datetime.fromtimestamp(self._clock(), UTC),
                    error=recorded_close_error(record),
                )
            except CursorStale:
                return False
            return True

        try:
            async with self._transaction() as conn:
                closed = await write(conn)
        except Exception:
            logger.warning(
                "studio store: conversation %s's turn %s is recorded with its cursor open; "
                "closing the cursor failed",
                durable.conversation_id,
                cursor.turn_index,
                exc_info=True,
            )
            return
        if closed:
            logger.info(
                "studio store: conversation %s's turn %s was recorded with its cursor open; "
                "closed the cursor %s",
                durable.conversation_id,
                cursor.turn_index,
                status.value,
            )

    def lease_live(self, principal: str, conversation_id: str) -> bool:
        """Whether the lease on the conversation's loaded state was live at
        the database clock's reading taken with the load, as
        `claim_segment` judges it. `load` must have run first."""
        entry = self._runtime[(principal, conversation_id)]
        live = entry.state.segment if entry.state is not None else None
        return live is not None and live.expires_at > entry.read_at

    def claim_segment(
        self, principal: str, conversation_id: str, state: ConversationState | None = None
    ) -> str:
        """Claim a fresh segment on `state`, a working copy from `draft`, or
        else on the live state already loaded on this instance (`create` or
        `load` must have run first). Raises
        `SegmentBusy` when the loaded lease had not lapsed by the database
        clock's reading taken with the load. Otherwise writes a fresh segment
        into that state and returns its id. The expiry it holds,
        `SEGMENT_TTL_S` after that reading, stands until the claim save
        writes the stored one, `SEGMENT_TTL_S` after the database clock's
        reading at that write, so time spent between the load and the save
        does not come off the lease.

        The claim also reserves the segment's frame numbers, one block of
        `SEQ_BLOCK`, and starts the state's `seq` there. For a turn
        `begin_turn` just opened the block runs from `0` to `SEQ_BLOCK`.
        Otherwise it starts after every number reserved before, so each
        number a stream resuming the turn sends is new to the client.

        Does no I/O. The caller's own claim save (the `/turn` claim or the
        `/interrupt` save) persists it, and refuses it when the stored lease
        is live by then."""
        entry = self._runtime[(principal, conversation_id)]
        target = state if state is not None else entry.state
        assert target is not None
        live = target.segment
        if live is not None and live.expires_at > entry.read_at:
            raise SegmentBusy(retry_after_seconds=math.ceil(live.expires_at - entry.read_at))
        segment = Segment(
            id=_wire_id("seg"), owner=_OWNER_ID, expires_at=entry.read_at + SEGMENT_TTL_S
        )
        target.segment = segment
        entry.lease_stamp = entry.read_stamp
        resumes = target.turn_index == entry.baseline.get(_KEY_TURN_INDEX)
        target.seq = entry.seq_reserved if resumes else 0
        entry.claimed_reserved = target.seq + SEQ_BLOCK
        return segment.id

    async def claim_turn_lease(
        self,
        principal: str,
        conversation_id: str,
        *,
        segment_id: str | None = None,
        after_turn: int | None = None,
    ) -> TurnLease:
        """Claim the conversation's lease for a new turn and write the lease
        alone, under the row lock, emptying the canvas write ids recorded
        under earlier leases (`_CanvasWrite`). The turn state, its version and the frame
        numbers reserved are left as stored, so a lapsed segment that wrote
        them since this instance's `load` does not refuse the claim, and a
        request that fails after the claim leaves the open turn's
        reservation as it was. The save that opens the turn reserves its
        frame numbers (`save`'s `seq_reserved`). Once the claim commits, the
        lapsed segment's writes are fenced out: its cursor writes on the
        lease id (`harness.turn.store`), its saves on the live lease this
        claim holds. The caller loads the conversation afresh before it
        begins the turn.

        `segment_id` is the lease's id, minted here when not given, so a
        caller that names it can drop the lease of a claim whose COMMIT may
        have landed though the call raised. `after_turn`, when given, is the
        turn the new one must follow: the claim is refused when the row's
        turn index, read under the row lock that orders every claim, is
        another. The COMMIT and the bookkeeping
        after it run in one task to the end (`harness.db.commit.committing`),
        so a cancel of the caller during the COMMIT lands the claim with the
        live state holding its segment.

        Raises:
            SegmentBusy: the loaded lease had not lapsed by the database
                clock's reading taken with the load.
            ConversationStale: a lease another segment holds is live on the
                row as locked.
            StopPending: a Stop is live for the conversation's open turn
                (`StopState.live`).
            TurnMoved: the row's turn index is not `after_turn`.
        """
        entry = self._runtime[(principal, conversation_id)]
        assert entry.pk is not None and entry.state is not None
        live = entry.state.segment
        if live is not None and live.expires_at > entry.read_at:
            raise SegmentBusy(retry_after_seconds=math.ceil(live.expires_at - entry.read_at))
        lease_id = segment_id if segment_id is not None else _wire_id("seg")
        stamp = self._clock()

        async def write(conn: AsyncConnection) -> Callable[[], TurnLease]:
            row = (await conn.execute(self._lock_conversation, {"pk": entry.pk})).first()
            if row is None:
                raise ConversationStale(conversation_id)
            stored_lease = _row_segment(row)
            if stored_lease is not None and stored_lease.expires_at > row.db_now:
                raise ConversationStale(conversation_id)
            if after_turn is not None and row.turn_index != after_turn:
                raise TurnMoved(conversation_id, latest=row.turn_index)
            if (await _stop_state(conn, row, lease_live=False)).live:
                raise StopPending(conversation_id)
            claimed = (
                await conn.execute(
                    text(
                        "UPDATE conversations SET lease_segment_id = :segment_id, "
                        "lease_owner = :owner, "
                        f"lease_expires_at = {self._time_sql} + :ttl, {_STAMP_STOP}, "
                        "canvas_writes = '[]'::jsonb "
                        "WHERE id = :pk RETURNING lease_expires_at"
                    ),
                    {
                        "minted": _wire_id("stop"),
                        "pk": entry.pk,
                        "segment_id": lease_id,
                        "owner": _OWNER_ID,
                        "ttl": SEGMENT_TTL_S,
                    },
                )
            ).one()

            def committed() -> TurnLease:
                segment = Segment(id=lease_id, owner=_OWNER_ID, expires_at=claimed.lease_expires_at)
                # Read now: the live state may have been evicted and loaded
                # afresh while the claim waited for the write lock.
                if entry.state is not None:
                    entry.state.segment = segment
                self._note_segment(entry, segment)
                entry.lease_stamp = stamp
                entry.held = lease_id
                entry.claimed.set()
                entry.claimed = asyncio.Event()
                assert entry.pk is not None
                return TurnLease(segment_id=lease_id, conversation=entry.pk)

            return committed

        async with entry.write_lock:
            return await committing(self._transaction(), write)

    async def save(
        self,
        principal: str,
        conversation_id: str,
        state: ConversationState | None = None,
        *,
        stopping: bool = False,
        alongside: Callable[[AsyncConnection, int], Awaitable[None]] | None = None,
        seq_reserved: int | None = None,
    ) -> bool:
        """Write this instance's changed durable fields, compare-and-swapped
        on the turn state's version. Given `state`, the working copy `draft`
        handed out, the fields are the copy's, and once the write commits
        each field the copy changed is assigned to the live state, which
        stays as it was when the write fails. Only the keys whose serialised
        value changed since the last load or save on this instance are
        written, and a save with nothing changed writes nothing.

        Raises `ConversationStale` when another instance's claim, save or
        card write moved the version since this instance last read or wrote
        it, or when the stored lease is live on the database clock and is
        neither the segment this save claims nor the one this instance
        holds: a claim over a live lease (a renewal does not move the
        version, so the lease is compared itself), and a lease-less save
        while another instance streams, whose turn state that stream alone
        writes. A renewal, a cancel, a canvas edit or a run-ledger write does
        not move the version. A save that persists `claim_segment`'s claim
        also stores the frame numbers it reserved. Only a claim save, or one
        given `seq_reserved`, moves them, so the end the claim read with its
        load is the stored one.

        Raises `StopPending` when the save persists a claim and a Stop is live
        for the conversation's turn as the locked row and its cursor stand
        (`StopState.live`), unless `stopping`: the claim then acts on the
        Stop, closing the turn `cancelled`.

        `alongside`, when given, runs on the save's connection and in its
        transaction, with the conversation's row id, once the row is locked
        and the checks above passed and before the fields are written: what
        it writes commits with the save or not at all, and anything it raises
        rolls the save back and propagates. It may change the durable fields
        of the state being saved, the segment excepted, and the save writes
        what it changed. It runs even when no field
        changed. The interrupt route admits an approved plan's run through it
        (`harness.run_ledger.StudioRunLedger.admit_with_approval`) and names the run on
        the pending interrupt.

        `seq_reserved`, when given, is the end of the frame numbers the save
        reserves, written with the fields it changes: `/turn`'s save that opens a turn on a
        lease `claim_turn_lease` claimed reserves the new turn's `0` to
        `SEQ_BLOCK`.

        A save that persists `claim_segment`'s claim writes its expiry as
        `SEGMENT_TTL_S` past the database clock's reading at the write and
        sets the state's segment to the expiry stored, and the lease deadline
        (`lease_deadline`) counts from the save's start.

        The write writes the `TURN_LOG_EVENT` line and `turn` outcome of each
        turn close it commits (`_committed_lines`), the abandoned turn a
        `/turn` claim save closes and a turn `/cancel` closes among them, in
        the task that commits it, and starts their export without waiting
        for it (`start_telemetry_flush`), so a claim save adds no export time
        before its stream's first frame. Returns whether it wrote a line; a
        route that ends with no stream awaits `flush_telemetry` itself when
        it did. The commit and everything after it run in one task to the
        end (`harness.db.commit.committing`), so a cancel of the caller cannot
        land the write without its bookkeeping and lines. A cancel before
        the commit rolls the write back. A save that closes a turn and whose
        COMMIT raises is confirmed by its transaction's status
        (`_committing_lines`), and one that committed returns as one that was
        acknowledged.
        """
        entry = self._runtime[(principal, conversation_id)]
        assert entry.pk is not None and entry.state is not None
        target = state if state is not None else entry.state
        async with entry.write_lock:
            delta = _changed_since(_serialize_durable(target), entry.baseline)
            if not delta and alongside is None:
                self._take_draft(entry, target)
                return False
            claimed = _deserialize_segment(delta.get(_KEY_SEGMENT))
            reserved = entry.claimed_reserved if claimed is not None else seq_reserved
            # A segment `claim_segment` claimed, which reserved its frame
            # numbers; its expiry is set at the write.
            claiming = claimed is not None and entry.claimed_reserved is not None
            stamp = self._clock()

            async def write(conn: AsyncConnection) -> tuple[str | None, Callable[[], bool]]:
                assert entry.pk is not None
                changed = delta
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
                if claimed is not None and not stopping:
                    lease_live = stored_lease is not None and stored_lease.expires_at > row.db_now
                    if (await _stop_state(conn, row, lease_live=lease_live)).live:
                        raise StopPending(conversation_id)
                if alongside is not None:
                    await alongside(conn, entry.pk)
                    changed = _changed_since(_serialize_durable(target), entry.baseline)
                lines = _committed_lines(changed, stored=entry.baseline)
                written = (
                    await self._write(
                        conn, entry.pk, changed, seq_reserved=reserved, claim=claiming
                    )
                    if changed
                    else _Written(row.version, row.lease_expires_at)
                )

                def committed() -> bool:
                    nonlocal claimed
                    # The lines first: they hold for the committed write
                    # whatever the bookkeeping below does.
                    wrote_lines = _write_turn_lines(conversation_id, lines)
                    entry.version = written.version
                    entry.baseline.update(changed)
                    if claimed is not None and claiming:
                        assert written.lease_expires_at is not None
                        claimed = dataclasses.replace(claimed, expires_at=written.lease_expires_at)
                        target.segment = claimed
                        entry.baseline[_KEY_SEGMENT] = _serialize_segment(claimed)
                        entry.lease_stamp = stamp
                    self._take_draft(entry, target)
                    if claimed is not None:
                        entry.held = claimed.id
                        entry.claimed.set()
                        entry.claimed = asyncio.Event()
                    if reserved is not None:
                        entry.seq_reserved = reserved
                        entry.claimed_reserved = None
                    return wrote_lines

                return (row.xact if lines else None), committed

            return await self._committing_lines(
                write, what=f"conversation {conversation_id}'s save"
            )

    @staticmethod
    def _take_draft(entry: _RuntimeEntry, state: ConversationState) -> None:
        """Assign each field the working copy `state` changed since `draft`
        took it to the live state, once its save committed. Nothing when
        `state` is the live state itself."""
        if entry.draft is None or entry.draft[0] is not state:
            return
        work, base = entry.draft
        entry.draft = None
        assert entry.state is not None
        _assign_changed(entry.state, work, base)

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
        whoever streams the conversation. One transaction locks the
        conversation's row, reads its stored canvas and the digest entries
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
        when the store does not hold the conversation. This instance's live
        state, when it holds one, takes the canvas read or written unless it
        already holds a later revision.
        """
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
        *,
        lineage: CardLineage | None = None,
        landed: Callable[[T], bool] = bool,
    ) -> T:
        """The write of a card change the turn makes: each card it lands,
        holds, swaps, revises or removes. A write of the whole
        conversation, fenced on the segment the live state holds, with
        `change` applied: the canvas `change` produces and every durable
        field `change` changed commit in one transaction (`_fenced_save`),
        so a stored card always has its digest entry, write token and turn
        line beside it. No other field is written, so a write the turn's
        drive makes (`harness.studio_state.make_remove_cards`) commits
        nothing the stream's owner has changed. `lineage`, the drawn card's
        frame lineage, is serialised before the row lock is taken and commits
        in the same transaction, once per card, when `landed` says from what
        `change` returned that it wrote the card's block: a held card, or a
        revise whose block is gone, stores none. A lineage that cannot be
        stored leaves the card to land without it (`_card_lineage_row`,
        `_insert_card_lineage`).

        Raises `SegmentLost` and writes nothing when the stored lease is not
        that segment, so a stream another instance took over lands no card.
        Otherwise `change` runs on a copy of the live state whose canvas is
        the canvas as stored. It must replace what it changes (a digest
        entry included) and must not mutate it in place, since the copy is
        one level deep. The fields `change` changed are assigned to the live
        state only once the write has committed, so a write that fails
        leaves the live state as it was. A transient database fault is
        retried until the lease's deadline (`_retrying`). The call's write id
        (`_CanvasWrite`), kept across its retries, commits with the write, so
        a retry after a commit whose acknowledgement was lost applies
        `change` no second time and returns what the attempt that committed
        returned. Returns what `change` returns.
        """
        entry = self._runtime[(principal, conversation_id)]
        state = entry.state
        assert entry.pk is not None and state is not None
        segment_id = state.segment.id if state.segment is not None else None
        if segment_id is None:
            raise SegmentLost(f"conversation {conversation_id!r} holds no segment")
        row = _card_lineage_row(lineage) if lineage is not None else None
        write = _CanvasWrite()
        return await self._retrying(
            entry,
            conversation_id,
            lambda: self._fenced_save(
                entry,
                conversation_id,
                segment_id,
                change,
                lineage_row=row,
                landed=landed,
                write=write,
            ),
        )

    async def _retrying[T](
        self,
        entry: _RuntimeEntry,
        conversation_id: str,
        attempt: Callable[[], Awaitable[T]],
    ) -> T:
        """Run `attempt`, a write fenced on the live segment, retrying a
        transient database fault (`compiler.surface.transient`, the
        harness's one fault classifier) with backoff until the lease's
        deadline (`lease_deadline`), then raising `DatabaseUnavailable` from
        the last fault. By then another instance may claim the conversation,
        though none has. Each retry sleeps with no lock held, so the
        heartbeat's renewal does not queue behind it. An attempt whose commit
        landed but whose acknowledgement was lost is retried as well, so
        every attempt must leave the stored row as one attempt would."""
        delays = retry_delays()
        delay = next(delays)
        while True:
            try:
                return await attempt()
            except Exception as exc:
                if not transient(exc):
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
            delay = next(delays)

    async def _fenced_save[T](
        self,
        entry: _RuntimeEntry,
        conversation_id: str,
        segment_id: str,
        change: Callable[[ConversationState], T],
        *,
        lineage_row: Mapping[str, Any] | None = None,
        landed: Callable[[T], bool] = bool,
        write: _CanvasWrite | None = None,
    ) -> T:
        """`change_canvas`'s one write. Under the row lock
        and fenced on `segment_id`, run `change` on a one-level copy of the live
        state holding the canvas as stored, then write the copy's canvas when
        `change` replaced it, and the durable keys below whose value differs
        from `entry.baseline`, and `lineage_row` when given and `landed` holds
        for what `change` returned, in one transaction.
        Once it commits, record the
        written keys in the baseline and the version, and assign the canvas
        read back and each field `change` changed to the live state.

        The durable keys are those `change` changed on the copy. A field the
        stream's owner changed since the last write waits for the owner's
        own write, whichever task makes this one. A field the live stream changes
        during the write keeps its new value, which differs from the
        baseline and goes out with a later write.

        Given `write`, a canvas change's (`change_canvas`), the write also
        records `write.id` in the row's `canvas_writes`. The attempt's copy,
        changed keys and result are kept on `write`, and `write` is queued on
        `entry.canvas_pending` before the commit and taken off once the
        commit is acknowledged. Every fenced write first resolves the
        pending ones from the locked row (`_resolve_canvas_writes`), before
        it copies the live state, so no write builds on a live state that
        lacks a commit whose acknowledgement was lost. A `write` resolved as
        landed returns its kept result, and `change` does not run again. A
        claim empties `canvas_writes`, so an id found there is this
        lease's.

        The write writes the `TURN_LOG_EVENT` line and `turn` outcome of each
        turn close it commits (`_committed_lines`), and a write found landed
        by `_resolve_canvas_writes` writes those of its attempt, which never
        reached them. The commit and everything after it run in one task to
        the end (`harness.db.commit.committing`), so a cancel of the caller
        cannot land the write without its bookkeeping and lines. A cancel
        before the commit rolls the write back."""
        state = entry.state
        assert entry.pk is not None and state is not None
        async with entry.write_lock:

            async def attempt(conn: AsyncConnection) -> Callable[[], T]:
                assert entry.pk is not None and state is not None
                row = (await conn.execute(self._lock_conversation, {"pk": entry.pk})).first()
                if row is None:
                    raise SegmentLost(f"conversation {conversation_id!r} is no longer stored")
                stored = (
                    CompositionManifest.model_validate(row.composition) if row.composition else None
                )
                if entry.canvas_pending and await self._resolve_canvas_writes(
                    conn, entry, state, row, stored
                ):
                    _refold_turn_cards(state, entry.baseline)
                if write is not None and write.landed:
                    landed_result = write.result
                    return lambda: landed_result
                fenced = _row_segment(row)
                if fenced is None or fenced.id != segment_id:
                    raise SegmentLost(
                        f"conversation {conversation_id!r} segment {segment_id!r} is not the lease"
                    )
                base = _shallow_copy(state)
                work = _shallow_copy(state)
                work.composition = stored
                result = change(work)
                durable = _changed_since(_serialize_durable(work), _serialize_durable(base))
                changed = _changed_since(durable, entry.baseline)
                lines = _committed_lines(changed, stored=entry.baseline)
                if work.composition is not stored:
                    await self._write_composition(
                        conn,
                        entry.pk,
                        work.composition.model_dump(mode="json")
                        if work.composition is not None
                        else None,
                    )
                version = (
                    (await self._write(conn, entry.pk, changed)).version if changed else row.version
                )
                if lineage_row is not None and landed(result):
                    await _insert_card_lineage(conn, entry.pk, lineage_row)
                if write is not None:
                    await conn.execute(
                        text(
                            "UPDATE conversations SET canvas_writes = canvas_writes"
                            " || to_jsonb(CAST(:write_id AS text)) WHERE id = :pk"
                        ),
                        {"pk": entry.pk, "write_id": write.id},
                    )
                    write.work, write.base, write.changed = work, base, changed
                    write.result = result
                    entry.canvas_pending.append(write)

                def committed() -> T:
                    # The lines first: they hold for the committed write
                    # whatever the bookkeeping below does.
                    _write_turn_lines(conversation_id, lines)
                    if write is not None and write in entry.canvas_pending:
                        entry.canvas_pending.remove(write)
                    entry.version = version
                    entry.baseline.update(changed)
                    _assign_changed(state, work, base)
                    return result

                return committed

            return await committing(self._transaction(), attempt)

    async def _resolve_canvas_writes(
        self,
        conn: AsyncConnection,
        entry: _RuntimeEntry,
        state: ConversationState,
        row: Any,
        stored: CompositionManifest | None,
    ) -> bool:
        """Settle each write on `entry.canvas_pending` against `row`, the
        conversation's row as locked by the caller (`_fenced_save`,
        `_close_segment`), and take it off the list once settled. A fault
        while reading the row's items raises before any write is settled or
        the live state touched, and the fault fails the calling write: a
        fenced write raises, and the closing write ends `FAILED`, whose
        owner evicts the entry so the next claim reloads from the row.
        Returns whether any write landed.

        A write whose id the row's `canvas_writes` holds committed. The row
        holds that commit and every later one, so the live state reloads
        from it the durable keys the write's `change` changed, the baseline
        reloads the keys the write wrote, the version and the canvas are the
        row's, and the write is marked `landed`. A process-local field
        `change` set (`ConversationState._removal_unreported`) is set on the
        live state. A write whose id the row lacks did not commit: it is
        dropped, and its call's retry runs `change` again on a fresh copy.
        Each landed write's attempt raised before it wrote the lines of the
        turn closes it committed (`_committed_lines`), so they are written
        here, once it is known to have landed."""
        landed = [write for write in entry.canvas_pending if write.id in row.canvas_writes]
        flat: dict[str, Any] = {}
        if landed:
            flat = _row_flat(row)
            for prefix, items in _ITEM_TABLES.items():
                for ident, value in await conn.execute(items.select(), {"pk": row.id}):
                    flat[f"{prefix}{ident}"] = value
        # No await from here on: a write leaves the pending list only once
        # it is settled, so a fault above leaves every write pending.
        for write in landed:
            assert write.work is not None and write.base is not None
            logger.info(
                "studio store: conversation %s canvas write %s was saved by an attempt "
                "whose acknowledgement was lost",
                state.conversation_id,
                write.id,
            )
            before = _serialize_durable(write.base)
            # In the order the write produced them, so new per-item keys
            # land in the live state in that order.
            owned = [
                key
                for key, value in _serialize_durable(write.work).items()
                if before.get(key, _NO_BASELINE_VALUE) != value and key in flat
            ]
            _apply_delta(state, {key: flat[key] for key in owned})
            _write_turn_lines(
                state.conversation_id,
                _committed_lines(write.changed, stored=entry.baseline),
            )
            entry.baseline.update({key: flat[key] for key in write.changed if key in flat})
            for name in _PROCESS_LOCAL_FIELDS:
                value = getattr(write.work, name)
                if value != getattr(write.base, name):
                    setattr(state, name, value)
            write.landed = True
            entry.canvas_pending.remove(write)
        for write in [w for w in entry.canvas_pending if w.id not in row.canvas_writes]:
            write.work = write.base = None
            write.changed = {}
            entry.canvas_pending.remove(write)
        if not landed:
            return False
        entry.version = row.version
        state.composition = stored
        return True

    async def renew(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        *,
        timeout: float = 5.0,
    ) -> Renewal:
        """Push `segment_id`'s expiry to `SEGMENT_TTL_S` past the database
        clock's reading in one conditional update of the lease columns, which
        never disturbs whatever the live stream is doing to `entry.state` and
        never writes the rest of the conversation.

        `Renewal.held` is `False` when the stored segment is no longer
        `segment_id` (another instance took over after this one's TTL
        lapsed), and the caller ends the stream fatally. Otherwise
        `Renewal.stop_id` is the Stop live for the lease's turn
        (`StopState.live`), read after the update locked the row. A set
        `lease_cancel` is recorded as a fresh id naming the row's turn index
        in the same update (`_STAMP_STOP`). The renewal marks nothing read,
        so later renewals report the same Stop until its turn closes or a
        pause write records it taken (`harness.turn.store.TurnStore.pause`).
        A database error propagates, as does a `TimeoutError` when the renewal
        takes longer than `timeout`. The caller's own heartbeat decides how
        long to keep trying.

        Runs to the end even when the caller is cancelled
        (`run_to_completion`), since the heartbeat that calls it is
        cancelled whenever its stream ends.
        """
        return await run_to_completion(
            self._renew(principal, conversation_id, segment_id, timeout=timeout)
        )

    async def _renew(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float
    ) -> Renewal:
        """`renew`'s write, run as its own task."""
        stamp = self._clock()
        async with asyncio.timeout(timeout):
            entry = self._entry(principal, conversation_id)
            async with entry.write_lock:
                async with self._transaction() as conn:
                    row = (
                        await conn.execute(
                            text(
                                "UPDATE conversations"
                                f" SET lease_expires_at = {self._time_sql} + :ttl, {_STAMP_STOP}"
                                " WHERE principal = :principal"
                                " AND conversation_id = :conversation_id"
                                " AND lease_segment_id = :segment_id"
                                " RETURNING id, lease_owner, lease_expires_at, lease_cancel,"
                                f" stop_id, stop_turn, {_TURN_INDEX_SQL} AS turn_index"
                            ),
                            {
                                "principal": principal,
                                "conversation_id": conversation_id,
                                "segment_id": segment_id,
                                "ttl": SEGMENT_TTL_S,
                                "minted": _wire_id("stop"),
                            },
                        )
                    ).first()
                    if row is None:
                        return Renewal(held=False, stop_id=None)
                    stop = await _stop_state(conn, row, lease_live=True)
                entry.lease_stamp = max(entry.lease_stamp, stamp)
                self._note_segment(
                    entry,
                    Segment(id=segment_id, owner=row.lease_owner, expires_at=row.lease_expires_at),
                )
                return Renewal(held=True, stop_id=stop.stop_id if stop.live else None)

    def _note_segment(self, entry: _RuntimeEntry, segment: Segment) -> None:
        """Record a lease write in this instance's baseline, and in its live
        state when that still holds the same segment."""
        entry.baseline[_KEY_SEGMENT] = _serialize_segment(segment)
        live = entry.state
        if live is not None and live.segment is not None and live.segment.id == segment.id:
            live.segment = segment

    async def close_segment(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float = 5.0
    ) -> SegmentClose:
        """The closing write of the stream that holds `segment_id`: one
        transaction, under the row lock and fenced on `segment_id`, that
        stores this instance's changed durable fields (the turn's record,
        the pending interrupts and `seq_next` among them) and clears the
        lease. The canvas is read back from the row, as `change_canvas` reads
        it, since every card commits as it lands. Before it copies the live
        state, it settles the canvas writes whose acknowledgement was lost
        from the locked row (`_resolve_canvas_writes`), as every fenced write
        does, and folds a landed card's turn line into a record the stream
        built without it (`_refold_turn_cards`).

        When a Stop is live for the open turn as the locked row and its
        cursor stand (`StopState.live`, with this segment's own lease
        counted live), the same transaction closes the turn `cancelled`. The
        cursor closes through `harness.turn.store.finish_on`, which ends the
        conversation's unfinished runs. The record keeps the cursor's
        records and clears the pending interrupts, so no answer reaches
        them. A turn with no cursor, an Edit Card's, is open while its
        record is, and a Stop on the lease closes its record the same way. A
        record the stream's own close already wrote on a failure, its cursor
        still open, is recorded `cancelled` in its place, so record, cursor
        and stream agree. A Stop that commits after this transaction finds
        no live lease, and `/cancel` takes it under the conversation lock.

        Returns how the write ended (`SegmentClose`). It makes one attempt
        and raises nothing. `timeout` bounds the attempt up to its COMMIT,
        the wait for this instance's write lock included. The COMMIT and the
        work owed to it run to their end (`harness.db.commit.committing`),
        bounded by the connection's `command_timeout`
        (`harness.db.engine.COMMAND_TIMEOUT_S`, 90 s), so a COMMIT that lands
        after the timeout returns as committed. A COMMIT that raises is read
        back in the same task, whatever the timeout, for up to
        `COMMIT_READ_BACK_LIMIT_S` (5 s, `_committing_lines`): when its
        transaction committed, the write landed with its acknowledgement
        lost, and its lines and bookkeeping run then, with no second write.
        The caller waits for that task, so its wait is `timeout` up to the
        COMMIT, then the COMMIT and any read back. Any other failure is logged and
        returned as `FAILED`, and the lease then lapses by its TTL. A COMMIT
        the server finishes after the read back gave up leaves its record in
        the row with no line written. The write settles the
        pending canvas writes under the row lock and rebuilds the record's
        cards from the row's items. A fault reading those items fails the
        closing write. After `LOST` or `FAILED` the live state may hold what
        was not stored, so the caller evicts it (`evict`) and the next claim
        reloads from the row. Either way this instance stops counting
        `segment_id` as held.
        """
        entry = self._runtime.get((principal, conversation_id))
        if entry is None or entry.state is None or entry.pk is None:
            logger.error(
                "studio store: conversation %s is not loaded; segment %s closes unsaved",
                conversation_id,
                segment_id,
            )
            return SegmentClose.FAILED
        landed: list[SegmentClose] = []
        try:
            try:
                async with asyncio.timeout(timeout), entry.write_lock:
                    return await self._close_segment(
                        entry, entry.state, conversation_id, segment_id, landed=landed
                    )
            except TimeoutError:
                if not landed:
                    logger.warning(
                        "studio store: conversation %s segment %s's closing write did not "
                        "complete within %ss",
                        conversation_id,
                        segment_id,
                        timeout,
                    )
            except Exception:
                logger.exception(
                    "studio store: conversation %s segment %s's closing write failed",
                    conversation_id,
                    segment_id,
                )
            if landed:
                # The commit landed after the timeout: the write stands, and
                # so do its lines.
                logger.warning(
                    "studio store: conversation %s segment %s's closing write committed "
                    "after its %ss timeout",
                    conversation_id,
                    segment_id,
                    timeout,
                )
                return landed[0]
            logger.warning(
                "studio store: conversation %s segment %s's closing write saved nothing; the "
                "lease lapses by its TTL",
                conversation_id,
                segment_id,
            )
            return SegmentClose.FAILED
        finally:
            if entry.held == segment_id:
                entry.held = None

    async def _close_segment(
        self,
        entry: _RuntimeEntry,
        state: ConversationState,
        conversation_id: str,
        segment_id: str,
        *,
        landed: list[SegmentClose],
    ) -> SegmentClose:
        """`close_segment`'s transaction. The caller holds the write lock.
        The commit and what follows it run to the end
        (`harness.db.commit.committing`), and once it commits the result is
        put on `landed` first, for a caller whose timeout cut in during the
        commit. A COMMIT that raises is read back in the same task, by the
        write's own transaction id (`_committing_lines`), and one that
        landed runs the work owed to its commit as one that returned does.
        Bookkeeping that raises after the commit is logged and drops the
        live state, as `evict` does, so the next request loads the row the
        commit wrote; the dropped state keeps the canvas the write committed,
        for the stream's ending frame."""

        async def write(conn: AsyncConnection) -> tuple[str | None, Callable[[], SegmentClose]]:
            assert entry.pk is not None
            row = (await conn.execute(self._lock_conversation, {"pk": entry.pk})).first()
            lease = _row_segment(row) if row is not None else None
            if row is None or lease is None or lease.id != segment_id:
                logger.warning(
                    "studio store: conversation %s segment %s is not the lease; its closing "
                    "write saved nothing",
                    conversation_id,
                    segment_id,
                )
                return None, lambda: SegmentClose.LOST
            if entry.canvas_pending:
                stored = (
                    CompositionManifest.model_validate(row.composition) if row.composition else None
                )
                if await self._resolve_canvas_writes(conn, entry, state, row, stored):
                    _refold_turn_cards(state, entry.baseline)
            stop = await _stop_state(conn, row, lease_live=True)
            # A Stop on this segment's own lease is live for it whatever
            # the cursor. A turn with no cursor (an Edit Card's) is open
            # until its record closes.
            stopped = stop.live and (stop.cursor_status is not None or not state.turn_recorded)
            base = _shallow_copy(state)
            work = _shallow_copy(state)
            work.composition = (
                CompositionManifest.model_validate(row.composition) if row.composition else None
            )
            work.segment = None
            if stopped:
                cursor = await cursor_at(conn, row.id, row.turn_index)
                now = datetime.fromtimestamp(self._clock(), UTC)
                if cursor is not None:
                    await finish_on(
                        conn, cursor, segment_id=segment_id, status=TurnStatus.CANCELLED, now=now
                    )
                else:
                    await end_unfinished(conn, row.id, now=now)
                work.pending = {}
                if work.turn_recorded:
                    # The work's own close recorded the turn on its failure
                    # while its cursor stayed open. The Stop closes it
                    # `cancelled`, as the cursor and the stream say.
                    work.turns = [
                        *work.turns[:-1],
                        {**work.turns[-1], "error": ErrorCode.CANCELLED.value},
                    ]
                else:
                    if cursor is not None:
                        keep_turn_records(work, cursor.state)
                    work.close_turn(
                        final_text=work.turn_final_text, error=ErrorCode.CANCELLED.value
                    )
            elif work.turn_recorded and work.turn_index:
                await self._close_cursor_with_record(conn, row.id, work, segment_id)
            changed = _changed_since(_serialize_durable(work), entry.baseline)
            lines = _committed_lines(changed, stored=entry.baseline)
            version = (
                (await self._write(conn, entry.pk, changed)).version if changed else row.version
            )

            def committed() -> SegmentClose:
                # What landed first, then the lines: both hold for the
                # committed write whatever the bookkeeping below does.
                closed = SegmentClose.STOPPED if stopped else SegmentClose.SAVED
                landed.append(closed)
                _write_turn_lines(conversation_id, lines)
                try:
                    _assign_changed(state, work, base)
                    entry.version = version
                    entry.baseline.update(changed)
                except Exception:
                    logger.exception(
                        "studio store: conversation %s segment %s's closing write committed "
                        "and its bookkeeping failed; the live state is dropped",
                        conversation_id,
                        segment_id,
                    )
                    # The stream's ending frame shows the canvas the write
                    # committed (`harness.studio_stream`).
                    state.composition = work.composition
                    entry.state = None
                    entry.draft = None
                    entry.baseline = {}
                return closed

            return row.xact, committed

        return await self._committing_lines(
            write, what=f"conversation {conversation_id} segment {segment_id}'s closing write"
        )

    async def _close_cursor_with_record(
        self, conn: AsyncConnection, pk: int, work: ConversationState, segment_id: str
    ) -> None:
        """Close the cursor of `work`'s recorded turn on `conn`, in the
        closing write's transaction, when it is still open: with the status
        the record states and the record's code as its error
        (`record_status`, `recorded_close_error`). A segment whose node
        failed and whose own close of the cursor failed, a value JSON could
        not encode say, records the turn failed and leaves the cursor
        running; this keeps the two together, so a committed record always
        comes with a closed cursor."""
        at = (await conn.execute(_TURN_PAUSE, {"pk": pk, "turn_index": work.turn_index})).first()
        if at is None or TurnStatus(at.status) in CLOSED:
            return
        cursor = await cursor_at(conn, pk, work.turn_index)
        assert cursor is not None
        record = work.turns[-1]
        await finish_on(
            conn,
            cursor,
            segment_id=segment_id,
            status=record_status(record),
            error=recorded_close_error(record),
            now=datetime.fromtimestamp(self._clock(), UTC),
        )

    async def _committing_lines[T](
        self,
        write: Callable[[AsyncConnection], Awaitable[tuple[str | None, Callable[[], T]]]],
        *,
        what: str,
    ) -> T:
        """Run `write` in a transaction of its own through
        `harness.db.commit.committing`, and return what the function it
        returns gives once the transaction commits. `write` also returns the
        id of its transaction (`_lock_conversation`'s `xact`) when it is to
        be confirmed, and `None` otherwise: a save or a record of a closed
        turn returns it when what it commits carries a turn line, and the
        closing write (`_close_segment`) always does, since its
        `SegmentClose` tells the stream whether its state was saved. A COMMIT that
        raises is then confirmed by that transaction's status, read for up
        to `COMMIT_READ_BACK_LIMIT_S` (`_transaction_committed`): one that
        committed runs the function and returns as a COMMIT that returned
        does, so its lines are written once; any other raises. A write that
        returned `None`, or a status still unknown at the limit, raises the
        COMMIT's error and writes nothing."""
        xacts: list[str] = []

        async def run(conn: AsyncConnection) -> Callable[[], T]:
            xact, committed = await write(conn)
            if xact is not None:
                xacts.append(xact)
            return committed

        async def read_back() -> bool:
            return bool(xacts) and await self._transaction_committed(xacts[0], what)

        return await committing(self._transaction(), run, read_back=read_back)

    async def _transaction_committed(self, xact: str, what: str) -> bool:
        """Whether `what`'s transaction `xact`, whose COMMIT raised,
        committed (`harness.db.commit.transaction_committed`, reading
        `_xact_status` for up to `COMMIT_READ_BACK_LIMIT_S`). `False` when
        it aborted, is unknown, or is still unsettled or unreadable at the
        limit: that last case, a COMMIT that lands later, leaves what it
        stored in the row with no line written."""
        landed = await transaction_committed(
            lambda: self._xact_status(xact), what=what, limit_s=COMMIT_READ_BACK_LIMIT_S
        )
        if landed:
            logger.warning("studio store: %s landed with its acknowledgement lost", what)
        return landed

    async def _xact_status(self, xact: str) -> str | None:
        """The status Postgres holds for transaction `xact`: `committed`,
        `aborted`, `in progress`, or `None` once it is too old to tell."""
        async with self._transaction() as conn:
            return (
                await conn.execute(
                    text("SELECT pg_xact_status(CAST(CAST(:xact AS text) AS xid8))"), {"xact": xact}
                )
            ).scalar_one()

    def evict(self, principal: str, conversation_id: str) -> None:
        """Forget this instance's live state for the conversation, so the
        next request loads the row (`load`). The owner of a stream whose
        closing write ended `LOST` or `FAILED` calls it before it releases
        the conversation's lock, since the live state then holds what the
        row does not. It stops counting any segment as held here, since a
        stream evicted without a closing write left a lease it no longer
        writes under. The runtime entry, with its locks, stays."""
        entry = self._runtime.get((principal, conversation_id))
        if entry is None:
            return
        entry.state = None
        entry.draft = None
        entry.baseline = {}
        entry.held = None

    async def drop_lease(
        self, principal: str, conversation_id: str, segment_id: str, *, timeout: float = 5.0
    ) -> None:
        """Clear the stored lease when it is `segment_id` and write nothing
        else. This is one conditional update of the lease columns, which
        leaves the turn state and its version as the claim save stored them
        and records a Stop set on the lease (`_STAMP_STOP`). It releases a
        claim whose stream never started (a route failing between its claim
        save and its response), so a retry of the same request claims at
        once and finds what the claim save stored, a resuming pending entry
        included. Whatever the failed request changed in this instance's
        live state after its claim save stays unsaved, and the retry's
        `load` reads the row afresh.

        Runs to the end even when the caller is cancelled, and raises nothing
        but that cancellation. A clear that cannot land is logged, and the
        lease then expires by its own TTL.
        """
        await run_to_completion(
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
                            f"lease_owner = NULL, lease_expires_at = NULL, {_STAMP_STOP} "
                            "WHERE principal = :principal "
                            "AND conversation_id = :conversation_id "
                            "AND lease_segment_id = :segment_id"
                        ),
                        {
                            "principal": principal,
                            "conversation_id": conversation_id,
                            "segment_id": segment_id,
                            "minted": _wire_id("stop"),
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
        """Record a Stop request on `conversation_id`'s live segment, whoever
        owns it, this instance included, by setting `lease_cancel`. The
        segment's next renewal (`renew`) records it under a fresh id and
        reports it, and a segment that reads only the flag sees it as well.
        One conditional update of the lease columns.

        Returns whether a live segment was found and signalled. `False`
        means there is none, or the one on record has already expired by the
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
                ),
            )
            return True

    async def stop_live(self, principal: str, conversation_id: str) -> bool:
        """Whether a Stop is live for `conversation_id`'s turn
        (`StopState.live`). The row is locked `FOR SHARE` first, so a lease
        write in flight commits before the read, and the turn's cursor is
        read in a later statement. `False` when the store does not hold the
        conversation."""
        async with self._transaction() as conn:
            row = (
                await conn.execute(
                    text(
                        f"SELECT {_CONVERSATION_COLUMNS}, {self._time_sql} AS db_now "
                        "FROM conversations WHERE principal = :principal "
                        "AND conversation_id = :conversation_id FOR SHARE"
                    ),
                    {"principal": principal, "conversation_id": conversation_id},
                )
            ).first()
            if row is None:
                return False
            lease = _row_segment(row)
            live_lease = lease is not None and lease.expires_at > row.db_now
            return (await _stop_state(conn, row, lease_live=live_lease)).live

    def claimed_event(self, principal: str, conversation_id: str) -> asyncio.Event:
        """An event set when the next claim save on this instance commits
        for `conversation_id`."""
        return self._entry(principal, conversation_id).claimed

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

    async def card_lineage(
        self,
        principal: str,
        conversation_id: str,
        *,
        component_id: str | None = None,
        frame: tuple[str | None, str | None] | None = None,
    ) -> CardLineage | None:
        """The stored lineage of card `component_id`, else of a card drawn from
        `frame`, else of a card whose write named `frame`
        (`harness.provenance.CardLineages`), read on the conversation this
        instance loaded over the lineage engine, in one attempt with no lock
        and no retry: the caller bounds the read and goes on without it. One
        row is read; only when it does not decode, logged with its key, is the
        next read, leaving it out. `None` when none is stored or none decodes.

        Raises:
            LookupError: This instance has not loaded the conversation.
        """
        spec_id, content_version = frame if frame is not None else (None, None)
        if component_id is None and (spec_id is None or content_version is None):
            return None
        entry = self._runtime.get((principal, conversation_id))
        if entry is None or entry.pk is None:
            raise LookupError(f"conversation {conversation_id!r} is not loaded here")
        skipped: list[str] = []
        async with self._lineage_engine.begin() as conn:
            while True:
                row = (
                    await conn.execute(
                        _SELECT_CARD_LINEAGE,
                        {
                            "pk": entry.pk,
                            "component_id": component_id,
                            "spec_id": spec_id,
                            "content_version": content_version,
                            "skipped": skipped,
                        },
                    )
                ).first()
                if row is None:
                    return None
                try:
                    lineage = lineage_from_json(row.members)
                except CardLineageDecodeError:
                    logger.exception(
                        "studio store: skipping the lineage row of conversation %s "
                        "component_id=%r, which does not decode",
                        conversation_id,
                        row.component_id,
                    )
                    skipped.append(row.component_id)
                    continue
                break
        requested = (row.requested_spec_id, row.requested_content_version)
        return CardLineage(
            component_id=row.component_id,
            frame=(row.spec_id, row.content_version),
            lineage=lineage,
            stat_var_count=row.stat_var_count,
            requested=requested if requested != (None, None) else None,
        )

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
