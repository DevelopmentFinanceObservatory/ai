"""The run ledger kept in a conversation's ledger session.

`StudioRunLedger` is the `executor.ledger.RunLedger` the studio harness runs
under. Each run is one item, `studio:run:{run_id}`, and each idempotency key
one index item, `studio:run_key:{key}`, naming its run. The canonical
submission a run executes is one more item, `studio:run_submission:{run_id}`,
written in the admission's own event and never again, and the `get_data`
results its inputs were sealed from another, `studio:run_retrievals:{run_id}`,
written in the event of each seal that changes the inputs. The plan that
compiled a run's work is one item per revision, `studio:run_plan:{revision}`,
written before the run is admitted and never again. Every admission and
every change is its own event, appended through
`StudioStore.append_run_items` and fenced on the conversation's live segment,
so a run's record is durable before the dispatch it precedes, and a writer
whose segment another instance took over gets `SegmentLost` and records
nothing, publication included. A stale append decides again against the
ledger as the other writer left it, which is how two concurrent admissions of
one key settle on one run. A stored run is read with
`executor.ledger.stored_run`, so one another release wrote loads. A stored
run this release still cannot read is recorded failed
(`run_record_unreadable`) where it is found unfinished or its key admits
work again, its status and failure written over the stored item. The item
stays unreadable but is failed, so the run is never carried on again and its
key is free for a new run.

The ledger is built once per runner, beside the recorder, so it finds its
conversation the way the recorder does: `current_conversation_key`, set by
`harness.studio` for each streamed segment. The segment it fences on is the
one that conversation's held `ConversationState` claimed. Outside a
conversation it reads nothing and refuses every write with `SegmentLost`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from executor.ledger import (
    TERMINAL_STATUSES,
    Admission,
    RunChange,
    RunNotFoundError,
    RunRecord,
    RunStatus,
    Seal,
    admit_against,
    advance,
    stored_run,
)
from pydantic import JsonValue, ValidationError

from harness.conversation import current_conversation_key
from harness.studio_state import (
    RUN_ITEM_PREFIX,
    RUN_KEY_ITEM_PREFIX,
    RUN_PLAN_ITEM_PREFIX,
    RUN_RETRIEVALS_ITEM_PREFIX,
    RUN_SUBMISSION_ITEM_PREFIX,
    SegmentLost,
    StudioStore,
)
from harness.tool_contracts import HarnessCode

logger = logging.getLogger(__name__)

_TERMINAL = frozenset(status.value for status in TERMINAL_STATUSES)


def _unreadable(raw: Any, reason: str) -> dict[str, Any]:
    """Stored run item `raw`, which this release cannot read, recorded failed."""
    return {
        **raw,
        "status": RunStatus.FAILED.value,
        "failure": {"code": HarnessCode.RUN_RECORD_UNREADABLE.value, "reason": reason},
    }


def _status_of(raw: Any) -> str | None:
    """The status a stored run item names, read without validating the rest."""
    status = raw.get("status") if isinstance(raw, dict) else None
    return status if isinstance(status, str) else None


def _run_item(run_id: str) -> str:
    return f"{RUN_ITEM_PREFIX}{run_id}"


def _key_item(idempotency_key: str) -> str:
    return f"{RUN_KEY_ITEM_PREFIX}{idempotency_key}"


def _submission_item(run_id: str) -> str:
    return f"{RUN_SUBMISSION_ITEM_PREFIX}{run_id}"


def _retrievals_item(run_id: str) -> str:
    return f"{RUN_RETRIEVALS_ITEM_PREFIX}{run_id}"


def _plan_item(revision: str) -> str:
    return f"{RUN_PLAN_ITEM_PREFIX}{revision}"


class StudioRunLedger:
    """A `RunLedger` over `store`'s conversation ledger sessions."""

    def __init__(self, store: StudioStore) -> None:
        self._store = store

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._store.now(), UTC)

    def _segment(self) -> tuple[str, str, str]:
        """The current conversation's `(principal, conversation_id,
        segment_id)`, the segment every write is fenced on."""
        key = current_conversation_key.get()
        state = self._store.current(*key) if key is not None else None
        if key is None or state is None or state.segment is None:
            raise SegmentLost("no conversation segment is streaming on this instance")
        return key[0], key[1], state.segment.id

    async def _items(self) -> dict[str, Any]:
        key = current_conversation_key.get()
        return await self._store.run_items(*key) if key is not None else {}

    async def admit(self, candidate: RunRecord, *, submission: JsonValue) -> Admission:
        admitted: list[Admission] = []

        def items_for(items: dict[str, Any]) -> dict[str, Any]:
            admitted.clear()
            existing_id = items.get(_key_item(candidate.idempotency_key))
            item = _run_item(existing_id) if existing_id is not None else None
            raw = items.get(item) if item is not None else None
            # A failed run binds nothing (`admit_against`), readable or not,
            # and one this release cannot read is recorded failed here.
            failed: dict[str, Any] = {}
            existing = None
            if item is not None and isinstance(raw, dict) and _status_of(raw) != RunStatus.FAILED:
                try:
                    existing = stored_run(raw)
                except ValidationError as exc:
                    logger.error("run ledger: stored run %s is unreadable: %s", existing_id, exc)
                    reason = f"the stored run is unreadable: {exc}"
                    failed[item] = _unreadable(raw, reason)
            admission = admit_against(existing, candidate, now=self._now())
            admitted.append(admission)
            if admission.resumed:
                return {}
            return failed | {
                _run_item(candidate.run_id): candidate.model_dump(mode="json"),
                _key_item(candidate.idempotency_key): candidate.run_id,
                _submission_item(candidate.run_id): submission,
            }

        await self._store.append_run_items(*self._segment(), items_for)
        return admitted[0]

    async def get(self, run_id: str) -> RunRecord | None:
        raw = (await self._items()).get(_run_item(run_id))
        return stored_run(raw) if raw is not None else None

    async def submission(self, run_id: str) -> JsonValue:
        return (await self._items()).get(_submission_item(run_id))

    async def retrievals(self, run_id: str) -> JsonValue:
        return (await self._items()).get(_retrievals_item(run_id))

    async def store_plan(self, revision: str, plan: JsonValue) -> None:
        item = _plan_item(revision)

        def items_for(items: dict[str, Any]) -> dict[str, Any]:
            return {} if item in items else {item: plan}

        await self._store.append_run_items(*self._segment(), items_for)

    async def plan(self, revision: str) -> JsonValue:
        return (await self._items()).get(_plan_item(revision))

    async def unfinished(self) -> tuple[RunRecord, ...]:
        runs: list[RunRecord] = []
        unreadable: dict[str, str] = {}
        for item, raw in (await self._items()).items():
            # `carried_on`, read off the stored item before validating it.
            status = _status_of(raw)
            if not item.startswith(RUN_ITEM_PREFIX) or status == RunStatus.FAILED:
                continue
            if status == RunStatus.COMPLETED and raw.get("delivered") is True:
                continue
            try:
                runs.append(stored_run(raw))
            except ValidationError as exc:
                logger.error("run ledger: stored run %s is unreadable: %s", item, exc)
                unreadable[item] = f"the stored run is unreadable: {exc}"
        if unreadable:
            await self._fail_unreadable(unreadable)
        return tuple(runs)

    async def _fail_unreadable(self, reasons: dict[str, str]) -> None:
        """Record each stored run item in `reasons` failed, with its reason."""

        def items_for(items: dict[str, Any]) -> dict[str, Any]:
            return {
                item: _unreadable(items[item], reason)
                for item, reason in reasons.items()
                if isinstance(items.get(item), dict) and _status_of(items[item]) not in _TERMINAL
            }

        await self._store.append_run_items(*self._segment(), items_for)

    async def record(self, run_id: str, change: RunChange) -> RunRecord:
        recorded: list[RunRecord] = []

        def items_for(items: dict[str, Any]) -> dict[str, Any]:
            recorded.clear()
            raw = items.get(_run_item(run_id))
            if raw is None:
                raise RunNotFoundError(run_id)
            run = stored_run(raw)
            advanced = advance(run, change, now=self._now())
            recorded.append(advanced)
            if advanced == run:
                return {}
            items = {_run_item(run_id): advanced.model_dump(mode="json")}
            if isinstance(change, Seal) and advanced.manifest != run.manifest:
                items[_retrievals_item(run_id)] = list(change.retrievals)
            return items

        await self._store.append_run_items(*self._segment(), items_for)
        return recorded[0]
