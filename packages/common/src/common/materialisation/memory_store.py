"""`InMemoryMaterialisationStore`: the in-process `MaterialisationStore`
implementation, keyed by the plain `(spec_id, content_version)` tuple a
`FrameRef` names -- content-addressed, not session-scoped, so `put` is
idempotent and `get` never gates on who is asking.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from common.materialisation.frame_ref import FrameRef
from common.materialisation.store import FrameExpired, FrameTooLarge, HeldFrame
from common.materialisation.trace import StoredFrameTrace

if TYPE_CHECKING:
    from dfo_frame_grammar import Frame

# Matches the Datasets working tier's own retention window.
FRAME_TTL = timedelta(days=30)

# Mirrors `data_mcp.definition_store.caching.DEFAULT_MAX_ENTRIES`'s own bound on a
# long-lived shared process: every `get_data` mint and every verb result adds an
# entry here, so this store needs the same cap its sibling already has.
DEFAULT_MAX_ENTRIES = 256


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def _held(
    ref: FrameRef, frame: Frame, trace: StoredFrameTrace, *, max_rows: int
) -> tuple[Frame, StoredFrameTrace]:
    if frame.num_rows > max_rows:
        raise FrameTooLarge(
            ref, reason=f"frame has {frame.num_rows} rows, over this read's {max_rows}-row limit"
        )
    return frame, trace


@dataclass(frozen=True, kw_only=True)
class _StoredEntry:
    frame: Frame
    trace: StoredFrameTrace
    created_at: datetime


class InMemoryMaterialisationStore:
    """The in-process `MaterialisationStore`: a plain dict keyed by
    `(spec_id, content_version)`, lost on process restart -- acceptable for
    the same reason `data_mcp.definition_store.InMemoryDefinitionStore` is.

    Two separate things share `_entries`' `OrderedDict` but are kept apart:
    `created_at` is set once, at an entry's first `put`, and never touched again --
    it's the TTL clock, absolute from first write. The dict's own position (moved to
    the end on every `put` or `get`) is a second, independent axis: recency of touch,
    used only to pick an eviction victim under `max_entries`. Retention doesn't
    refresh on touch, so position order and `created_at` order can diverge -- a
    `get` on the oldest-written entry moves it to the front of the eviction queue
    without changing when it expires -- so `_drop_expired` scans every entry rather
    than assuming the dict is sorted by age.
    """

    def __init__(
        self, *, now: Callable[[], datetime] = _utcnow, max_entries: int = DEFAULT_MAX_ENTRIES
    ) -> None:
        self._now = now
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], _StoredEntry] = OrderedDict()

    async def put(
        self, ref: FrameRef, *, frame: Frame, trace: StoredFrameTrace, writer: str
    ) -> None:
        del writer  # kept on the signature for a later entitlement check; unread here
        key = (ref.spec_id, ref.content_version)
        existing = self._entries.get(key)
        if existing is not None:
            if self._now() - existing.created_at <= FRAME_TTL:
                # Live entry: keeps the existing frame, trace, and created_at exactly
                # as first written -- an idempotent re-put, so it never counts against
                # `max_entries`. Still touched for eviction-recency purposes.
                self._entries.move_to_end(key)
                return
            # Expired but not yet evicted: this write heals it exactly as a fresh
            # insert under this key would (a new created_at, this call's own
            # frame/trace) -- `ref` is content-addressed, so any two writes under it
            # are equivalent bytes either way.
        elif len(self._entries) >= self._max_entries:
            self._drop_expired()
            if len(self._entries) >= self._max_entries:
                # Evicting a live entry to make room is safe by design: a caller that
                # later resolves this exact pair gets `frame_not_held`, whose own fix
                # text already tells the model to re-run `get_data` (or the verb) to
                # re-materialise it.
                self._entries.popitem(last=False)
        self._entries[key] = _StoredEntry(frame=frame, trace=trace, created_at=self._now())
        self._entries.move_to_end(key)

    async def get(self, ref: FrameRef) -> tuple[Frame, StoredFrameTrace] | None:
        entry = self._live_entry(ref)
        return None if entry is None else (entry.frame, entry.trace)

    async def open(self, ref: FrameRef) -> HeldFrame | None:
        """The held entry, with its frame's exact row count, whose `read` returns the
        frame this store already holds decoded, so the claim a caller takes before it
        covers no new allocation here."""
        entry = self._live_entry(ref)
        if entry is None:
            return None
        return HeldFrame(
            num_rows=entry.frame.num_rows,
            read=functools.partial(_held, ref, entry.frame, entry.trace),
        )

    def _live_entry(self, ref: FrameRef) -> _StoredEntry | None:
        key = (ref.spec_id, ref.content_version)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._now() - entry.created_at > FRAME_TTL:
            raise FrameExpired(ref)
        self._entries.move_to_end(key)
        return entry

    async def aclose(self) -> None:
        pass

    def _drop_expired(self) -> None:
        """Removes every expired entry. A full scan, not an oldest-touched-first
        early return: `created_at` is fixed at first write while an entry's dict
        position moves on every touch, so the two orders can disagree and no prefix
        of `_entries` is reliably all-expired.
        """
        now = self._now()
        expired_keys = [
            key for key, entry in self._entries.items() if now - entry.created_at > FRAME_TTL
        ]
        for key in expired_keys:
            del self._entries[key]
