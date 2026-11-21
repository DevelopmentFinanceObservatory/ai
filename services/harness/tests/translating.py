"""A segment's translation for a test that drives the segment itself: its
events marked and queued as `harness.studio._SegmentWork` queues them, and
translated as the stream's owner translates them."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from harness.studio import event_key
from harness.studio_stream import _Events
from harness.studio_translate import RecordMarks, SegmentEvent, _EventRecords, _Translator


async def translated(translator: _Translator, event: SegmentEvent) -> list[Any]:
    """The frames of `event`, after those of every call recorded so far."""
    marks = _EventRecords(translator._recorder, translator._surface_recorder)
    return await translator.translate_recorded(*marks.recorded(event))


class Relay:
    """`translator`'s events, queued by `emit` as a drive emits them and
    translated by `frames` as an owner takes them."""

    def __init__(self, translator: _Translator) -> None:
        self._translator = translator
        self._marks = _EventRecords(translator._recorder, translator._surface_recorder)
        self._events: _Events[tuple[SegmentEvent, RecordMarks]] = _Events()

    def emit(self, event: SegmentEvent) -> None:
        self._events.put(self._marks.recorded(event), key=event_key(event))

    async def frames(self, segment: asyncio.Future[Any]) -> AsyncIterator[Any]:
        """The frames of each event emitted, in order, until `segment` is
        done and every event it emitted is translated."""
        while True:
            while self._events:
                for frame in await self._translator.translate_recorded(*self._events.take()):
                    yield frame
            if segment.done():
                return
            arrived = asyncio.ensure_future(self._events.arrived())
            try:
                await asyncio.wait((arrived, segment), return_when=asyncio.FIRST_COMPLETED)
            finally:
                arrived.cancel()
