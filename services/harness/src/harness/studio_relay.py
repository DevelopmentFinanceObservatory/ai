"""The response of a Studio stream: it relays the frames its owner puts on a
channel (`harness.studio_stream`) as SSE.

`StudioStream` serialises the channel's frames as SSE `data:` lines and sends
a `: ka` comment while the next frame is slow. A `finally` around its ASGI
call sets the client-gone flag and waits, shielded, for the owner to finish,
however the call ends: the client left, the body ended, or the server
cancelled the request. It holds the channel, the flag and the owner's task,
and nothing of the conversation or the store: this module imports neither.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import override

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

SSE_MEDIA_TYPE = "text/event-stream"

# How long the response waits for a frame before it sends a `: ka` comment.
KEEPALIVE_S = 5.0

# The channel from an owner to its response: frames, then `None` at its end.
type Channel = asyncio.Queue[BaseModel | None]


class StudioStream(StreamingResponse):
    """The response of a Studio stream: relays its owner's channel as SSE
    and, however its ASGI call ends, sets the client-gone flag and waits for
    the owner to finish (module docstring)."""

    def __init__(
        self,
        *,
        channel: Channel,
        owner: asyncio.Task[None],
        gone: asyncio.Event,
        keepalive_s: float = KEEPALIVE_S,
    ) -> None:
        super().__init__(self._serialised(), media_type=SSE_MEDIA_TYPE)
        self._channel = channel
        self._owner = owner
        self._gone = gone
        self._keepalive_s = keepalive_s

    async def _serialised(self) -> AsyncIterator[str]:
        while True:
            try:
                frame = await asyncio.wait_for(self._channel.get(), timeout=self._keepalive_s)
            except TimeoutError:
                yield ": ka\n\n"
                continue
            if frame is None:
                return
            yield f"data: {frame.model_dump_json(exclude_none=True)}\n\n"

    @override
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._gone.set()
            await asyncio.shield(self._owner)
