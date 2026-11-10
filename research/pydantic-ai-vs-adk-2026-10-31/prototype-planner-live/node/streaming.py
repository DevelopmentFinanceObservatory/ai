"""A streaming in-process client for the Studio routes.

`httpx2.ASGITransport`, which `tests.orchestrator_fakes.studio_client` uses, buffers a whole
response before returning it, so it can time no frame and cannot disconnect. This transport
hands the body over as the app sends it, and sends `http.disconnect` when the client closes
the stream.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any

import httpx2
from harness.wire import STUDIO_FRAME_ADAPTER

AUTH = {"Authorization": "Bearer test-token"}


class _Body(httpx2.AsyncByteStream):
    def __init__(self, transport_call: "_Call") -> None:
        self._call = transport_call

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._call.chunks.get()
            if chunk is None:
                return
            yield chunk

    async def aclose(self) -> None:
        await self._call.close()


@dataclass
class _Call:
    app_task: asyncio.Task[None] | None = None
    chunks: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    started: asyncio.Future[tuple[int, list[Any]]] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    disconnected: asyncio.Event = field(default_factory=asyncio.Event)

    async def close(self) -> None:
        """The client is gone: tell the app, and wait for it to finish."""
        self.disconnected.set()
        assert self.app_task is not None
        await asyncio.wait_for(asyncio.shield(self.app_task), 10)


class StreamingASGITransport(httpx2.AsyncBaseTransport):
    def __init__(self, app: Any) -> None:
        self.app = app

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        assert isinstance(request.stream, httpx2.AsyncByteStream)
        body = b"".join([chunk async for chunk in request.stream])
        call = _Call()
        sent_body = False
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }

        async def receive() -> dict[str, Any]:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            await call.disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                call.started.set_result((message["status"], message.get("headers", [])))
            elif message["type"] == "http.response.body":
                if message.get("body"):
                    call.chunks.put_nowait(message["body"])
                if not message.get("more_body", False):
                    call.chunks.put_nowait(None)

        async def run() -> None:
            try:
                await self.app(scope, receive, send)
            finally:
                call.chunks.put_nowait(None)

        call.app_task = asyncio.ensure_future(run())
        status, headers = await call.started
        return httpx2.Response(status, headers=headers, stream=_Body(call))


def streaming_client(app: Any) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=StreamingASGITransport(app), base_url="http://studio.test", timeout=30
    )


@dataclass
class Timed:
    """Frames with the seconds since the request started at which each arrived."""

    frames: list[dict[str, Any]] = field(default_factory=list)
    at: list[float] = field(default_factory=list)


async def read_timed(
    response: httpx2.Response, timed: Timed, started: float, *, on_frame: Any = None
) -> None:
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        parsed = json.loads(line.removeprefix("data: "))
        STUDIO_FRAME_ADAPTER.validate_python(parsed)
        timed.frames.append(parsed)
        timed.at.append(time.monotonic() - started)
        if on_frame is not None:
            on_frame(parsed)
