"""Test-only wiring: the node swap, the translator's progress frame, and a
streaming in-process client.

- `use_pai_planner` puts `node.pai_planner_node` where `create_root_workflow`
  builds dev's `planner_node`, by replacing the name `harness.graph.workflow`
  imported. The seam `create_root_workflow` is given still reaches `switch`.
- `show_planner_progress` makes `_Translator` turn a `planner` node event's
  `custom_metadata["progress"]` into a frame. Dev's translator drops it (its
  comment: "the seam's late `progress` reports ... add no frame"). The frame is
  the running `activity.step` shape the translator already sends for `execute`'s
  step progress, on the planner's own status line ("Checking the plan against the
  data", which dev opens at the ADK planner's `propose_plan` call event), with the
  report as `detail`. Both S1 arms run with it, so the control differs only in
  the node.
- `StreamingASGITransport` hands the response body over as the app sends it, and
  sends `http.disconnect` when the client closes the stream. `httpx2.ASGITransport`
  (what `studio_client` uses) buffers the whole body before returning, so it can
  time no frame and cannot disconnect.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest
from harness.graph import workflow as workflow_module
from harness.studio_translate import _CHECKING, _node_name, _Translator
from harness.wire import STUDIO_FRAME_ADAPTER, ActivityStep, StepStatus
from node import Leak, pai_planner_node
from studio_planner import StudioDeps

AUTH = {"Authorization": "Bearer test-token"}


def use_pai_planner(
    monkeypatch: pytest.MonkeyPatch, deps: StudioDeps, *, leak: Leak | None = None
) -> None:
    """Build every root workflow with the Pydantic AI planner node."""

    def factory(*, planner: Any) -> Any:
        del planner
        return pai_planner_node(deps=deps, leak=leak)

    monkeypatch.setattr(workflow_module, "planner_node", factory)


def progress_detail(progress: dict[str, Any]) -> str:
    return f"check {progress['done']} of {progress['total']}"


def show_planner_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render the planner node's progress reports as running status frames."""
    original = _Translator._plan_path_frames

    def plan_path_frames(self: _Translator, event: Any) -> list[Any]:
        progress = (event.custom_metadata or {}).get("progress")
        if _node_name(event) != "planner" or not isinstance(progress, dict):
            return original(self, event)
        frames: list[Any] = self._open_status(_CHECKING)
        assert self._status is not None
        step_id, label, _ = self._status
        frames.append(
            ActivityStep(
                event="activity.step",
                seq=self._conversation.next_seq(),
                step_id=step_id,
                label=label,
                status=StepStatus.RUNNING,
                detail=progress_detail(progress),
            )
        )
        return frames

    monkeypatch.setattr(_Translator, "_plan_path_frames", plan_path_frames)


def is_progress(frame: dict[str, Any]) -> bool:
    return (
        frame["event"] == "activity.step"
        and frame["status"] == "running"
        and str(frame.get("detail") or "").startswith("check ")
    )


def events(frames: list[dict[str, Any]]) -> list[str]:
    return [f["event"] for f in frames]


def frame(frames: list[dict[str, Any]], event: str) -> dict[str, Any]:
    (found,) = [f for f in frames if f["event"] == event]
    return found


# ---------------------------------------------------------------------------
# Streaming transport
# ---------------------------------------------------------------------------


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
