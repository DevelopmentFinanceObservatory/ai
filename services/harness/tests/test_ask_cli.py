"""Regression coverage for `scripts/ask.py`'s env-var defaults and its
segment reader.

`scripts/ask.py` sits outside the uv workspace (`pyproject.toml`'s
`[tool.uv.workspace]` lists only `packages/*` and `services/*`), so it can't
be imported as a package. It's loaded by file path instead, the same way
`test_studio_orchestrator_eval.py` loads `evals/studio/checks.py`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import httpx2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ASK_PATH = REPO_ROOT / "scripts" / "ask.py"
WIRE_FIXTURES = REPO_ROOT / "services" / "harness" / "tests" / "wire_fixtures" / "studio"


def _load_ask() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ask", ASK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ask", module)
    spec.loader.exec_module(module)
    return module


ask = _load_ask()


def test_blank_studio_api_token_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`just env` carries `STUDIO_API_TOKEN=` from `.env.example` into `.env`
    verbatim, even blank, so the key is always present -- never absent. A
    present-but-empty value must fall back to `_DEFAULT_TOKEN` the same way
    an absent one does; otherwise `--token` becomes `""` and the CLI builds
    the literal header `Bearer `, which httpx2 refuses to send.
    """
    monkeypatch.setenv("STUDIO_API_TOKEN", "")
    args = ask.parse_args(["question"])
    assert args.token == ask._DEFAULT_TOKEN
    assert args.token != ""


def test_unset_studio_api_token_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_API_TOKEN", raising=False)
    args = ask.parse_args(["question"])
    assert args.token == ask._DEFAULT_TOKEN


def test_real_studio_api_token_is_used_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_API_TOKEN", "a-real-token")
    args = ask.parse_args(["question"])
    assert args.token == "a-real-token"


def test_blank_studio_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDIO_URL", "")
    args = ask.parse_args(["question"])
    assert args.url == ask._DEFAULT_URL


def test_unset_studio_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_URL", raising=False)
    args = ask.parse_args(["question"])
    assert args.url == ask._DEFAULT_URL


@pytest.mark.parametrize(
    ("fixture", "event"),
    [
        ("event-03-plan-proposed.json", "plan.proposed"),
        ("event-09-elicitation-request.json", "elicitation.request"),
        ("event-12-done.json", "done"),
    ],
)
async def test_stream_segment_reads_to_the_end_before_returning_its_ending_frame(
    fixture: str, event: str
) -> None:
    """The server releases the conversation after the pausing frame, and its
    stream ends once it has; `ask` posts `/interrupt` as soon as
    `stream_segment` returns, so the stream must already be at its end."""
    frame = json.loads((WIRE_FIXTURES / fixture).read_text())
    drained = False

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal drained
            yield f"data: {json.dumps(frame)}\n\n".encode()
            yield b": ka\n\n"
            drained = True

    transport = httpx2.MockTransport(
        lambda request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Body()
        )
    )
    async with httpx2.AsyncClient(transport=transport, base_url="http://studio.test") as client:
        ending = await ask.stream_segment(
            client,
            "/api/studio/turn",
            {},
            ask.Console(tty=False, color=False),
            ask.Waiter(start=0.0),
        )

    assert drained
    assert ending is not None
    assert ending.event == event
