"""dfo-apis' analysis runner, replayed from its recorded answers.

`fixtures/runner_replies.json` maps each request the registered tools send in
these tests to the answer dfo-apis' runner gave it: its status, its
`Retry-After` and its JSON body. A request is keyed by its path and the RFC
8785 form of its body, so a test passes only when the tools send the exact tree
the runner was recorded answering.

The answers are recorded from dfo-apis' own runner (`datasets_api.analysis.
run_analysis` and its door's refusal bodies) over the frames these tests seed,
under the pairs they seed them at. To record new ones, run the tests with
`RUNNER_RECORD=<path>`: each request with no recorded answer is written there
with the frames its roots name, as base64 Parquet, and answers
501. Compute the answers in a dfo-apis checkout over that file, merge them
into `fixtures/runner_replies.json`, and run again until no request is
written.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx2
import pytest
import rfc8785
from analysis_mcp import runner_client
from analysis_mcp.runner_client import RunnerClient, runner_holder
from common.materialisation import FrameRef, get_store
from common.materialisation.frame_codec import frame_to_parquet_bytes
from dfo_frame_grammar import Frame, canonical_frame_bytes

REPLIES_PATH = Path(__file__).resolve().parent / "fixtures" / "runner_replies.json"
RUNNER_URL = "http://runner.test"


def request_key(path: str, body: object) -> str:
    """The key of a request: its path and the RFC 8785 form of its body."""
    return f"{path} {rfc8785.dumps(body).decode()}"


def _replies() -> dict[str, Any]:
    if not REPLIES_PATH.exists():
        return {}
    return json.loads(REPLIES_PATH.read_text())


def _roots(body: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every dataset root a request names."""
    if "roots" in body:
        yield from body["roots"]
        return
    stack = [body["tree"]]
    while stack:
        node = stack.pop()
        if "verb" in node:
            stack.extend(node["inputs"].values())
        elif str(node.get("spec_id", "")).startswith("spec_"):
            yield node


async def _record(path: str, body: Mapping[str, Any], key: str) -> None:
    frames: dict[str, str] = {}
    for root in _roots(body):
        ref = FrameRef(spec_id=root["spec_id"], content_version=root["content_version"])
        held = await get_store().get(ref)
        if held is not None:
            frames[f"{ref.spec_id} {ref.content_version}"] = base64.b64encode(
                frame_to_parquet_bytes(held[0])
            ).decode()
    record = Path(os.environ["RUNNER_RECORD"])
    written = json.loads(record.read_text()) if record.exists() else {}
    written[key] = {"path": path, "body": body, "frames": frames}
    record.write_text(json.dumps(written, indent=1, sort_keys=True))


class ReplayedRunner:
    """A transport answering each request from `fixtures/runner_replies.json`.
    `requests` keeps every request body, in order."""

    def __init__(self) -> None:
        self.replies = _replies()
        self.requests: list[dict[str, Any]] = []

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        key = request_key(request.url.path, body)
        reply = self.replies.get(key)
        if reply is None:
            if os.environ.get("RUNNER_RECORD"):
                await _record(request.url.path, body, key)
            return httpx2.Response(501, json={"unrecorded": key})
        headers = {"retry-after": reply["retry_after"]} if reply.get("retry_after") else {}
        return httpx2.Response(reply["status"], json=reply["body"], headers=headers)


async def _no_token() -> Mapping[str, str]:
    return {}


def replayed_runner(monkeypatch: pytest.MonkeyPatch) -> ReplayedRunner:
    """Make every runner client the registered tools build answer from the
    recorded replies, through one `ReplayedRunner`. A server's lifespan closes
    its client on shutdown, and the next call builds another the same way."""
    replayed = ReplayedRunner()

    def client() -> RunnerClient:
        return RunnerClient(
            base_url=RUNNER_URL,
            timeout_s=5.0,
            caller_principal=None,
            auth=_no_token,
            transport=httpx2.MockTransport(replayed.handle),
        )

    monkeypatch.setattr(runner_client, "_default_client", client)
    runner_holder.set(None)
    return replayed


def frame_digest(frame: Frame) -> str:
    """32 hex digits naming `frame` by its canonical bytes: a seeded pair that is
    the same on every run, so its request matches its recorded reply."""
    return hashlib.sha256(canonical_frame_bytes(frame)).hexdigest()[:32]
