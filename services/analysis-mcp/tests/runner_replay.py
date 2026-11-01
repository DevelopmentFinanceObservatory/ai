"""dfo-apis' analysis runner, replayed from its recorded answers.

`fixtures/runner_replies.json` maps each request the registered tools send in
these tests to the answer dfo-apis' runner gave it: its status, its
`Retry-After` and its JSON body. A request is keyed by its path and the RFC
8785 form of its body, so a test passes only when the tools send the exact tree
the runner was recorded answering. A read of a frame the runner stored (`GET
/api/datasets/derived/...`) is keyed by its path and query, and its answer is
the manifest as JSON or the file as base64 `content`. Another suite replays
its own file through `replayed_runner(..., replies_path=...)`.

The answers are recorded from dfo-apis' own runner (`datasets_api.analysis.
run_analysis` and its door's refusal bodies) over the frames these tests seed,
under the pairs they seed them at. To record new ones, run the tests with
`RUNNER_RECORD=<path>`: each request with no recorded answer is written there
with the frames its roots name, as base64 Parquet under the root's pair and the
columns it is read with, and answers
409 `unrecorded`. Compute the answers in a dfo-apis checkout over that file, merge them
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


def read_key(path: str, query: str) -> str:
    """The key of a GET: its method, path and query."""
    return f"GET {path}?{query}"


def _replies(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _roots(body: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every dataset root a request names, its anchor roots included."""
    if "roots" in body:
        yield from body["roots"]
        return
    stack = [body["tree"], *body.get("anchor_roots", ())]
    while stack:
        node = stack.pop()
        if "verb" in node:
            stack.extend(node["inputs"].values())
        elif str(node.get("spec_id", "")).startswith("spec_"):
            yield node


async def _held_root(ref: FrameRef, columns: list[str] | None) -> Frame | None:
    """The frame the store holds for the root `ref` read with `columns`: under the
    frame pair of a mint record naming `ref` as its dataset with those columns, as
    `get_data` holds a frame of enrichment columns, else under `ref` itself. A root
    sends its columns sorted (`runner_tree.root_node`), so the record's are compared
    sorted. One dataset version read with and without columns is two frames."""
    store = get_store()
    for listing in await store.list_frames():
        record = await store.get_mint_record(listing.ref)
        if (
            record is not None
            and record.dataset == ref
            and sorted(set(record.options.get("columns") or [])) == (columns or [])
            and (held := await store.get(listing.ref)) is not None
        ):
            return held[0]
    held = await store.get(ref)
    return None if held is None else held[0]


async def _newest(spec_id: str) -> FrameRef | None:
    """The dataset version a Spec root reads: the dataset a mint record of `spec_id`
    names, else a frame the store holds under `spec_id`."""
    store = get_store()
    listed = [listing.ref for listing in await store.list_frames()]
    for ref in listed:
        record = await store.get_mint_record(ref)
        if record is not None and record.dataset.spec_id == spec_id:
            return record.dataset
    return next((ref for ref in listed if ref.spec_id == spec_id), None)


async def _record(key: str, entry: dict[str, Any]) -> None:
    """Write the unrecorded request `entry` under `key` to `RUNNER_RECORD`, with the
    frames its roots name, each under its pair and the columns it is read with. A
    Spec root's frame is recorded under the version `_newest` finds, which the
    answering runner reads as the Spec's newest."""
    frames: dict[str, str] = {}
    for root in _roots(entry["body"]) if "body" in entry else ():
        if "content_version" in root:
            ref = FrameRef(spec_id=root["spec_id"], content_version=root["content_version"])
        elif (newest := await _newest(root["spec_id"])) is not None:
            ref = newest
        else:
            continue
        columns = (root.get("rendering") or {}).get("columns")
        held = await _held_root(ref, columns)
        if held is not None:
            frames[f"{ref.spec_id} {ref.content_version} {json.dumps(columns or [])}"] = (
                base64.b64encode(frame_to_parquet_bytes(held)).decode()
            )
    # Each pytest-xdist worker writes a file of its own beside the named one.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    record = Path(os.environ["RUNNER_RECORD"] + (f".{worker}" if worker else ""))
    written = json.loads(record.read_text()) if record.exists() else {}
    written[key] = {**entry, "frames": frames}
    record.write_text(json.dumps(written, indent=1, sort_keys=True))


class ReplayedRunner:
    """A transport answering each request from the recorded replies at
    `replies_path`. `requests` keeps every POST body, in order, and `stored` the
    pair of each frame an answer named, by its recipe's replay key, so a fake of
    dfo-apis can hold what the runner stored for this test alone.

    An unrecorded request answers 409 with the problem code `unrecorded`, which
    the tools refuse at once and no caller retries."""

    def __init__(self, replies_path: Path = REPLIES_PATH) -> None:
        self.replies = _replies(replies_path)
        self.requests: list[dict[str, Any]] = []
        self.stored: dict[str, tuple[str, str]] = {}

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if request.method == "GET":
            query = request.url.query.decode()
            key = read_key(path, query)
            entry: dict[str, Any] = {"method": "GET", "path": path, "query": query}
        else:
            body = json.loads(request.content)
            self.requests.append(body)
            key = request_key(path, body)
            entry = {"path": path, "body": body}
        reply = self.replies.get(key)
        if reply is None:
            if os.environ.get("RUNNER_RECORD"):
                await _record(key, entry)
            return httpx2.Response(
                409, json={"code": "unrecorded", "retryable": False, "detail": key}
            )
        headers = {"retry-after": reply["retry_after"]} if reply.get("retry_after") else {}
        if path == runner_client.ANALYSIS_PATH and reply["status"] == httpx2.codes.OK:
            for member in reply["body"]["ref"]["members"]:
                pair = (member["spec"]["spec_id"], member["content_version"])
                self.stored[member["spec"]["recipe"]["replay_key"]] = pair
        if "content" in reply:
            return httpx2.Response(
                reply["status"], content=base64.b64decode(reply["content"]), headers=headers
            )
        return httpx2.Response(reply["status"], json=reply["body"], headers=headers)


async def _no_token() -> Mapping[str, str]:
    return {}


def replayed_runner(
    monkeypatch: pytest.MonkeyPatch, *, replies_path: Path = REPLIES_PATH
) -> ReplayedRunner:
    """Make every runner client the registered tools build answer from the
    replies recorded at `replies_path`, through one `ReplayedRunner`. A server's
    lifespan closes its client on shutdown, and the next call builds another the
    same way."""
    replayed = ReplayedRunner(replies_path)

    def client() -> RunnerClient:
        return RunnerClient(
            base_url=RUNNER_URL,
            timeout_s=5.0,
            read_timeout_s=5.0,
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
