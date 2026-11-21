"""analysis-mcp's in-process verbs, replayed from what they stored.

Before analysis-mcp computed every verb on dfo-apis' analysis runner, its tools
computed in process and stored each result in the materialisation store: the frame
under its `ops_` pair with its trace, and `resolve_anchor`'s anchor record. A store
still holds frames from then, and `harness.recipe_backfill`, the store branch of
`harness.recipe_replay` and `harness.rolling_cards` read them. No runner call
writes such a frame, so these tests put them in the store from
`fixtures/stored_verbs.json`: for each call they make, what the verb answered and
what it stored.

A call is keyed by its verb, the RFC 8785 form of its arguments, and what the store
holds under each frame it names: the frame's Parquet bytes and its trace. A replayed
call writes what the verb stored and returns its answer. The answers were recorded
from the in-process verbs of analysis-mcp before their bodies were removed, by
running these tests with `STORED_VERBS_RECORD=<path>` in a checkout that still
holds them; a call this fixture lacks fails the test.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import rfc8785
from analysis_mcp.trace import Trace
from common.materialisation import AnchorRecord, FrameRef, get_store
from common.materialisation.frame_codec import (
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
)
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import BaseModel

REPLIES_PATH = Path(__file__).resolve().parent / "fixtures" / "stored_verbs.json"

# The in-process function of each verb, by module, as analysis-mcp held them.
_BODIES = {
    "aggregate": ("analysis_mcp.aggregate_tool", "aggregate"),
    "filter": ("analysis_mcp.filter_tool", "filter_verb"),
    "growth": ("analysis_mcp.growth_tool", "growth"),
    "join": ("analysis_mcp.join_tool", "join"),
    "resolve_anchor": ("analysis_mcp.rolling_tool", "resolve_anchor"),
}
_FRAME_ARGUMENTS = ("frame_ref", "right_frame_ref", "denominator_frame_ref")


def _json(value: object) -> object:
    """`value` as JSON: a model as its JSON dump, a dataclass trace as its fields."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json(item) for item in value]
    return value


def _named_frames(verb: str, arguments: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if verb == "resolve_anchor":
        return list(arguments.get("roots") or ())
    return [
        arguments[name] for name in _FRAME_ARGUMENTS if isinstance(arguments.get(name), Mapping)
    ]


async def _held(ref: Mapping[str, Any]) -> object:
    """What the store holds under `ref`: its frame's Parquet digest and its trace."""
    pair = FrameRef(spec_id=ref["spec_id"], content_version=ref["content_version"])
    held = await get_store().get(pair)
    if held is None:
        return None
    frame, trace = held
    return {
        "frame": hashlib.sha256(frame_to_parquet_bytes(frame)).hexdigest(),
        "trace": _json(trace),
    }


async def _key(verb: str, arguments: Mapping[str, Any]) -> str:
    held = [await _held(ref) for ref in _named_frames(verb, arguments)]
    body = {"arguments": _json(arguments), "held": held}
    return f"{verb} {rfc8785.dumps(body).decode()}"


def _text(result: ToolResult) -> str:
    return "".join(part.text for part in result.content if isinstance(part, TextContent))


async def _stored(verb: str, result: ToolResult) -> dict[str, Any]:
    """What `result`'s verb stored: its frame and trace, or its anchor record."""
    content = result.structured_content or {}
    if result.is_error:
        return {}
    store = get_store()
    if verb == "resolve_anchor":
        record = await store.get_anchor_record(content["anchor_key"])
        assert record is not None
        return {"anchor_record": record.model_dump(mode="json")}
    ref = FrameRef(spec_id=content["spec_id"], content_version=content["content_version"])
    held = await store.get(ref)
    assert held is not None
    frame, trace = held
    return {
        "frame": base64.b64encode(frame_to_parquet_bytes(frame)).decode(),
        "trace": _json(trace),
    }


async def _record(key: str, verb: str, arguments: Mapping[str, Any]) -> ToolResult:
    """Run the in-process verb and write its answer and what it stored under `key`
    to `STORED_VERBS_RECORD`, one file per pytest-xdist worker."""
    module, name = _BODIES[verb]
    result = await getattr(importlib.import_module(module), name)(**arguments)
    entry = {
        "text": _text(result),
        "structured_content": result.structured_content,
        "is_error": bool(result.is_error),
        **await _stored(verb, result),
    }
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    path = Path(os.environ["STORED_VERBS_RECORD"] + (f".{worker}" if worker else ""))
    written = json.loads(path.read_text()) if path.exists() else {}
    written[key] = entry
    path.write_text(json.dumps(written, indent=1, sort_keys=True))
    return result


async def _replayed(verb: str, entry: Mapping[str, Any]) -> ToolResult:
    store = get_store()
    content = entry["structured_content"]
    if "anchor_record" in entry:
        await store.put_anchor_record(AnchorRecord.model_validate(entry["anchor_record"]))
    elif "frame" in entry:
        ref = FrameRef(spec_id=content["spec_id"], content_version=content["content_version"])
        frame = frame_from_verified_parquet_bytes(base64.b64decode(entry["frame"]))
        await store.put(ref, frame=frame, trace=Trace.model_validate(entry["trace"]), writer=verb)
    return ToolResult(
        content=[TextContent(type="text", text=entry["text"])],
        structured_content=content,
        is_error=entry["is_error"],
    )


@functools.cache
def _replies() -> Mapping[str, Any]:
    return json.loads(REPLIES_PATH.read_text()) if REPLIES_PATH.exists() else {}


async def call(verb: str, **arguments: Any) -> ToolResult:
    """The answer `verb` gave `arguments` over what the store holds, with what it
    stored written to the store."""
    key = await _key(verb, arguments)
    if os.environ.get("STORED_VERBS_RECORD"):
        return await _record(key, verb, arguments)
    replies = _replies()
    if key not in replies:
        pytest.fail(f"no stored verb answer is recorded for {key}")
    return await _replayed(verb, replies[key])


async def aggregate(**arguments: Any) -> ToolResult:
    return await call("aggregate", **arguments)


async def filter_verb(**arguments: Any) -> ToolResult:
    return await call("filter", **arguments)


async def growth(**arguments: Any) -> ToolResult:
    return await call("growth", **arguments)


async def join(**arguments: Any) -> ToolResult:
    return await call("join", **arguments)


async def resolve_anchor(**arguments: Any) -> ToolResult:
    return await call("resolve_anchor", **arguments)
