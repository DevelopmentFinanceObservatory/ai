"""What analysis-mcp's in-process verbs stored, put back in the store for tests.

`fixtures/stored_verbs.json` holds 76 calls of analysis-mcp's in-process `aggregate`,
`filter`, `growth`, `join` and `resolve_anchor`, recorded from the in-process tools at
dfo-ai commit ff05fe91. Each entry is the call's answer (its text,
`structured_content` and error flag) and what the verb stored: its frame as base64
Parquet with its trace, or `resolve_anchor`'s anchor record. A store can hold
such frames, and `harness.recipe_backfill`, the store branch of `harness.recipe_replay`
and `harness.rolling_cards` read them.

`EXECUTION_IDENTITY` is the engine identity every stored trace carries.

A call is keyed by its verb, the RFC 8785 form of its arguments, and what the store
holds under each frame it names: the frame's `frame_sha256` and its trace without its
`execution_identity`. A replayed call writes what the verb stored and returns its
answer. A call the fixture lacks fails the test.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import rfc8785
from analysis_mcp.execution_identity import ExecutionIdentity
from analysis_mcp.trace import Trace
from common.materialisation import AnchorRecord, FrameRef, get_store
from common.materialisation.frame_codec import frame_from_verified_parquet_bytes
from dfo_frame_grammar import frame_sha256
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import BaseModel

REPLIES_PATH = Path(__file__).resolve().parent / "fixtures" / "stored_verbs.json"

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
    """What the store holds under `ref`: its frame's `frame_sha256` and its trace
    without its `execution_identity`."""
    pair = FrameRef(spec_id=ref["spec_id"], content_version=ref["content_version"])
    held = await get_store().get(pair)
    if held is None:
        return None
    frame, trace = held
    shown = _json(trace)
    assert isinstance(shown, dict)
    shown.pop("execution_identity", None)
    return {"frame": frame_sha256(frame), "trace": shown}


async def _key(verb: str, arguments: Mapping[str, Any]) -> str:
    held = [await _held(ref) for ref in _named_frames(verb, arguments)]
    body = {"arguments": _json(arguments), "held": held}
    return f"{verb} {rfc8785.dumps(body).decode()}"


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
    return json.loads(REPLIES_PATH.read_text())


def _recorded_identity() -> ExecutionIdentity:
    (identity,) = {
        json.dumps(entry["trace"]["execution_identity"], sort_keys=True)
        for entry in _replies().values()
        if "trace" in entry
    }
    return ExecutionIdentity(**json.loads(identity))


EXECUTION_IDENTITY = _recorded_identity()


async def call(verb: str, **arguments: Any) -> ToolResult:
    """The answer `verb` gave `arguments` over what the store holds, with what it
    stored written to the store."""
    key = await _key(verb, arguments)
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
