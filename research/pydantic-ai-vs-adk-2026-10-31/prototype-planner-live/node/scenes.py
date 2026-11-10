"""Helpers the node's tests share: a Studio turn's frames, driven on either arm, and what
differs between two runs of the same script."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from tests.orchestrator_fakes import CONVERSATION

TURN = "/api/studio/turn"
INTERRUPT = "/api/studio/interrupt"

# What differs between any two runs of a turn: sequence numbers, ids minted per run, times.
VOLATILE = {
    "seq",
    "step_id",
    "interrupt_id",
    "duration_ms",
    "observed_at",
    "run_id",
    "plan_version_id",
    "approved_at",
    "at",
    "trace_id",
    "supersedes",
}


def stable(value: Any) -> Any:
    """`value` without the fields that differ between runs. JSON held as a string is read."""
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    if isinstance(value, str) and value.startswith("{"):
        try:
            return stable(json.loads(value))
        except ValueError:
            return value
    return value


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """The paths at which `a` and `b` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(set(a) | set(b), key=str)
            for d in differences(a.get(key), b.get(key), f"{path}.{key}")
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def events(frames: Sequence[dict[str, Any]]) -> list[str]:
    return [f["event"] for f in frames]


def frame(frames: Sequence[dict[str, Any]], event: str) -> dict[str, Any]:
    (found,) = [f for f in frames if f["event"] == event]
    return found


def interrupt_body(interrupt_id: str, decision: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "interrupt_id": interrupt_id,
        "decision": decision,
        **extra,
    }
