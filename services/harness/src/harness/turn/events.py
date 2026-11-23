"""The events the turn loop itself emits as a turn runs, which the Studio
translator turns into wire frames beside the events of the graph's own
nodes (`harness.turn.node.NodeCtx.emit`).

Each is a frozen dataclass carrying what the translator reads. `TurnEvent`
is their union.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NodeDone:
    """Node `node` finished on edge `label`, handing `output` to the next
    node. Emitted once the cursor that moved past it committed."""

    node: str
    label: str
    output: Any = None


@dataclass(frozen=True)
class Paused:
    """The turn paused on a question of `kind` to the user under the wire id
    `interrupt_id`, as its pause `round`. Emitted once the cursor that holds
    the pause committed."""

    kind: str
    interrupt_id: str
    round: int
    payload: dict[str, Any]
    message: str = ""


@dataclass(frozen=True)
class TurnFailed:
    """The turn ended on `error` at `node`: a node raised (closed `failed`),
    or the cursor named a node the running graph cannot run (closed
    `abandoned`, `error` a `harness.turn.loop.TurnExpired`). The cursor
    records it unless the close could not be written, which leaves the turn
    as last written (`harness.turn.loop.Outcome`).

    `error` is the exception itself, so the Studio error frame reads what it
    states (`harness.studio_translate._stream_fatal`): its `refusal`, the MCP
    call whose pinned timeout it hit, and its cause chain. The cursor's
    `error` records the same facts for a reader with no exception at hand
    (`harness.turn.loop.failure`)."""

    node: str
    error: Exception


type TurnEvent = NodeDone | Paused | TurnFailed
