"""The events a turn emits as it runs, which the Studio translator turns into
wire frames.

Each is a frozen dataclass and carries what the translator reads and nothing
of how it was produced. `TurnEvent` is their union.
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
    """The turn ended on `error` at `node`, and its cursor records it: a node
    raised (closed `failed`), or the cursor named a node the running graph
    cannot run (closed `abandoned`, `error` a `harness.turn.loop.TurnExpired`)."""

    node: str
    error: BaseException


type TurnEvent = NodeDone | Paused | TurnFailed
