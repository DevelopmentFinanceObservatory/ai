"""The node API of the turn loop (`harness.turn.loop`).

A node is a plain async function of a `NodeCtx` and its input that returns
where the turn goes next: `Next` (an edge label and the next node's input)
or `Pause` (a question for the user). A node that pauses also has a
`resume` function, which the loop calls with the user's answer in place of
`run`, so the work a node does before its pause runs once. An error is a
raised exception. The loop records it on the turn's cursor and ends the
turn.

A node's events have a type of the graph's own, `E`, which the loop passes
to the caller's `emit` beside its own `TurnEvent`s; the loop reads none of
them.

A node writes to the store only through the loop. It reads and changes
`ctx.state` and `ctx.transcript`, and the loop writes both with the cursor
at the node's boundary. Changes since the last boundary die with the
process, and the node then runs again from its start.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from harness.turn.events import TurnEvent
from harness.turn.transcript import Transcript


@dataclass(frozen=True)
class TurnKey:
    """The conversation a turn belongs to, and the turn's number in it."""

    principal: str
    conversation_id: str
    turn_index: int


@dataclass(frozen=True)
class Next:
    """Go on along the edge `label`, handing `output` to the node it reaches."""

    label: str
    output: Any = None


@dataclass(frozen=True)
class Pause:
    """Pause the turn on a question of `kind` to the user. The loop mints the
    wire interrupt id the client answers, and stores `payload` and
    `message` with the pause."""

    kind: Literal["ask", "planner_ask", "approve"]
    payload: dict[str, Any]
    message: str = ""


class StopScope:
    """The work a node runs under a Stop that ends only that work (a
    plan-card switch, a pick's replan).

    While a node holds the scope open, the first Stop sets the event `open`
    yielded and the scope keeps that Stop's id (`taken`). The node goes on
    along its own edge, and the turn's next pause write records the id on
    the turn's cursor, which ends that Stop. Any Stop while no scope is
    open, or once the open scope's event is set, ends the turn. The loop
    marks it (`mark_turn_stopped`) before it cancels the node, so work the
    cancellation reaches tells a Stop (`turn_stopped`) from a client that
    left or a lost lease, which leave the turn for a later segment to carry
    on."""

    def __init__(self) -> None:
        self._event: asyncio.Event | None = None
        self._turn_stopped = False
        self.taken: str | None = None

    @contextlib.asynccontextmanager
    async def open(self) -> AsyncIterator[asyncio.Event]:
        """Hold the scope open for the block; the event is set on a Stop."""
        self._event = asyncio.Event()
        try:
            yield self._event
        finally:
            self._event = None

    @property
    def armed(self) -> bool:
        """Whether a Stop would end only the open scope's work."""
        return self._event is not None and not self._event.is_set()

    def take(self, stop_id: str) -> bool:
        """Stop the open scope's work on the Stop `stop_id`, keeping its id
        (`taken`). Returns whether the scope was armed. When it was not, the
        Stop is the turn's and nothing changes."""
        if self._event is None or self._event.is_set():
            return False
        self._event.set()
        self.taken = stop_id
        return True

    def mark_turn_stopped(self) -> None:
        """Record that a Stop ends the turn. The loop then cancels the node."""
        self._turn_stopped = True

    @property
    def turn_stopped(self) -> bool:
        """Whether a Stop ended the turn. A node's cancellation with this
        unset comes from a client that left or a lost lease."""
        return self._turn_stopped


@dataclass
class NodeCtx[E]:
    """What a node reads and writes.

    `inputs` are the turn's inputs, fixed when it opened. `state` is the
    turn-scoped state the nodes share, the turn's spec blocks included
    (`harness.turn.store.SPECS_KEY`). It is a copy of the state as stored,
    which the node changes as it likes. `transcript` holds the agents'
    records. The loop writes both with the cursor at the node's boundary.
    `emit` streams an event to the client at once, either one of the
    graph's own (`E`) or a `TurnEvent`.
    `stop` is the turn's `StopScope`. `after_commit` registers work that must
    follow the boundary's write, such as acknowledging what a hand-back
    delivered. `on_cut_close` registers what undoes that work when a cut
    closes the turn after the node's pause committed, such as cancelling
    the retrievals a plan card's offer started.
    """

    key: TurnKey
    inputs: dict[str, Any]
    state: dict[str, Any]
    transcript: Transcript
    emit: Callable[[E | TurnEvent], None]
    stop: StopScope
    committed: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    cut_closes: list[Callable[[], None]] = field(default_factory=list)

    def after_commit(self, work: Callable[[], Awaitable[None]]) -> None:
        """Run `work` once the write at this node's boundary commits."""
        self.committed.append(work)

    def on_cut_close(self, undo: Callable[[], None]) -> None:
        """Run `undo` when a cut (a Stop) closes the turn after this node's
        pause committed (`harness.turn.loop.run_segment`)."""
        self.cut_closes.append(undo)


type Run[E] = Callable[[NodeCtx[E], Any], Awaitable[Next | Pause]]
type Resume[E] = Callable[[NodeCtx[E], Any, dict[str, Any]], Awaitable[Next | Pause]]


@dataclass(frozen=True)
class Node[E]:
    """One node: `run` on entry with the node's input, and, for a node that
    pauses, `resume` with the same input and the user's answer.

    A node that holds the turn's `StopScope` open names in `stop_edges` the
    edges it may leave on once its scope took a Stop. Each must lead to a
    node that pauses (`Graph` checks), whose pause records the Stop taken;
    leaving any other way closes the turn `cancelled`
    (`harness.turn.loop.drive`)."""

    run: Run[E]
    resume: Resume[E] | None = None
    stop_edges: frozenset[str] = frozenset()
