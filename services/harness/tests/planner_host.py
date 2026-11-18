"""Runs `harness.planner_adapter.PlannerAdapter` as the `planner` node hosts it, for tests.

`PlannerHarness` stands for one turn: its state, its transcript and the pause
the planner's ask leaves open. Each `run` is one segment of the turn's
`planner` node, in memory: it opens on the user's message (a new turn message
unless the run answers a pause), calls the adapter on a `NodeCtx` over the
turn's state and transcript, pauses on an ask as the node does
(`harness.graph.planning.planner_ask_pause`), and then keeps the state as the
turn loop stores it, through JSON, and the transcript records the run
appended. The planner's tools read those records as they would in a turn
(`planner.session.SessionCall`).

`node_ctx` and `in_node` give a test that calls the adapter directly the
context a node runs on.

`RefusingPlanner` and `FailingPlanner` are `PlannerModel`s with no model: one
refuses to draft, as `PlannerAdapter` does at its round cap, and one raises
`PlannerAdapterError`, as `PlannerAdapter` does when it cannot finish an entry.

The record readers (`planner_calls`, `planner_answers`, `pauses`) read a
transcript as the planner's own readers do.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from harness.graph import switch
from harness.graph.planning import planner_ask_pause
from harness.planner_run import PlannerAdapterError
from harness.studio_state import _wire_id
from harness.turn import transcript
from harness.turn.node import NodeCtx, StopScope, TurnKey
from harness.turn.store import AgentRecord
from harness.turn.transcript import Transcript
from planner.plan import (
    AskBreakdown,
    SessionState,
    load_ask_breakdowns,
    load_asks,
    load_asks_across_years,
)
from planner.session import ANSWERS, RESPONSE

KEY = TurnKey("user-1", "conv_01test0000", 1)
# The message a run opens the turn on when the test names none.
PROBE = "probe"


@dataclass(frozen=True)
class OpenPause:
    """The pause a run left open: its wire id, round and payload."""

    interrupt_id: str
    round: int
    payload: dict[str, Any]


def node_ctx(
    *,
    state: dict[str, Any] | None = None,
    message: str = PROBE,
    records: Sequence[AgentRecord] | None = None,
) -> NodeCtx[Any]:
    """A `NodeCtx` of a turn opened on `message`, over `state` itself, for a
    test that calls the adapter directly as a node would. Its transcript
    holds `records`, or the message alone."""
    return NodeCtx[Any](
        key=KEY,
        inputs={"message": message},
        state={} if state is None else state,
        specs={},
        transcript=Transcript(
            list(records) if records is not None else [transcript.user_message(message)]
        ),
        emit=lambda event: None,
        stop=StopScope(),
    )


async def in_node(
    body: Callable[[NodeCtx[Any]], Awaitable[Any]],
    *,
    state: dict[str, Any] | None = None,
    message: str = PROBE,
) -> NodeCtx[Any]:
    """Run `body` on a `node_ctx` and return the context it ran on."""
    ctx = node_ctx(state=state, message=message)
    await body(ctx)
    return ctx


class PlannerHarness:
    """One turn's `planner` node, run in memory, segment by segment.

    `state` is the turn's opening state. Every run keeps what the adapter left
    in the state, as JSON holds it, and the records it appended, so a later
    run reads them as a later segment of the turn would.
    """

    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = _as_stored(dict(state or {}))
        self._records: list[AgentRecord] = []
        self._message = PROBE
        self.pause: OpenPause | None = None
        self.emitted: list[Any] = []
        self._live_state: dict[str, Any] = {}
        self._rounds = 0

    async def run(
        self,
        adapter: Any,
        *,
        answer: dict[str, Any] | None = None,
        message: str | None = None,
        stoppable: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Run `adapter` once, as the `planner` node calls it, and return what it
        answered, or `None` when it paused on the planner's ask.

        With `answer`, the run resumes the turn by the user's answer: to the
        open pause, which the transcript records, or, with none open, the run
        passes it to the adapter alone, as a resume whose pause a test does
        not stage. Without it, the turn's transcript gains the user's
        `message` (`PROBE` when None), which the run's question is. With
        `stoppable` the adapter runs under the switch's `_unless_stopped`, as
        the node runs a pick's replan.
        """
        records = Transcript(self._records)
        if answer is None:
            self._message = message if message is not None else PROBE
            records.append(transcript.user_message(self._message))
        elif self.pause is not None:
            records.append(
                transcript.answer(
                    interrupt_id=self.pause.interrupt_id, round=self.pause.round, value=answer
                )
            )
        self.pause = None
        # The node works on a copy: a run that raises or is cancelled
        # leaves the turn's state as the last boundary stored it.
        ctx = NodeCtx[Any](
            key=KEY,
            inputs={"message": self._message},
            state=_as_stored(self._state),
            specs={},
            transcript=records,
            emit=self.emitted.append,
            stop=StopScope(),
        )
        self._live_state = ctx.state
        drafting = adapter(ctx=ctx, answer=answer, **kwargs)
        drafted = await (switch._unless_stopped(ctx, drafting) if stoppable else drafting)
        output: dict[str, Any] | None = drafted
        if isinstance(drafted, dict) and "asked" in drafted:
            paused = planner_ask_pause(drafted["asked"])
            self._rounds += 1
            self.pause = OpenPause(_wire_id("int"), self._rounds, paused.payload)
            records.append(
                transcript.pause(
                    kind=paused.kind,
                    interrupt_id=self.pause.interrupt_id,
                    round=self.pause.round,
                    payload=paused.payload,
                    message=paused.message,
                )
            )
            output = None
        self._records = list(records.records)
        self._state = _as_stored(ctx.state)
        return output

    async def state(self) -> SessionState:
        """The turn's state as the last run left it."""
        return self._state

    async def records(self) -> list[AgentRecord]:
        """The turn's transcript, in order."""
        return list(self._records)

    def restore(self, records: Sequence[AgentRecord], state: Mapping[str, Any]) -> None:
        """Make `records`, on `state`, this turn's, for a test that resumes a
        turn another build stored."""
        self._records = list(records)
        self._state = _as_stored(dict(state))

    def live_state(self) -> Mapping[str, Any]:
        """The state the last run read and wrote in process."""
        return self._live_state

    def seed_state(self, key: str, value: Any) -> None:
        """Set `key` in the turn's state, for a test that needs state no real
        call would produce."""
        self._state[key] = value


def _as_stored(state: dict[str, Any]) -> dict[str, Any]:
    """`state` as the turn's cursor stores it: through JSON."""
    return json.loads(json.dumps(state))


def planner_calls(records: Sequence[AgentRecord]) -> list[dict[str, Any]]:
    """Every function call of the planner's response records, in order, as
    stored (`id`, `name`, `args`)."""
    return [
        part["function_call"]
        for record in records
        if record.agent == "planner" and record.kind == RESPONSE
        for part in record.body.get("parts") or ()
        if "function_call" in part
    ]


def planner_answers(records: Sequence[AgentRecord]) -> list[dict[str, Any]]:
    """Every function response of the planner's answers records, in order,
    as stored (`id`, `name`, `response`)."""
    return [
        part["function_response"]
        for record in records
        if record.agent == "planner" and record.kind == ANSWERS
        for part in record.body.get("parts") or ()
        if "function_response" in part
    ]


def pauses(records: Sequence[AgentRecord]) -> list[dict[str, Any]]:
    """The body of every pause record, in order."""
    return [
        record.body
        for record in records
        if record.agent == "loop" and record.kind == transcript.PAUSE
    ]


class FailingPlanner:
    """A planner that raises `PlannerAdapterError`, the way `PlannerAdapter` fails."""

    def __init__(
        self,
        message: str = (
            "a plan that is draft was declared ready, not pending_approval; "
            "refusing to report it ready"
        ),
    ) -> None:
        self.message = message

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise PlannerAdapterError(self.message)


class RefusingPlanner:
    """A planner that refuses to draft, as `PlannerAdapter` does at its round cap.

    It records each entry's question and readings, and the asks, their
    breakdowns and the asks across years the entry's state holds."""

    def __init__(self, *, final: bool = False) -> None:
        self.questions: list[str] = []
        self.readings: list[Any] = []
        self.asks: list[tuple[str, ...]] = []
        self.breakdowns: list[tuple[AskBreakdown, ...]] = []
        self.across_years: list[tuple[int, ...]] = []
        self.final = final

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        state = kwargs["ctx"].state
        self.questions.append(kwargs["question"])
        self.readings.append(kwargs.get("readings"))
        self.asks.append(load_asks(state))
        self.breakdowns.append(load_ask_breakdowns(state))
        self.across_years.append(load_asks_across_years(state))
        return {"refused": {"text": "the planner ran out of rounds", "final": self.final}}
