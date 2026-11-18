"""Runs Studio turns through the turn loop (`harness.turn`) for tests.

`TurnDriver` stands for one harness instance on a test database: it claims
the conversation's lease on a real `StudioStore`, as the routes do, opens or
resumes the turn on a real `TurnStore`, runs one segment
(`harness.turn.loop.run_segment`) over the `Graph` it is given, releases the
lease, and returns the segment with every event it emitted (`SegmentRun`).
Two drivers on one engine are two instances, so a test can pause a turn on
one and answer it on the other.

`ScriptedModels` holds one scripted google-genai client per agent
(`tests.scripted_genai.ScriptedGenai`), so each agent runs its real Pydantic
AI code over `harness.gemini_wire.GeminiStream`: `orchestrator_model` and
`planner_model` build the agents' models on the scripted clients, and each
client records the requests it was sent. `numbered_call_ids` makes the ids
given calls the model sent without one stable across runs, and
`request_text` reads a recorded request back as text.

The orchestrator's tools over a scripted surface are in
`tests.scripted_tools`.
"""

from __future__ import annotations

import contextlib
import itertools
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

from harness import gemini_wire
from harness.studio_store import StudioStore
from harness.turn import transcript as entries
from harness.turn.events import Paused, TurnEvent
from harness.turn.loop import Answer, Graph, Outcome, run_segment
from harness.turn.store import AgentRecord, Cursor, TurnStore
from harness.turn_budget import TurnBudget
from orchestrator.model import orchestrator_model
from planner.model import planner_model
from pydantic_ai.models.google import GoogleModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .scripted_genai import Request, ScriptedGenai, Step

PRINCIPAL = "user-1"
CONVERSATION = "conv_01test0000"
MODEL = "gemini-test"
# Short enough that a Stop reaches a running node within a test's patience.
HEARTBEAT_S = 0.02


@dataclass
class ScriptedModels:
    """One scripted google-genai client per agent, each answering its
    agent's calls from its own script in order."""

    orchestrator: ScriptedGenai
    planner: ScriptedGenai

    def orchestrator_model(self, *, budget: TurnBudget | None = None) -> GoogleModel:
        """The orchestrator's model (`orchestrator.model.orchestrator_model`)
        on the orchestrator's scripted client."""
        return orchestrator_model(MODEL, client=self.orchestrator.client, budget=budget)

    def planner_model(self, *, budget: TurnBudget | None = None) -> GoogleModel:
        """The planner's model (`planner.model.planner_model`) on the
        planner's scripted client."""
        return planner_model(MODEL, client=self.planner.client, budget=budget)


def scripted_models(
    *, orchestrator: Sequence[Step] = (), planner: Sequence[Step] = ()
) -> ScriptedModels:
    """Scripted clients answering the orchestrator from `orchestrator` and
    the planner from `planner`. A call past its agent's script raises
    `tests.scripted_genai.ScriptExhaustedError`."""
    return ScriptedModels(orchestrator=ScriptedGenai(orchestrator), planner=ScriptedGenai(planner))


@contextlib.contextmanager
def numbered_call_ids() -> Iterator[None]:
    """Number the ids `harness.gemini_wire.client_call_id` gives calls the
    model sent without one, from `adk-1`, so each call carries the same id
    every time a test runs."""
    counter = itertools.count(1)

    def numbered() -> str:
        return f"{gemini_wire.CLIENT_CALL_ID_PREFIX}{next(counter)}"

    with mock.patch.object(gemini_wire, "client_call_id", numbered):
        yield


def request_text(request: Request) -> str:
    """Every text part of a recorded request, and every function response
    in it as JSON, one per line."""
    lines: list[str] = []
    for content in request.content_list:
        for part in content.parts or []:
            if part.function_response is not None:
                lines.append(json.dumps(part.function_response.response, sort_keys=True))
            elif part.text:
                lines.append(part.text)
    return "\n".join(lines)


def unconnected_store() -> StudioStore:
    """A `StudioStore` with no database behind its engine, for a test that
    reads only the in-process state a store holds (`StudioStore.current`), as
    the recorder does. Any read or write the test makes through it fails at
    once: port 1 on the loopback refuses the connection, and a 1 s connect
    timeout bounds the attempt where it does not."""
    return StudioStore(
        create_async_engine(
            "postgresql+asyncpg://unconnected@127.0.0.1:1/none", connect_args={"timeout": 1}
        ),
    )


@dataclass(frozen=True)
class SegmentRun:
    """One segment of a turn: how it ended, the cursor as it left it, and
    the events it emitted, in order."""

    outcome: Outcome
    cursor: Cursor
    events: list[TurnEvent] = field(default_factory=list)

    @property
    def pause(self) -> Paused:
        """The pause the segment ended on.

        Raises:
            AssertionError: the segment did not end paused.
        """
        assert self.outcome is Outcome.PAUSED, self.outcome
        last = self.events[-1]
        assert isinstance(last, Paused), last
        return last


class TurnDriver:
    """One harness instance running turns of one conversation over `graph`.

    `studio` is the instance's `StudioStore`, which holds the conversation's
    lease and its Stop flag, and `turns` its `TurnStore`. `store_kwargs`
    reach the `StudioStore`, such as a `tests.store_clock.StoreClock`'s.
    `observe`, when set, hears each event as the segment emits it, as
    Studio's segment work does (`harness.studio._SegmentWork.drive`).
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        graph: Graph,
        principal: str = PRINCIPAL,
        conversation_id: str = CONVERSATION,
        heartbeat_seconds: float = HEARTBEAT_S,
        store_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.graph = graph
        self.principal = principal
        self.conversation_id = conversation_id
        self.heartbeat_seconds = heartbeat_seconds
        self.observe: Callable[[TurnEvent], None] | None = None
        self.turns = TurnStore(engine)
        self.studio = StudioStore(engine, **(store_kwargs or {}))

    async def turn(
        self,
        message: str,
        *,
        turn_index: int = 1,
        inputs: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> SegmentRun:
        """Open turn `turn_index` on `message` at the graph's start and run
        its first segment. `inputs` join `{"message": message}` as the
        turn's fixed inputs; `state` is its opening state. The transcript
        opens on the user's message."""
        segment_id = await self._claim()
        cursor = await self.turns.open_turn(
            self.principal,
            self.conversation_id,
            segment_id=segment_id,
            turn_index=turn_index,
            node=self.graph.start,
            inputs={"message": message, **(inputs or {})},
            state=state or {},
            records=[entries.user_message(message)],
        )
        return await self._run(cursor, segment_id, answer=None)

    async def answer(self, value: dict[str, Any], *, to: Paused) -> SegmentRun:
        """Answer the pause `to` with `value` and run the segment that
        resumes the turn."""
        segment_id = await self._claim()
        cursor = await self.open_cursor()
        answer = Answer(interrupt_id=to.interrupt_id, round=to.round, value=value)
        return await self._run(cursor, segment_id, answer=answer)

    async def stop(self) -> bool:
        """Post a Stop for the conversation, as `/cancel` does; returns
        whether a live segment was signalled."""
        return await self.studio.request_cancel(self.principal, self.conversation_id)

    async def open_cursor(self) -> Cursor:
        """The conversation's open turn.

        Raises:
            AssertionError: the conversation has no open turn.
        """
        cursor = await self.turns.open_cursor(self.principal, self.conversation_id)
        assert cursor is not None, "the conversation has no open turn"
        return cursor

    async def records(self, cursor: Cursor) -> list[AgentRecord]:
        """The transcript of `cursor`'s turn as stored."""
        return await self.turns.records(cursor)

    async def _claim(self) -> str:
        """Claim the conversation's lease, creating the conversation on its
        first turn."""
        if await self.studio.load(self.principal, self.conversation_id) is None:
            await self.studio.create(self.principal, self.conversation_id)
        segment_id = self.studio.claim_segment(self.principal, self.conversation_id)
        await self.studio.save(self.principal, self.conversation_id)
        return segment_id

    async def _run(self, cursor: Cursor, segment_id: str, *, answer: Answer | None) -> SegmentRun:
        events: list[TurnEvent] = []

        def emit(event: TurnEvent) -> None:
            events.append(event)
            if self.observe is not None:
                self.observe(event)

        try:
            segment = await run_segment(
                self.turns,
                self.studio,
                cursor,
                principal=self.principal,
                conversation_id=self.conversation_id,
                segment_id=segment_id,
                graph=self.graph,
                emit=emit,
                answer=answer,
                heartbeat_seconds=self.heartbeat_seconds,
            )
        finally:
            await self.studio.drop_lease(self.principal, self.conversation_id, segment_id)
        assert segment.outcome is not None
        return SegmentRun(outcome=segment.outcome, cursor=segment.cursor, events=events)
