"""The Studio turn as an explicit state machine over two Pydantic AI agents.

Every state is one async function that returns an `Outcome`; the loop persists the
outcome (cursor plus changed conversations, one compare-and-swap) before it goes on.
A pause is a cursor in an `AWAIT_*` state, so a resume reads the cursor and nothing
else.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal

from agents import Deps, PlanProgress, accepted_plans
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, Conversation, DeferredToolRequests
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from store import StaleCursorError, Store

type Frame = dict[str, Any]
type Emit = Callable[[Frame], None]


class State(StrEnum):
    ORCHESTRATE = "ORCHESTRATE"
    PLANNING = "PLANNING"
    AWAIT_ASK = "AWAIT_ASK"
    AWAIT_APPROVAL = "AWAIT_APPROVAL"
    EXECUTE = "EXECUTE"
    PLAN_DONE = "PLAN_DONE"
    DONE = "DONE"


PAUSED = {State.AWAIT_ASK, State.AWAIT_APPROVAL}


class RefusedError(Exception):
    """A resume or start that the stored cursor does not allow."""


class Approve(BaseModel):
    kind: Literal["approve"] = "approve"
    revision: int


class Amend(BaseModel):
    kind: Literal["amend"] = "amend"
    text: str


PlanAnswer = TypeAdapter(Annotated[Approve | Amend, Field(discriminator="kind")])


class PlannerInput(BaseModel):
    """What the planner is run with: the brief, the user's choice, or an amendment."""

    kind: Literal["brief", "answer", "amend"]
    text: str


class Cursor(BaseModel):
    state: str
    user_text: str
    plan_call_id: str | None = None
    planner_input: PlannerInput | None = None
    revision: int | None = None
    plan_hash: str | None = None
    interrupt_id: str | None = None
    pauses: int = 0
    exec_result: dict | None = None


@dataclass
class Outcome:
    cursor: Cursor
    conversations: dict[str, Conversation] = field(default_factory=dict)
    frame: Frame | None = None  # sent after the checkpoint: a pause or `done`


type StateFn = Callable[[str, Cursor, Emit], Awaitable[Outcome]]


def execute_stub(plan_hash: str) -> dict[str, Any]:
    """The deterministic stand-in for running the approved plan."""
    return {"executed": True, "plan_hash": plan_hash, "rows": 12}


@dataclass
class Loop:
    """The turn driver.

    `buffer_planner_events` and `drain_before_pause` switch on deliberately broken
    behaviour so tests can show that they fail without the correct one.
    `after_load` is awaited between a resume reading the cursor and claiming it, so a
    test can hold two resumers inside that window.
    """

    store: Store
    orchestrator: Agent[Deps, Any]
    planner: Agent[Deps, Any]
    deps: Deps
    buffer_planner_events: bool = False
    drain_before_pause: bool = True
    after_load: Callable[[], Awaitable[None]] | None = None
    states: dict[str, StateFn] = field(init=False)

    def __post_init__(self) -> None:
        self.states = {
            State.ORCHESTRATE: self._orchestrate,
            State.PLANNING: self._planning,
            State.EXECUTE: self._execute,
            State.PLAN_DONE: self._plan_done,
        }

    # Entry points ---------------------------------------------------------

    async def start_turn(self, conversation_id: str, user_text: str) -> AsyncGenerator[Frame]:
        cursor = Cursor(state=self.opening_state(), user_text=user_text)
        try:
            self.store.create_turn(conversation_id, cursor.model_dump_json())
        except StaleCursorError as error:
            raise RefusedError(str(error)) from error
        async for frame in self._stream(conversation_id, cursor, 0):
            yield frame

    async def resume(
        self, conversation_id: str, interrupt_id: str, answer: Any
    ) -> AsyncGenerator[Frame]:
        cursor, version = self._load(conversation_id)
        if self.after_load is not None:
            await self.after_load()
        claimed = self._claim(cursor, interrupt_id, answer)
        try:
            version = self.store.checkpoint(
                conversation_id,
                expected_version=version,
                cursor=claimed.model_dump_json(),
                conversations={},
            )
        except StaleCursorError as error:
            raise RefusedError(f"another resume took this pause: {error}") from error
        async for frame in self._stream(conversation_id, claimed, version, resumed=True):
            yield frame

    async def recover(self, conversation_id: str) -> AsyncGenerator[Frame]:
        """Run a turn whose process died between pauses, from its last checkpoint."""
        cursor, version = self._load(conversation_id)
        if cursor.state in PAUSED or cursor.state == State.DONE:
            raise RefusedError(f"state {cursor.state} has nothing to recover")
        async for frame in self._stream(conversation_id, cursor, version):
            yield frame

    def opening_state(self) -> str:
        return State.ORCHESTRATE

    async def on_resume(self, conversation_id: str, cursor: Cursor, emit: Emit) -> None:
        """Hook run before the first state of a resumed turn. Does nothing."""

    # Claim ----------------------------------------------------------------

    def _load(self, conversation_id: str) -> tuple[Cursor, int]:
        row = self.store.read_turn(conversation_id)
        if row is None:
            raise RefusedError(f"no turn {conversation_id}")
        return Cursor.model_validate_json(row[0]), row[1]

    def _claim(self, cursor: Cursor, interrupt_id: str, answer: Any) -> Cursor:
        """The cursor a valid answer moves a paused turn to, or a refusal."""
        if cursor.state not in PAUSED or cursor.interrupt_id != interrupt_id:
            raise RefusedError(f"interrupt {interrupt_id} is not pending")
        if cursor.state == State.AWAIT_ASK:
            planner_input = PlannerInput(kind="answer", text=str(answer))
            return cursor.model_copy(
                update={
                    "state": State.PLANNING,
                    "planner_input": planner_input,
                    "interrupt_id": None,
                }
            )
        parsed = PlanAnswer.validate_python(answer)
        if isinstance(parsed, Approve):
            if parsed.revision != cursor.revision:
                raise RefusedError(
                    f"revision {parsed.revision} is not the stored {cursor.revision}"
                )
            update = {"state": State.EXECUTE, "interrupt_id": None}
        else:
            planner_input = PlannerInput(kind="amend", text=parsed.text)
            update = {
                "state": State.PLANNING,
                "planner_input": planner_input,
                "interrupt_id": None,
            }
        return cursor.model_copy(update=update)

    # Streaming ------------------------------------------------------------

    async def _stream(
        self,
        conversation_id: str,
        cursor: Cursor,
        version: int,
        *,
        resumed: bool = False,
    ) -> AsyncGenerator[Frame]:
        """Run the machine in a task and forward its frames as they are produced."""
        queue: asyncio.Queue[Frame] = asyncio.Queue()

        async def machine() -> Frame | None:
            if resumed:
                await self.on_resume(conversation_id, cursor, queue.put_nowait)
            return await self._advance(conversation_id, cursor, version, queue.put_nowait)

        task = asyncio.create_task(machine())
        getter: asyncio.Future[Frame] | None = None
        try:
            while not task.done():
                getter = asyncio.ensure_future(queue.get())
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if not getter.done():
                    getter.cancel()
                    break
                yield self._sent(conversation_id, getter.result())
            if self.drain_before_pause:
                while not queue.empty():
                    yield self._sent(conversation_id, queue.get_nowait())
            final = task.result()
            if final is not None:
                yield self._sent(conversation_id, final)
        finally:
            if getter is not None and not getter.done():
                getter.cancel()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _sent(self, conversation_id: str, frame: Frame) -> Frame:
        self.store.append_frame(conversation_id, frame)
        return frame

    async def _advance(
        self, conversation_id: str, cursor: Cursor, version: int, emit: Emit
    ) -> Frame | None:
        frame = None
        while cursor.state not in PAUSED and cursor.state != State.DONE:
            outcome = await self.states[cursor.state](conversation_id, cursor, emit)
            version = self.store.checkpoint(
                conversation_id,
                expected_version=version,
                cursor=outcome.cursor.model_dump_json(),
                conversations=outcome.conversations,
            )
            cursor, frame = outcome.cursor, outcome.frame
        return frame

    # Agents ---------------------------------------------------------------

    async def _run_agent(
        self,
        agent: Agent[Deps, Any],
        name: str,
        emit: Emit,
        *,
        prompt: str | None,
        conversation: Conversation,
        deferred_tool_results: Any = None,
    ) -> AgentRunResult[Any]:
        async with agent.run_stream_events(
            prompt,
            deps=self.deps,
            conversation=conversation,
            deferred_tool_results=deferred_tool_results,
        ) as events:
            async for event in events:
                match event:
                    case PartStartEvent(part=TextPart(content=text)) if text:
                        emit({"type": "message.delta", "agent": name, "text": text})
                    case PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
                        emit({"type": "message.delta", "agent": name, "text": text})
                    case PlanProgress():
                        emit(
                            {
                                "type": "activity.step",
                                "agent": name,
                                "done": event.done,
                                "total": event.total,
                                "note": event.note,
                            }
                        )
                    case AgentRunResultEvent():
                        return event.result
        raise RuntimeError(f"{name} run ended without a result")

    # States ---------------------------------------------------------------

    async def _orchestrate(self, conversation_id: str, cursor: Cursor, emit: Emit) -> Outcome:
        result = await self._run_agent(
            self.orchestrator,
            "orchestrator",
            emit,
            prompt=cursor.user_text,
            conversation=Conversation(conversation_id=f"{conversation_id}:orchestrator"),
        )
        conversations = {"orchestrator": result.conversation}
        output = result.output
        if isinstance(output, DeferredToolRequests):
            (call,) = output.calls
            planning = PlannerInput(kind="brief", text=call.args_as_dict()["brief"])
            moved = cursor.model_copy(
                update={
                    "state": State.PLANNING,
                    "plan_call_id": call.tool_call_id,
                    "planner_input": planning,
                }
            )
            return Outcome(moved, conversations)
        done = cursor.model_copy(update={"state": State.DONE})
        return Outcome(done, conversations, {"type": "done", "text": output})

    async def _planning(self, conversation_id: str, cursor: Cursor, emit: Emit) -> Outcome:
        planner_input = cursor.planner_input
        assert planner_input is not None
        buffered: list[Frame] = []
        sink = buffered.append if self.buffer_planner_events else emit

        if planner_input.kind == "brief":
            result = await self._run_agent(
                self.planner,
                "planner",
                sink,
                prompt=planner_input.text,
                conversation=Conversation(conversation_id=f"{conversation_id}:planner"),
            )
        else:
            stored = self.store.require_conversation(conversation_id, "planner")
            results = None
            prompt = planner_input.text
            if planner_input.kind == "answer":
                requests = stored.deferred_tool_requests
                assert requests is not None
                (ask,) = requests.calls
                results = requests.build_results(calls={ask.tool_call_id: prompt})
                prompt = None
            result = await self._run_agent(
                self.planner,
                "planner",
                sink,
                prompt=prompt,
                conversation=stored,
                deferred_tool_results=results,
            )
        for frame in buffered:
            emit(frame)

        conversations = {"planner": result.conversation}
        output = result.output
        pauses = cursor.pauses + 1
        interrupt_id = f"{conversation_id}:{pauses}"
        if isinstance(output, DeferredToolRequests):
            (ask,) = output.calls
            args = ask.args_as_dict()
            frame = {
                "type": "elicitation.request",
                "interrupt_id": interrupt_id,
                "message": args["message"],
                "options": args["options"],
            }
            moved = cursor.model_copy(
                update={
                    "state": State.AWAIT_ASK,
                    "interrupt_id": interrupt_id,
                    "pauses": pauses,
                }
            )
            return Outcome(moved, conversations, frame)

        accepted, steps = accepted_plans(result.conversation.messages)[-1]
        frame = {
            "type": "plan.proposed",
            "interrupt_id": interrupt_id,
            "revision": accepted["revision"],
            "plan_hash": accepted["plan_hash"],
            "steps": [step.model_dump() for step in steps],
        }
        moved = cursor.model_copy(
            update={
                "state": State.AWAIT_APPROVAL,
                "interrupt_id": interrupt_id,
                "pauses": pauses,
                "revision": accepted["revision"],
                "plan_hash": accepted["plan_hash"],
            }
        )
        return Outcome(moved, conversations, frame)

    async def _execute(self, conversation_id: str, cursor: Cursor, emit: Emit) -> Outcome:
        assert cursor.plan_hash is not None
        total = 2
        for done in range(1, total + 1):
            emit(
                {
                    "type": "activity.step",
                    "agent": "executor",
                    "done": done,
                    "total": total,
                    "note": f"execute step {done} of {total}",
                }
            )
            await asyncio.sleep(self.deps.delay)
        moved = cursor.model_copy(
            update={
                "state": State.PLAN_DONE,
                "exec_result": execute_stub(cursor.plan_hash),
            }
        )
        return Outcome(moved)

    async def _plan_done(self, conversation_id: str, cursor: Cursor, emit: Emit) -> Outcome:
        stored = self.store.require_conversation(conversation_id, "orchestrator")
        requests = stored.deferred_tool_requests
        assert requests is not None and cursor.plan_call_id is not None
        results = requests.build_results(calls={cursor.plan_call_id: cursor.exec_result})
        result = await self._run_agent(
            self.orchestrator,
            "orchestrator",
            emit,
            prompt=None,
            conversation=stored,
            deferred_tool_results=results,
        )
        done = cursor.model_copy(update={"state": State.DONE})
        frame = {"type": "done", "text": result.output, "plan_hash": cursor.plan_hash}
        return Outcome(done, {"orchestrator": result.conversation}, frame)
