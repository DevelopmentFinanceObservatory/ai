"""The planner's code without its model, as plan sessions an outside model drives.

A plan session holds what one planner entry holds on dev: the session state the
orchestrator's `search` and `plan` tools leave, the state `PlannerSeam.__call__`
writes before it runs the planner, and the session's events, which the
planner's own readers walk to learn what the user was asked and answered.

- `PlanningService.open` runs each search through the orchestrator's own
  `search` tool (`harness.orchestrator_tools`), so pin sources come from Search's
  answers and never from blocks a client asserts. It hands the brief and asks to
  the orchestrator's `plan` tool, runs the graph's real `route` and `plan_start`
  nodes on that state, and then `PlannerSeam.__call__` itself, stopped at the
  model call (`_EntrySeam`). The seam writes the pin sources and every
  entry-scoped slot, and renders the planner's first message.
- `PlanningService.propose` runs the served planner's own `propose_plan` tool,
  taken from the seam's agent, through ADK's `FunctionTool.run_async` over a
  tool context backed by the session, and records the call and its response as
  ADK events.
- `PlanningService.answer` records a question the outside model asked its user
  and the answer, checked by the planner's own ask tool, as the
  `adk_request_input` call and response events the readers pair.
- `PlanningService.run` runs an accepted version on dev's own `execute` and
  `plan_done` nodes (`run_graph`), and hands back what `plan_done` hands the
  orchestrator.

Sessions are keyed on an id `open` returns, bound to the caller that opened
them.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from compiler.surface import ToolSurface
from google.adk.events import Event, EventActions
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from harness.graph.orchestrate import route
from harness.graph.planning import plan_start
from harness.graph.state import ORCHESTRATOR_SPECS_KEY, conversation_readings
from harness.orchestrator_tools import HANDED_OFF, Resolver, build_orchestrator_tools
from harness.planner_seam import PlannerSeam, store_searched_pin_readings
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.turn_budget import TurnBudget, current_turn_budget
from harness.turn_events import REQUEST_INPUT_FUNCTION_CALL_NAME
from planner.agent import AGENT_NAME as PLANNER_AGENT
from planner.plan import Plan, PlanStatus, current_recorded_identity, load_plan
from translator.discovery import CutCalls, current_cut_calls, resolving_once
from translator.tool import Translate

PROPOSE_PLAN = "propose_plan"

# The codes this server answers with itself, in the planner's refusal shape.
SESSION_UNKNOWN = "plan_session_unknown"
SEARCH_FAILED = "search_failed"
HAND_OFF_REFUSED = "hand_off_refused"
VERSION_NOT_READY = "plan_version_not_ready"
VERSION_SUPERSEDED = "plan_version_superseded"
RUN_FAILED = "run_failed"


def refusal(code: str, message: str, **detail: Any) -> dict[str, Any]:
    """A refusal in the shape of `planner.tool_results.ToolProblem`."""
    answer: dict[str, Any] = {"outcome": "refused", "code": code, "message": message}
    if detail:
        answer["detail"] = detail
    return answer


class _EntryOpened(Exception):  # noqa: N818 -- a stop signal, not a fault
    """Raised where `PlannerSeam._propose` would run the model, carrying the first message."""

    def __init__(self, message: str) -> None:
        super().__init__("entry opened")
        self.message = message


class _EntrySeam(PlannerSeam):
    """`PlannerSeam`, stopped where its entry would call the model.

    `__call__` runs to `_propose` as on dev: it resets the entry's slots, writes
    the pin sources and renders the planner's first message. Its `finally` then
    stores the searched pins and mirrors the live state, as on dev.
    """

    async def _propose(self, *, ctx: Any, message: str, on_progress: Any) -> Any:
        del ctx, on_progress
        raise _EntryOpened(message)


@dataclass
class Run:
    """One run of an accepted version: its task, and the hand-back once it ends."""

    run_id: str
    version: int
    task: asyncio.Task[dict[str, Any]]


@dataclass
class PlanSession:
    """One planner entry an outside model drives, keyed on `session_id`.

    `state` is the session state (orchestrator, graph and planner slots in one
    dict, as dev's graph session holds them). `events` are the session's events,
    the propose and ask calls and responses this server records. `ready` maps
    each version declared ready for approval to the plan as declared.
    """

    session_id: str
    caller: str | None
    state: dict[str, Any]
    conversation: ConversationState
    first_message: str
    events: list[Event] = field(default_factory=list)
    ready: dict[int, Plan] = field(default_factory=dict)
    runs: dict[str, Run] = field(default_factory=dict)

    def tool_context(self, call_id: str) -> Any:
        """The tool context the planner's tools read: this session's state and events."""
        return SimpleNamespace(
            state=self.state,
            session=SimpleNamespace(id=self.session_id, state=self.state, events=self.events),
            actions=EventActions(),
            function_call_id=call_id,
            invocation_id=self.session_id,
            tool_confirmation=None,
            resume_inputs=None,
            user_id=self.caller or "outside",
        )

    def record_call(self, call_id: str, name: str, args: Mapping[str, Any]) -> None:
        self.events.append(
            Event(
                author=PLANNER_AGENT,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(id=call_id, name=name, args=dict(args))
                        )
                    ],
                ),
            )
        )

    def record_response(self, call_id: str, name: str, response: Mapping[str, Any]) -> None:
        self.events.append(
            Event(
                author=PLANNER_AGENT,
                content=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=call_id, name=name, response=dict(response)
                            )
                        )
                    ],
                ),
            )
        )

    @contextlib.contextmanager
    def request_scope(self) -> Iterator[None]:
        """What dev sets around a planner request: its turn budget, the conversation's
        recorded identities and the turn's resolve memo."""
        budget = current_turn_budget.set(TurnBudget.from_settings(Settings()))
        identities = current_recorded_identity.set(self.conversation.recorded_identity)
        try:
            with resolving_once(self.state):
                yield
        finally:
            current_recorded_identity.reset(identities)
            current_turn_budget.reset(budget)


class PlanningService:
    """Plan sessions over the served planner's tools, translation and executor.

    `resolver` answers searches (`harness.mcp_client.McpResolver` when served).
    `surface` is the planner's tool surface, `translate` its translation
    (`translator.tool.build_translate` when served). `run_graph` runs an
    approved plan and answers the hand-back (`execution.dev_run_graph`).
    `check_version` is the version check `run` makes; only a control turns it off.
    """

    def __init__(
        self,
        *,
        resolver: Resolver,
        surface: ToolSurface,
        translate: Translate,
        run_graph: Callable[[PlanSession, Plan], Awaitable[dict[str, Any]]],
        check_version: bool = True,
    ) -> None:
        self._resolver = resolver
        self._seam = _EntrySeam(settings=Settings(), surface=surface, translate=translate)
        tools: dict[str, BaseTool] = {
            tool.name: tool for tool in self._seam._agent.tools if isinstance(tool, BaseTool)
        }
        self.propose_tool = tools[PROPOSE_PLAN]
        self.ask_tool = tools[REQUEST_INPUT_FUNCTION_CALL_NAME]
        self._run_graph = run_graph
        self._check_version = check_version
        self.sessions: dict[str, PlanSession] = {}

    async def open(
        self,
        *,
        brief: str,
        asks: Sequence[Any],
        searches: Sequence[Mapping[str, Any]],
        caller: str | None,
        resolver: Resolver | None = None,
    ) -> PlanSession | dict[str, Any]:
        """A new plan session from `searches`, `brief` and `asks`, or the refusal
        the orchestrator's own tools answered."""
        tools = {
            tool.__name__: tool
            for tool in build_orchestrator_tools(
                resolver=resolver or self._resolver, executor=_NoExecutor()
            )
        }
        context = SimpleNamespace(state={}, actions=EventActions())
        for search in searches:
            found = await tools["search"](
                query=search["query"],
                tool_context=context,
                parts=search.get("parts") or None,
                as_text=bool(search.get("as_text", False)),
            )
            if "error" in found:
                return refusal(SEARCH_FAILED, str(found["error"]), query=search["query"])
        handed = tools["plan"](brief=brief, asks=list(asks), tool_context=context)
        if handed.get(HANDED_OFF) != "plan":
            return refusal(HAND_OFF_REFUSED, "the brief or its asks were refused", answer=handed)
        state: dict[str, Any] = context.state
        graph = SimpleNamespace(state=state)
        (pending,) = [event.output for event in route(graph, None)]
        plan_start(graph, pending)
        session_id = "ps_" + secrets.token_urlsafe(16)
        conversation = ConversationState(conversation_id=session_id)
        conversation.turn_specs = dict(state.get(ORCHESTRATOR_SPECS_KEY) or {})
        session = PlanSession(
            session_id=session_id,
            caller=caller,
            state=state,
            conversation=conversation,
            first_message="",
        )
        ctx = SimpleNamespace(
            state=state,
            session=SimpleNamespace(id=session_id, state=state, events=session.events),
            resume_inputs=None,
            user_id=caller or "outside",
        )
        with session.request_scope():
            try:
                await self._seam(ctx=ctx, question=brief, readings=conversation_readings(state))
            except _EntryOpened as opened:
                session.first_message = opened.message
        self.sessions[session_id] = session
        return session

    def session(self, session_id: str, *, caller: str | None) -> PlanSession | None:
        """The session `session_id` names, when `caller` opened it."""
        found = self.sessions.get(session_id)
        if found is None or found.caller != caller:
            return None
        return found

    async def propose(self, session: PlanSession, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """`propose_plan` on `session`, as the served planner's tool answers it."""
        call_id = "pp_" + uuid.uuid4().hex[:16]
        session.record_call(call_id, PROPOSE_PLAN, arguments)
        ctx = session.tool_context(call_id)
        cut = CutCalls()
        token = current_cut_calls.set(cut)
        try:
            with session.request_scope():
                answer = await self.propose_tool.run_async(args=dict(arguments), tool_context=ctx)
            store_searched_pin_readings(ctx, session.state)
        finally:
            current_cut_calls.reset(token)
            await cut.close()
        session.record_response(call_id, PROPOSE_PLAN, answer)
        plan = load_plan(session.state)
        if "summary" in answer and plan is not None:
            session.ready[plan.version] = plan
        return answer

    async def answer(
        self, session: PlanSession, arguments: Mapping[str, Any], answer: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Record a question asked of the user and their `answer`, or the ask tool's refusal.

        `arguments` are the planner's ask arguments (`message`, `candidates`,
        `options`, `recommended_spec_id`, `rationale`). The ask tool checks them as
        it does on dev; a refused ask is recorded with its refusal and no answer.
        """
        call_id = "ask_" + uuid.uuid4().hex[:16]
        session.record_call(call_id, REQUEST_INPUT_FUNCTION_CALL_NAME, arguments)
        with session.request_scope():
            checked = await self.ask_tool.run_async(
                args=dict(arguments), tool_context=session.tool_context(call_id)
            )
        if checked is not None:
            session.record_response(call_id, REQUEST_INPUT_FUNCTION_CALL_NAME, checked)
            return checked
        session.record_response(
            call_id,
            REQUEST_INPUT_FUNCTION_CALL_NAME,
            {"result": {"answers": {"q1": dict(answer)}}},
        )
        return {"outcome": "ok", "recorded": call_id}

    def start_run(self, session: PlanSession, version: int) -> Run | dict[str, Any]:
        """Start, or find, the run of `version`, or the refusal that stops it.

        A version runs once it was declared ready and while it is the plan on
        record: a later accepted version supersedes it. The user's approval is
        stamped now, as the approval card stamps the answer's time on dev.
        """
        for run in session.runs.values():
            if run.version == version:
                return run
        declared = session.ready.get(version)
        if declared is None:
            return refusal(
                VERSION_NOT_READY,
                f"version {version} was never declared ready; propose a plan first",
                ready=sorted(session.ready),
            )
        current = load_plan(session.state)
        if self._check_version and (
            current is None
            or current.version != version
            or current.status is not PlanStatus.PENDING_APPROVAL
        ):
            return refusal(
                VERSION_SUPERSEDED,
                f"version {version} is not the plan on record; run the current version",
                current=current.version if current is not None else None,
            )
        approved = declared.approve(at=datetime.now(UTC))
        run_id = "run_" + uuid.uuid4().hex[:16]
        run = Run(run_id, version, asyncio.create_task(self._run_graph(session, approved)))
        session.runs[run_id] = run
        return run


class _NoExecutor:
    """The executor `build_orchestrator_tools` requires; `search` and `plan` never call it."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"opening a plan session runs no step: {kwargs.get('tool')}")
