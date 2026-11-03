"""The planner's code without its model, as plan sessions an outside model drives.

A plan session holds what one planner entry holds on dev: the session state the
orchestrator's `search` and `plan` tools leave, the state `PlannerSeam.__call__`
writes before it runs the planner, and the session's events, which the
planner's own readers walk to learn what the user was asked and answered.

- `PlanningService.open` reads each spec_id the client's own searches returned
  again through Search (`take_spec`) and runs each search it sends through the
  orchestrator's own `search` tool (`harness.orchestrator_tools`), so pin sources
  come from Search's answers and never from blocks a client asserts. It hands
  the brief and asks to the orchestrator's `plan` tool, runs the graph's real
  `route` and `plan_start` nodes on that state, and then `PlannerSeam.__call__`
  itself, stopped at the model call (`_EntrySeam`). The seam writes the pin
  sources and every entry-scoped slot, and renders the planner's first message.
- `PlanningService.propose` runs the served planner's own `propose_plan` tool,
  taken from the seam's agent, through ADK's `FunctionTool.run_async` over a
  tool context backed by the session, and records the call and its response as
  ADK events.
- `PlanningService.answer` records a question asked of the user and the
  answer, checked by the planner's own ask tool, as the `adk_request_input`
  call and response events the readers pair, with who gave the answer: `user`
  through MCP elicitation, or `client`.
- `PlanningService.start_run` runs an accepted version the user approved on
  dev's own `execute` and `plan_done` nodes (`run_graph`), and hands back what
  `plan_done` hands the orchestrator.

Sessions are keyed on an id `open` returns, bound to the caller that opened
them, and saved to a `SessionStore` after every change.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Protocol

from compiler.surface import ToolSurface
from data_mcp.definition_store import IdentityMismatch
from data_mcp.definition_writes import IdentityUnusable, caller_definition
from google.adk.events import Event, EventActions
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from harness.graph.orchestrate import route
from harness.graph.planning import plan_start
from harness.graph.state import (
    ORCHESTRATOR_SPECS_KEY,
    conversation_readings,
    frame_store_ref,
    store_search,
)
from harness.orchestrator_tools import HANDED_OFF, Resolver, build_orchestrator_tools
from harness.planner_seam import PlannerSeam, store_searched_pin_readings
from harness.seeds import stale_reason
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.tool_contracts import resolver_digest
from harness.turn_budget import TurnBudget, current_turn_budget
from harness.turn_events import REQUEST_INPUT_FUNCTION_CALL_NAME
from outward import table
from planner.agent import AGENT_NAME as PLANNER_AGENT
from planner.plan import Plan, PlanStatus, current_recorded_identity, load_plan
from translator.discovery import (
    CutCalls,
    DefinitionNotResolvedError,
    PartNotResolvedError,
    current_cut_calls,
    resolve_definition,
    resolved_spec,
    resolving_once,
)
from translator.tool import Translate

PROPOSE_PLAN = "propose_plan"

# The codes this server answers with itself, in the planner's refusal shape.
SESSION_UNKNOWN = "plan_session_unknown"
SEARCH_FAILED = "search_failed"
HAND_OFF_REFUSED = "hand_off_refused"
VERSION_NOT_READY = "plan_version_not_ready"
VERSION_SUPERSEDED = "plan_version_superseded"
RUN_FAILED = "run_failed"
SPEC_NOT_HELD = "spec_not_held"
SPEC_NOT_RESOLVED = "spec_not_resolved"
NOT_APPROVED = "plan_not_approved"

# Who gave an answer or an approval: the user, asked by this server through MCP
# elicitation, or the client, on its word.
BY_USER = "user"
BY_CLIENT = "client"

# A run's status while its task runs, and once a restart lost it.
RUNNING = "running"
INTERRUPTED = "interrupted"

# How many rows, and characters of them as compact JSON, `run_plan` returns
# across a run's frames. In the live clients' finished runs on the ten sample
# tasks, the frames a run handed back held at most 16 rows (series times years),
# and one run 648; 60 rows holds a 60-year annual series for one place. 6,000
# characters is about 1,500 tokens, under 4% of the 43,500 input tokens a Gemini
# client's request carried, so a run's rows cost less than one more request.
ROW_CAP = 60
ROW_CHARS_CAP = 6_000

# The answer `DefinitionLookup` gives a spec_id: the identity it hashes from, as
# Search served it, or `None` when no definition of it is held.
type DefinitionLookup = Callable[[str], Awaitable[Mapping[str, Any] | None]]

# The rows of the frame a `{spec_id, content_version}` pair names, or `None` when
# they cannot be read.
type RowReader = Callable[[Mapping[str, Any]], Awaitable[list[dict[str, Any]] | None]]


class DefinitionUnavailableError(Exception):
    """The definitions could not be read just now."""


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
    """One run of an accepted version: its task while it runs, its status, and its
    result once it ends. A run loaded after a restart has no task."""

    run_id: str
    version: int
    task: asyncio.Task[dict[str, Any]] | None
    status: str = RUNNING
    result: dict[str, Any] | None = None


@dataclass
class PlanSession:
    """One planner entry an outside model drives, keyed on `session_id`.

    `state` is the session state (orchestrator, graph and planner slots in one
    dict, as dev's graph session holds them). `events` are the session's events,
    the propose and ask calls and responses this server records. `ready` maps
    each version declared ready for approval to the plan as declared.
    `answers` holds each recorded answer with who gave it, and `approvals` each
    version's approval or refusal with who gave it.
    """

    session_id: str
    caller: str | None
    state: dict[str, Any]
    conversation: ConversationState
    first_message: str
    events: list[Event] = field(default_factory=list)
    ready: dict[int, Plan] = field(default_factory=dict)
    runs: dict[str, Run] = field(default_factory=dict)
    answers: list[dict[str, Any]] = field(default_factory=list)
    approvals: dict[int, dict[str, Any]] = field(default_factory=dict)

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


class SessionStore(Protocol):
    """Where sessions outlive the process: `store.SqliteSessions` when served."""

    def save(self, session: PlanSession) -> None: ...

    def load(self, session_id: str) -> PlanSession | None: ...


class MemorySessions:
    """Keeps nothing past the process: a session lives in the service's own dict."""

    def save(self, session: PlanSession) -> None:
        del session

    def load(self, session_id: str) -> PlanSession | None:
        del session_id


class PlanningService:
    """Plan sessions over the served planner's tools, translation and executor.

    `resolver` answers searches (`harness.mcp_client.McpResolver` when served).
    `surface` is the planner's tool surface, `translate` its translation
    (`translator.tool.build_translate` when served). `definitions` reads the
    identity of a spec_id a client's search returned (`serve.held_identity` when
    served); left unset, `open` takes no spec_ids. `run_graph` runs an approved
    plan and answers the hand-back (`execution.dev_run_graph`). `store` keeps
    sessions past a restart. `read_rows` reads a finished run's frames for
    `run_data`, within `row_cap` rows and `row_chars_cap` characters. `check_version`
    is the version check `start_run` makes; only a control turns it off.
    """

    def __init__(
        self,
        *,
        resolver: Resolver,
        surface: ToolSurface,
        translate: Translate,
        run_graph: Callable[[PlanSession, Plan], Awaitable[dict[str, Any]]],
        definitions: DefinitionLookup | None = None,
        store: SessionStore | None = None,
        read_rows: RowReader | None = None,
        row_cap: int = ROW_CAP,
        row_chars_cap: int = ROW_CHARS_CAP,
        check_version: bool = True,
    ) -> None:
        self._resolver = resolver
        self._surface = surface
        self._seam = _EntrySeam(settings=Settings(), surface=surface, translate=translate)
        tools: dict[str, BaseTool] = {
            tool.name: tool for tool in self._seam._agent.tools if isinstance(tool, BaseTool)
        }
        self.propose_tool = tools[PROPOSE_PLAN]
        self.ask_tool = tools[REQUEST_INPUT_FUNCTION_CALL_NAME]
        self._run_graph = run_graph
        self._definitions = definitions
        self.store: SessionStore = store or MemorySessions()
        self._read_rows = read_rows
        self._row_caps = (row_cap, row_chars_cap)
        self._check_version = check_version
        self.sessions: dict[str, PlanSession] = {}

    async def open(
        self,
        *,
        brief: str,
        asks: Sequence[Any],
        searches: Sequence[Mapping[str, Any]],
        caller: str | None,
        spec_ids: Sequence[str] = (),
        resolver: Resolver | None = None,
    ) -> PlanSession | dict[str, Any]:
        """A new plan session from `spec_ids`, `searches`, `brief` and `asks`, or the
        refusal that stops it: a spec_id `take_spec` cannot take, or what the
        orchestrator's own tools answered."""
        tools = {
            tool.__name__: tool
            for tool in build_orchestrator_tools(
                resolver=resolver or self._resolver, executor=_NoExecutor()
            )
        }
        context = SimpleNamespace(state={}, actions=EventActions())
        parts = []
        for spec_id in dict.fromkeys(spec_ids):
            taken = await self.take_spec(spec_id, context.state)
            if "outcome" in taken:
                return taken
            parts.append(taken)
        if parts:
            store_search(context.state, resolver_digest({"parts": parts}), origin="seed")
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
        self.store.save(session)
        return session

    async def take_spec(self, spec_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """Search's own part for `spec_id` now, or the refusal that leaves it out.

        The identity `definitions` holds for it must hash to it
        (`data_mcp.definition_writes.caller_definition`), and Search, asked that
        identity again through data-mcp's `resolve_definition`, must answer it as
        the one dataset `spec_id` names (`translator.discovery.resolved_spec`), as
        the graph's `seed` node takes a dataset the user brought from Search. The
        part is kept in `state` for the entry (`resolving_once`), so translation
        reads the same answer.
        """
        identity = await self._held_identity(spec_id)
        if "outcome" in identity:
            return dict(identity)
        try:
            with resolving_once(state):
                part = await resolve_definition(self._surface, spec_id=spec_id, identity=identity)
            resolved_spec(part, spec_id=spec_id)
        except DefinitionNotResolvedError as exc:
            code = SEARCH_FAILED if exc.retryable else SPEC_NOT_RESOLVED
            return refusal(code, f"{spec_id}: {exc}", spec_id=spec_id)
        except PartNotResolvedError as exc:
            return refusal(SPEC_NOT_RESOLVED, f"{spec_id}: {stale_reason(exc)}", spec_id=spec_id)
        return dict(part)

    async def _held_identity(self, spec_id: str) -> Mapping[str, Any]:
        """The identity `definitions` holds for `spec_id`, checked to hash to it, or
        the refusal that leaves it out. An identity holds no `outcome` member."""
        if self._definitions is None:
            return refusal(SPEC_NOT_HELD, "this server reads no spec_id; send `searches`")
        try:
            identity = await self._definitions(spec_id)
        except DefinitionUnavailableError as exc:
            return refusal(SEARCH_FAILED, f"{spec_id} could not be read just now: {exc}")
        if identity is None:
            return refusal(
                SPEC_NOT_HELD, f"no dataset {spec_id} was returned by search", spec_id=spec_id
            )
        try:
            caller_definition({"schema_version": 1, "spec_id": spec_id, "identity": identity})
        except (IdentityMismatch, IdentityUnusable) as exc:
            return refusal(SPEC_NOT_HELD, f"{spec_id} cannot be read: {exc}", spec_id=spec_id)
        return identity

    def session(self, session_id: str, *, caller: str | None) -> PlanSession | None:
        """The session `session_id` names, when `caller` opened it, from the store
        when this process has not seen it."""
        found = self.sessions.get(session_id)
        if found is None:
            found = self.store.load(session_id)
            if found is not None:
                self.sessions[session_id] = found
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
        self.store.save(session)
        return answer

    async def check_ask(self, session: PlanSession, arguments: Mapping[str, Any]) -> Any:
        """The ask tool's refusal of `arguments`, or `None` when it would ask them.
        The tool reads its arguments alone, so checking records nothing."""
        with session.request_scope():
            return await self.ask_tool.run_async(
                args=dict(arguments), tool_context=session.tool_context("check")
            )

    async def answer(
        self,
        session: PlanSession,
        arguments: Mapping[str, Any],
        answer: Mapping[str, Any],
        *,
        by: str,
    ) -> dict[str, Any]:
        """Record a question asked of the user and their `answer`, given `by` the user
        or the client, or the ask tool's refusal.

        `arguments` are the planner's ask arguments (`message`, `candidates`,
        `options`, `recommended_spec_id`, `rationale`). The ask tool checks them as
        it does on dev; a refused ask is recorded with its refusal and no answer.
        """
        call_id = "ask_" + uuid.uuid4().hex[:16]
        session.record_call(call_id, REQUEST_INPUT_FUNCTION_CALL_NAME, arguments)
        checked = await self.check_ask(session, arguments)
        if checked is not None:
            session.record_response(call_id, REQUEST_INPUT_FUNCTION_CALL_NAME, checked)
            self.store.save(session)
            return checked
        session.record_response(
            call_id,
            REQUEST_INPUT_FUNCTION_CALL_NAME,
            {"result": {"answers": {"q1": dict(answer)}}},
        )
        session.answers.append(
            {
                "recorded": call_id,
                "message": arguments.get("message"),
                "answer": dict(answer),
                "by": by,
            }
        )
        self.store.save(session)
        return {"outcome": "ok", "recorded": call_id, "by": by}

    def runnable(self, session: PlanSession, version: int) -> Run | Plan | dict[str, Any]:
        """The run of `version` already started, else the plan to run, else the refusal
        that stops it. A version runs once it was declared ready and while it is the
        plan on record: a later accepted version supersedes it."""
        for run in session.runs.values():
            if run.version == version and run.status != INTERRUPTED:
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
        return declared

    def approve(
        self, session: PlanSession, version: int, *, approved: bool, by: str, comment: str = ""
    ) -> None:
        """Record whether `version` is approved, and `by` whom."""
        record: dict[str, Any] = {
            "approved": approved,
            "by": by,
            "at": datetime.now(UTC).isoformat(),
        }
        if comment:
            record["comment"] = comment
        session.approvals[version] = record
        self.store.save(session)

    def start_run(self, session: PlanSession, version: int) -> Run | dict[str, Any]:
        """Start, or find, the run of `version`, or the refusal that stops it: one of
        `runnable`'s, or `NOT_APPROVED` when the user's approval of it is not on
        record. The approval is stamped on the plan as the approval card stamps the
        answer's time on dev."""
        found = self.runnable(session, version)
        if not isinstance(found, Plan):
            return found
        approval = session.approvals.get(version)
        if approval is None or not approval["approved"]:
            detail = {"by": approval["by"]} if approval else {}
            if approval and approval.get("comment"):
                detail["comment"] = approval["comment"]
            return refusal(
                NOT_APPROVED,
                f"version {version} has no approval on record; show the user the card first",
                **detail,
            )
        approved = found.approve(at=datetime.now(UTC))
        run_id = "run_" + uuid.uuid4().hex[:16]
        run = Run(run_id, version, task=None)
        run.task = asyncio.create_task(self._finish(session, run, approved))
        session.runs[run_id] = run
        self.store.save(session)
        return run

    async def _finish(self, session: PlanSession, run: Run, approved: Plan) -> dict[str, Any]:
        try:
            result = await self._run_graph(session, approved)
        except Exception as exc:
            run.status, run.result = RUN_FAILED, refusal(RUN_FAILED, f"the run failed: {exc!r}")
            self.store.save(session)
            raise
        registry = result.pop("frames", None) or {}
        if result.get("status") == "finished":
            result["data"] = await run_data(
                result["hand_back"].get("frames") or [],
                registry,
                read_rows=self._read_rows,
                caps=self._row_caps,
            )
        run.status, run.result = str(result.get("status")), result
        self.store.save(session)
        return result


async def run_data(
    frames: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
    *,
    read_rows: RowReader | None,
    caps: tuple[int, int],
) -> list[dict[str, Any]]:
    """Each frame a hand-back lists, as the public tools read it: its id, the
    `{spec_id, content_version}` pair `describe_frame` and `export_data` take, and
    its rows (`outward.table`) while all frames' rows stay within `caps`, else its
    row count alone. `registry` is the run's `ORCHESTRATOR_FRAMES_KEY`."""
    row_cap, chars_cap = caps
    rows_left, chars_left = row_cap, chars_cap
    data = []
    for listed in frames:
        entry = registry.get(str(listed.get("frame"))) or {}
        ref = entry.get("ref")
        item: dict[str, Any] = {"frame": listed.get("frame")}
        if not isinstance(ref, Mapping):
            data.append(item)
            continue
        item["frame_ref"] = {k: ref[k] for k in ("spec_id", "content_version") if k in ref}
        rows = await read_rows(frame_store_ref(entry)) if read_rows is not None else None
        if rows is None:
            data.append(item)
            continue
        shown = table(rows)
        size = len(json.dumps(shown, separators=(",", ":"), default=str))
        if len(rows) <= rows_left and size <= chars_left:
            item.update(shown)
            rows_left, chars_left = rows_left - len(rows), chars_left - size
        else:
            item["row_count"] = len(rows)
        data.append(item)
    return data


class _NoExecutor:
    """The executor `build_orchestrator_tools` requires; `search` and `plan` never call it."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"opening a plan session runs no step: {kwargs.get('tool')}")
