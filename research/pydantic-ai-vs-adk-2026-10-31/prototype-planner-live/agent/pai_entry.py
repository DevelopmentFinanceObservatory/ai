"""One planner entry with the planner on Pydantic AI, run as `evals/planner_entry/entry.py`
runs the ADK planner.

The entry is the eval's, step for step: the orchestrator's recorded `search` calls and
`plan` hand-off seed the state (`entry.orchestrator_state`), the graph's own `route`,
`plan_start` and `planner` node functions run over it, and the `planner` node calls the
real `PlannerSeam`, wrapped as the eval wraps it (`entry.Probe.watch`,
`harness.wiring.handing_back_planner_failures`). The seam's one ADK call, `ctx.run_node`,
runs the Pydantic AI agent instead (`PaiNodeContext`, `PaiRunner`).

An accepted ask ends the run like a plan declared ready: the tool's neutral body calls
`end_run`, and the runner stops `Agent.iter` before the next model request. The node
returns the ask, the entry reads it off session state as `entry.run_entry` does, and the
user's answer resumes the node in a run of its own, whose brief carries the answer.

    entry = await run_pai_entry(task, recording, prefix_cache=None)
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parents[2] / "evals" / "planner_entry"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from entry import (  # noqa: E402
    EntryRun,
    Probe,
    RecordedResolver,
    RecordedSurface,
    RecordingMismatchError,
    Task,
    TranslationCalls,
    _no_anchor_read,
    inputs_digest,
    orchestrator_state,
)
from google.adk import Event  # noqa: E402
from google.genai import Client, types  # noqa: E402
from harness.graph.orchestrate import route  # noqa: E402
from harness.graph.planning import plan_start, planner_node  # noqa: E402
from harness.graph.state import ORCHESTRATOR_SPECS_KEY  # noqa: E402
from harness.planner_seam import PlannerSeam  # noqa: E402
from harness.prefix_cache import PrefixCache  # noqa: E402
from harness.settings import Settings  # noqa: E402
from harness.studio_state import ConversationState  # noqa: E402
from harness.turn_budget import TurnBudget, current_turn_budget  # noqa: E402
from harness.wiring import handing_back_planner_failures  # noqa: E402
from pai_call import answer_message, user_message  # noqa: E402
from pai_planner import (  # noqa: E402
    CallHooks,
    PaiSession,
    PlannerClient,
    PlannerDeps,
    PlannerModel,
    PlannerParts,
    ToolWiring,
    genai_client,
    google_model,
    planner_agent,
)
from planner.agent import BRIEF_STATE_KEY, load_pending_ask  # noqa: E402
from planner.plan import current_recorded_identity  # noqa: E402
from planner.settings import AgentSettings  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.usage import UsageLimits  # noqa: E402
from translator.tool import build_translate  # noqa: E402

# Pydantic AI caps a run at 50 requests by default; the turn budget's 30-call cap is the
# bound here, as in production.
_NO_REQUEST_LIMIT = UsageLimits(request_limit=None)

SESSION_ID = "pai-planner-entry"


@dataclass
class PaiRunner:
    """The Pydantic AI runs of one entry, over one `PaiSession`."""

    agent: Agent[PlannerDeps, Any]
    deps: PlannerDeps

    async def run(self, brief: str) -> None:
        """Run the planner on `brief`, as `ctx.run_node` runs it, and add the run's messages
        to the session. Every model request streams. A tool that ends the run stops it
        before the next model request, where ADK's `skip_summarization` ends the turn.
        """
        session = self.deps.session
        history = session.history()
        self.deps.ended = False
        tail: list[ModelMessage] = []
        async with self.agent.iter(
            brief,
            message_history=history,
            deps=self.deps,
            usage_limits=_NO_REQUEST_LIMIT,
        ) as run:
            try:
                async for node in run:
                    if not Agent.is_model_request_node(node):
                        continue
                    if self.deps.ended:
                        # The tool returns the stopped request would have carried.
                        tail = [node.request]
                        break
                    async with node.stream(run.ctx) as stream:
                        async for _event in stream:
                            pass
            finally:
                session.record.extend([*run.all_messages()[len(history) :], *tail])


@dataclass
class _SessionView:
    """The session `PlannerSeam` reads through `ctx.session`: the live state, and no ADK
    events, since the Pydantic AI run appends none."""

    id: str
    state: dict[str, Any]
    events: list[Event] = field(default_factory=list)


@dataclass
class PaiNodeContext:
    """The ADK `Context` the graph's planner nodes and `PlannerSeam` read, over one plain
    state dict. `run_node` runs the Pydantic AI planner on the brief the seam left in state.
    ADK's `ctx.state` records a delta that later lands on the same session state; here both
    are one dict, so every write is visible at once, as the seam's own writes through
    `ctx.session.state` are."""

    runner: PaiRunner
    resume_inputs: dict[str, Any] | None = None
    session: _SessionView = field(init=False)

    def __post_init__(self) -> None:
        self.session = _SessionView(SESSION_ID, self.runner.deps.state)

    @property
    def state(self) -> dict[str, Any]:
        return self.runner.deps.state

    async def run_node(
        self,
        agent: Any,
        node_input: Any,
        *,
        run_id: str | None = None,
        override_branch: str | None = None,
    ) -> None:
        # The seam hands the run its brief in state and takes no node input; a run has no
        # branch or path of its own, since it reads only the messages it is given.
        del agent, run_id, override_branch
        assert node_input is None
        await self.runner.run(self.state[BRIEF_STATE_KEY])


async def _drain(events: AsyncIterator[Any] | Any) -> list[Any]:
    if hasattr(events, "__aiter__"):
        return [event async for event in events]
    return list(events)


@dataclass(frozen=True)
class RecordedEvent:
    """The function calls and responses of one step of the record, as
    `entry.tool_calls` reads an ADK event for them. It is the record's view only: no tool
    reads it."""

    calls: tuple[types.FunctionCall, ...] = ()
    responses: tuple[types.FunctionResponse, ...] = ()

    def get_function_calls(self) -> list[types.FunctionCall]:
        return list(self.calls)

    def get_function_responses(self) -> list[types.FunctionResponse]:
        return list(self.responses)


def recorded_events(messages: Sequence[ModelMessage]) -> list[RecordedEvent]:
    """The planner's calls and the responses to them, and the user's answers, in order."""
    events: list[RecordedEvent] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            calls = tuple(
                types.FunctionCall(
                    id=part.tool_call_id, name=part.tool_name, args=part.args_as_dict()
                )
                for part in message.parts
                if isinstance(part, ToolCallPart)
            )
            if calls:
                events.append(RecordedEvent(calls=calls))
        elif isinstance(message, ModelRequest):
            responses = tuple(
                types.FunctionResponse(
                    id=part.tool_call_id,
                    name=part.tool_name,
                    response=part.content
                    if isinstance(part.content, dict)
                    else {"result": part.content},
                )
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            )
            if responses:
                events.append(RecordedEvent(responses=responses))
    return events


@dataclass
class PaiEntry:
    """What one entry built, kept for checks that read past its record."""

    run: EntryRun
    session: PaiSession
    model: PlannerModel
    agent: Agent[PlannerDeps, Any]
    seam: PlannerSeam


async def run_pai_entry(
    task: Task,
    recording: Mapping[str, Any],
    *,
    prefix_cache: PrefixCache | None,
    timeout_s: float = 300.0,
    client: Client | None = None,
    answers: Sequence[Mapping[str, Any]] = (),
    keep_thoughts: bool = False,
    translation: TranslationCalls | None = None,
) -> PaiEntry:
    """Run `task`'s planner entry once with the planner on Pydantic AI, as
    `entry.run_entry` runs it with the planner on ADK.

    `client` stands in for the google-genai client, as `validate.py`'s scripted model stands
    in for the planner's. `answers` are the user's replies to the entry's asks, in order, each
    resuming the entry in a request of its own with a fresh budget. `keep_thoughts` replays
    thought summaries as Pydantic AI does by default. With `translation`, production's
    translation runs on the recorded plan over those calls (`entry.run_entry`).
    """
    probe = Probe()
    if translation is not None:
        probe.translator = build_translate(
            surface=translation,
            resolve_entities=translation.resolve_entities,
            anchor_read=_no_anchor_read,
            describe_nodes=translation.describe_nodes,
        )
    surface = RecordedSurface(recording.get("tools") or {})
    resolver = RecordedResolver(recording["searches"])
    state = await orchestrator_state(task, resolver)
    if resolver.left():
        raise RecordingMismatchError(f"{task.id}: {resolver.left()} recorded searches went unasked")

    settings = Settings()
    model_name = AgentSettings().planner_model_name
    seam = PlannerSeam(settings=settings, surface=surface, translate=probe.translate)
    hooks = CallHooks(seam=seam, before=probe.before_model, after=probe.after_model)
    planner_client = PlannerClient(client or genai_client(model_name), hooks)
    model = PlannerModel(
        google_model(model_name, planner_client.client),
        client=planner_client,
        prefix_cache=prefix_cache,
    )
    agent = planner_agent(
        seam=seam,
        model=model,
        wiring=ToolWiring(surface=surface, translate=probe.translate),
        parts=PlannerParts.of(seam),
        keep_thoughts=keep_thoughts,
    )
    session = PaiSession(SESSION_ID, [user_message(task.question)])
    runner = PaiRunner(agent, PlannerDeps(state=state, session=session))
    ctx = PaiNodeContext(runner)
    draft = planner_node(planner=handing_back_planner_failures(probe.watch(seam)))
    # The conversation a studio turn runs in, holding the spec blocks the orchestrator's
    # searches returned (`entry.run_entry`).
    conversation = ConversationState(conversation_id=SESSION_ID)
    conversation.turn_specs = dict(state.get(ORCHESTRATOR_SPECS_KEY) or {})
    pending = list(answers)
    budgets: list[TurnBudget] = []
    error: str | None = None
    timed_out = False

    async def request() -> None:
        """One request's nodes, under a fresh budget. A resume runs the planner node alone."""
        budgets.append(TurnBudget.from_settings(settings))
        budget_token = current_turn_budget.set(budgets[-1])
        identity_token = current_recorded_identity.set(conversation.recorded_identity)
        try:
            if ctx.resume_inputs is None:
                routed = await _drain(route(ctx, None))
                await _drain(plan_start(ctx, routed[-1].output))
            await _drain(draft(ctx, {}))
        finally:
            current_recorded_identity.reset(identity_token)
            current_turn_budget.reset(budget_token)

    async def drive() -> None:
        await request()
        while probe.plan is None and pending and (ask := load_pending_ask(state)) is not None:
            answer = dict(pending.pop(0))
            session.record.append(answer_message(ask.call_id, answer))
            # ADK resumes a node with the response's `result` when it has one.
            ctx.resume_inputs = {ask.call_id: answer.get("result", answer)}
            await request()

    started = time.perf_counter()
    limit = asyncio.timeout(timeout_s)
    try:
        async with limit:
            await drive()
    except Exception as exc:
        timed_out = limit.expired()
        if not timed_out:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"[:3000]
    wall_s = time.perf_counter() - started
    run = EntryRun(
        events=cast(list[Event], recorded_events(session.record)),
        state=dict(state),
        probe=probe,
        inputs=inputs_digest(task, recording),
        budget_closed=next(
            (str(budget.closed["code"]) for budget in reversed(budgets) if budget.closed), None
        ),
        surface_misses=list(surface.misses),
        error=error,
        timed_out=timed_out,
        wall_s=wall_s,
    )
    return PaiEntry(run=run, session=session, model=model, agent=agent, seam=seam)
