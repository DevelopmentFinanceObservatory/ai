"""The planner node of the real Studio turn with the planner on Pydantic AI.

The node is dev's own (`harness.graph.planning.planner_node`). Only the seam it awaits
differs: `PaiPlannerSeam.__call__` runs `PlannerSeam.__call__` unchanged over a context
whose `run_node` runs the Pydantic AI agent where ADK would run the planner `LlmAgent`.

That run is a child node of the planner node, named and branched as the ADK planner is
(`planner.agent.AGENT_NAME`, `harness.planner_seam._PLANNER_BRANCH`). It yields events as
the agent loop proceeds, so each reaches the runner's stream when it happens, as the
planner's own events reach it under ADK:

- each tool call and its response, as ADK appends them to the session. `harness.
  studio_translate` reads the `propose_plan` pair for the "Checking the plan against the data"
  status line (`_plan_path_frames`), and `planner.agent._session_steps`, which `revise` reads
  through `AdkCall`, reads the ask's pair;
- each progress report the seam's tool callback makes (`custom_metadata["progress"]`), which
  dev's node holds until the draft is done.

Pydantic AI tools write the session state dict and nothing else, where ADK carries each
tool's writes on its response event's `state_delta`, durable once the event is appended.
The tool bodies write through `pai_call.RecordingState`, which notes each key written, and
the response event of the tools that wrote them carries those keys (`PlannerDeps.
drain_writes`), so a tool's writes are stored when it ends, a Stop mid-planning included.

The planner's view of what the user was asked and answered (`pai_call.PaiCall`) is the
planner events of the session, which this node and ADK's planner both append
(`planner.agent._session_steps`), then the live run's own steps. A turn can pass between
the two planners at any ask, and no record of the node's own is stored.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Container
from dataclasses import dataclass, field
from typing import Any

from google.adk import Context, Event
from google.adk.events import EventActions
from google.adk.workflow import node
from google.genai import types
from harness.planner_seam import PlannerSeam
from pai_call import user_message
from pai_entry import _NO_REQUEST_LIMIT, PaiRunner
from pai_planner import (
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
from planner.agent import AGENT_NAME, BRIEF_STATE_KEY, _session_steps, load_pending_ask
from planner.call import Step
from planner.settings import AgentSettings
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

logger = logging.getLogger(__name__)

# `harness.planner_seam._PLANNER_BRANCH`, the branch every planner run carries.
PLANNER_BRANCH = "dfo_planner"


@dataclass
class Feed:
    """The events the Pydantic AI run owes the runner's stream: reports queued by the seam's
    tool callback, drained by the child node between steps of the agent loop."""

    pending: list[Event] = field(default_factory=list)

    def progress(self, report: dict[str, Any]) -> None:
        self.pending.append(Event(custom_metadata={"progress": report}))

    def drain(self) -> list[Event]:
        drained, self.pending = self.pending, []
        return drained


def call_events(response: ModelResponse, tools: Container[str] | None) -> list[Event]:
    """The function-call event ADK emits for a response's calls to `tools` (all when `None`)."""
    parts = [
        types.Part(
            function_call=types.FunctionCall(
                id=part.tool_call_id, name=part.tool_name, args=part.args_as_dict()
            )
        )
        for part in response.parts
        if isinstance(part, ToolCallPart) and (tools is None or part.tool_name in tools)
    ]
    if not parts:
        return []
    return [Event(content=types.Content(role="model", parts=parts), partial=False)]


def response_events(
    request: ModelRequest, drain_writes: Callable[[], dict[str, Any]], tools: Container[str] | None
) -> list[Event]:
    """The function-response event ADK emits for the calls to `tools` (all when `None`) a
    request answers, carrying in its `state_delta` the keys the tools wrote
    (`PlannerDeps.drain_writes`): the plan a `propose_plan` recorded, which the translator
    reads, and every other key the next reader needs durable."""
    parts = [
        types.Part(
            function_response=types.FunctionResponse(
                id=part.tool_call_id, name=part.tool_name, response=dict(part.content)
            )
        )
        for part in request.parts
        if isinstance(part, ToolReturnPart)
        and (tools is None or part.tool_name in tools)
        and isinstance(part.content, dict)
    ]
    if not parts:
        return []
    return [
        Event(
            content=types.Content(role="user", parts=parts),
            actions=EventActions(state_delta=drain_writes()),
        )
    ]


@dataclass
class FeedingRunner(PaiRunner):
    """`PaiRunner` that yields the planner's events as its loop proceeds."""

    feed: Feed = field(default_factory=Feed)
    emit: bool = True
    tools: frozenset[str] | None = None

    async def events(self, brief: str) -> AsyncIterator[Event]:
        """Run the planner on `brief` as `PaiRunner.run` does, yielding each event the run
        owes the stream: a `propose_plan` call before its tool runs, its response and the
        progress reports once the tools of the response have run."""
        session = self.deps.session
        self.deps.ended = False
        async with self.agent.iter(
            brief,
            message_history=session.history(),
            deps=self.deps,
            usage_limits=_NO_REQUEST_LIMIT,
            conversation_id=session.id,
        ) as run:
            try:
                async for step in run:
                    if Agent.is_call_tools_node(step):
                        for event in (
                            call_events(step.model_response, self.tools) if self.emit else ()
                        ):
                            yield event
                        continue
                    if not Agent.is_model_request_node(step):
                        continue
                    if self.emit:
                        for event in response_events(
                            step.request, self.deps.drain_writes, self.tools
                        ):
                            yield event
                        for event in self.feed.drain():
                            yield event
                    if self.deps.ended:
                        break
                    async with step.stream(run.ctx) as stream:
                        async for _event in stream:
                            pass
            except UnexpectedModelBehavior as exc:
                # As on ADK, a response Pydantic AI refuses (an empty stream, one the
                # provider's filter blocked) ends the run with no plan; the seam refuses
                # with `plan_not_ready`.
                logger.warning("planner response refused by Pydantic AI: %s", exc)


class PaiCtx:
    """The `Context` `PlannerSeam.__call__` reads, with `run_node` running the Pydantic AI
    agent as a child node in place of the planner's `LlmAgent`. Every other attribute is the
    real context's."""

    def __init__(self, ctx: Context, runner: FeedingRunner) -> None:
        self._ctx = ctx
        self._runner = runner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)

    async def run_node(
        self,
        agent: Any,
        node_input: Any,
        *,
        run_id: str | None = None,
        override_branch: str | None = None,
    ) -> None:
        del agent
        assert node_input is None
        runner, brief = self._runner, self._ctx.session.state[BRIEF_STATE_KEY]

        async def planner(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
            del node_input
            # As `LlmAgent` does, so the events are the planner's (`agents/llm_agent.py`).
            ctx.event_author = AGENT_NAME
            async for event in runner.events(brief):
                yield event

        await self._ctx.run_node(
            node(planner, name=AGENT_NAME), None, run_id=run_id, override_branch=override_branch
        )


def user_text(ctx: Context) -> str:
    """The message the user typed into the turn: the first user-authored event of the
    session that carries text and no branch (`planner.agent._user_message`)."""
    for event in ctx.session.events:
        if event.author == "user" and not event.branch and event.content and event.content.parts:
            text = "".join(part.text or "" for part in event.content.parts).strip()
            if text:
                return text
    raise AssertionError("the turn's session holds no message of the user's")


@dataclass
class PaiPlannerSeam:
    """`harness.graph.seams.PlannerModel` over `PlannerSeam`, the planner on Pydantic AI.

    `events` has the run yield the planner's events to the stream; without it the seam's
    progress reports go to the node's own queue, which it relays once the draft is done.
    `feed_tools` names the tools whose call and response events the run yields: all by
    default, as ADK's session holds them. The translator reads only `propose_plan`'s.
    `runner` is the agent loop's class.
    """

    seam: PlannerSeam
    agent: Agent[PlannerDeps, Any]
    events: bool = True
    runner: type[FeedingRunner] = FeedingRunner
    feed_tools: frozenset[str] | None = None

    async def __call__(
        self,
        *,
        ctx: Context,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        session = PaiSession(
            ctx.session.id, [user_message(user_text(ctx))], earlier=self._earlier(ctx)
        )
        feed = Feed()
        runner = self.runner(
            self.agent,
            PlannerDeps(state=ctx.session.state, session=session),
            feed,
            self.events,
            self.feed_tools,
        )
        return await self.seam(
            ctx=PaiCtx(ctx, runner),  # ty: ignore[invalid-argument-type]
            on_progress=feed.progress if self.events else on_progress,
            **kwargs,
        )

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        return await self.seam.revise(**kwargs)

    def _earlier(self, ctx: Context) -> list[Step]:
        """The steps of the turn so far: the session's planner events, read as ADK's own
        planner reads them. A call that resumes an ask the events hold no answer to yet
        adds the answer, as ADK's interrupt route adds its response to the session."""
        steps = list(_session_steps(ctx.session.events))
        asked = load_pending_ask(ctx.session.state)
        if not (ctx.resume_inputs and asked is not None and asked.call_id in ctx.resume_inputs):
            return steps
        if any(step.kind == "answer" and step.call_id == asked.call_id for step in steps):
            return steps
        args = next(
            (s.args for s in reversed(steps) if s.kind == "call" and s.call_id == asked.call_id),
            None,
        )
        steps.append(
            Step("answer", asked.call_id, args, {"result": ctx.resume_inputs[asked.call_id]})
        )
        return steps


def pai_planner(
    seam: PlannerSeam,
    *,
    client: Any,
    surface: Any,
    translate: Any,
    prefix_cache: Any = None,
    **options: Any,
) -> PaiPlannerSeam:
    """The planner of `seam` on Pydantic AI, talking to the google-genai `client`."""
    model_name = AgentSettings().planner_model_name
    planner_client = PlannerClient(client or genai_client(model_name), CallHooks(seam=seam))
    model = PlannerModel(
        google_model(model_name, planner_client.client),
        client=planner_client,
        prefix_cache=prefix_cache,
    )
    agent = planner_agent(
        seam=seam,
        model=model,
        wiring=ToolWiring(surface=surface, translate=translate),
        parts=PlannerParts.of(seam),
    )
    return PaiPlannerSeam(seam, agent, **options)
