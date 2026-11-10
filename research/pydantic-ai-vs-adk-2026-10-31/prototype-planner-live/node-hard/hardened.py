"""The planner node's hard cases closed in subclasses of `node/` and `agent/`.

Each class changes one behaviour and names the change `agent/` or `node/` needs for it.
Every fix has a switch, so a test can run the planner without it and show the check fail.

- `HoldingClient`: a model call's chunks reach Pydantic AI after `bounded` has settled the
  call, as ADK's aggregator hands its consumer a function call only when the stream ends.
  A call the budget cuts mid-stream delivers its closing sentence alone.
- `hardened_agent`: an unknown tool name is answered with ADK's tool-not-found text and
  does not end the run until the turn's model-call cap does.
- `HardRunner`: a response whose model call Pydantic AI refuses (`UnexpectedModelBehavior`)
  ends the run as a response with no function call ends it on ADK; each tool response event
  carries the state its tools wrote, durable when the tool ends as on ADK.
- `HardSeam._record`: the record a run reads of earlier runs is the planner events in the
  session, which both planners append, so a turn can pass between them.
- `hardened_agent(instrument=True)`: Pydantic AI's `Instrumentation` capability, which gives
  the planner generation, tool and agent spans.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

import pai_node
from google.adk import Context, Event
from google.adk.events import EventActions
from google.adk.flows.llm_flows.tools._error_handler import build_tool_not_found_response
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from harness.planner_seam import PlannerSeam
from harness.turn_budget import CLOSED_BY_KEY, bounded
from harness.turn_events import PLANNER_ASK
from pai_call import answer_message, user_message
from pai_entry import _NO_REQUEST_LIMIT
from pai_node import (
    FeedingRunner,
    PaiPlannerSeam,
    call_events,
    response_events,
)
from pai_planner import (
    _RAW_CHUNK,
    OUTPUT_TYPE,
    THINKING_SETTINGS,
    CallHooks,
    PlannerClient,
    PlannerDeps,
    PlannerModel,
    PlannerParts,
    ToolWiring,
    _as_chunk,
    drop_unsigned_thoughts,
    genai_client,
    google_model,
    planner_tools,
)
from planner.agent import AGENT_NAME, _session_steps, load_pending_ask
from planner.settings import AgentSettings
from pydantic_ai import Agent
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.capabilities.process_history import ProcessHistory
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)

logger = logging.getLogger(__name__)

# The model-call cap of one request (`harness.settings.Settings.turn_model_calls_max`) is
# what ends a run that keeps calling a tool that does not exist, as on ADK. Pydantic AI
# charges each such call against `retries`, so the allowance sits above the cap.
RETRIES = 1000


# --- A call settles before Pydantic AI reads it ------------------------------------------


class HoldingClient(PlannerClient):
    """`PlannerClient` that hands Pydantic AI a call's chunks once `bounded` has finished
    with the call.

    ADK's `Gemini` aggregates a streamed call and gives its consumer a function call with
    the final response, after the stream ended, so the budget's cut reaches a call whose
    function call was streamed but never delivered, and the session keeps the closing
    sentence alone. `PlannerClient` passes each chunk to Pydantic AI at once, so a cut after
    a function-call chunk would still run the tool. Holding the chunks gives Pydantic AI
    what ADK's consumer gets: the whole call, or the closing response of a cut one.

    The cost: Pydantic AI sees no partial text of a call before it ends. The planner's
    text is never shown while it streams, and every chunk still passes through `bounded`
    as it arrives, so the first-chunk deadline, the gaps and the model time are metered as
    before.
    """

    hold: bool = True

    def __init__(self, *args: Any, hold: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hold = hold

    async def _raw(self, *, model: str, contents: Any, config: Any) -> AsyncIterator[LlmResponse]:
        """`PlannerClient._raw` with each chunk's usage on the response, so `bounded` stamps
        `dfo.model.thoughts_token_count` on the call's span as it does for ADK's responses."""
        stream = await self.sent(model=model, contents=contents, config=config)
        async with aclosing(stream) as chunks:
            async for chunk in chunks:
                content = chunk.candidates[0].content if chunk.candidates else None
                yield LlmResponse(
                    content=content,
                    partial=True,
                    usage_metadata=chunk.usage_metadata,
                    custom_metadata={_RAW_CHUNK: chunk},
                )

    async def _stream(
        self, *, model: str, contents: Any, config: Any
    ) -> AsyncIterator[types.GenerateContentResponse]:
        if not self.hold:
            async for chunk in super()._stream(model=model, contents=contents, config=config):
                yield chunk
            return
        synthetic, self.pending.response = self.pending.response, None
        if synthetic is not None:
            yield _as_chunk(synthetic)
            return
        usage: types.GenerateContentResponseUsageMetadata | None = None
        closed = False
        held: list[types.GenerateContentResponse] = []
        responses = bounded(
            lambda: self._raw(model=model, contents=contents, config=config), stream=True
        )
        async with aclosing(responses) as stream:
            async for response in stream:
                metadata = response.custom_metadata or {}
                if _RAW_CHUNK in metadata:
                    chunk: types.GenerateContentResponse = metadata[_RAW_CHUNK]
                    usage = chunk.usage_metadata or usage
                    held.append(chunk)
                elif CLOSED_BY_KEY in metadata:
                    closed = True
                    self.hooks.after(None, response)
                    replaced = self.hooks.seam._after_planner_model(None, response)
                    # The session keeps the closing answer alone, whatever was streamed.
                    held = [_as_chunk(replaced or response)]
        if not closed:
            final = LlmResponse(usage_metadata=usage, partial=False)
            self.hooks.after(None, final)
            self.hooks.seam._after_planner_model(None, final)
        for chunk in held:
            yield chunk


# --- Unknown tool names ------------------------------------------------------------------


def adk_tool_not_found(
    tools: Sequence[BaseTool],
) -> Callable[[list[ModelMessage]], list[ModelMessage]]:
    """A history processor that gives a call to a tool the planner lacks the answer ADK
    gives it (`build_tool_not_found_response`) in place of Pydantic AI's retry text."""
    names = [tool.name for tool in tools]

    def process(messages: list[ModelMessage]) -> list[ModelMessage]:
        known = dict.fromkeys(names)
        rewritten: list[ModelMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                rewritten.append(
                    ModelRequest(
                        parts=[_not_found(part, known) for part in message.parts],
                        instructions=message.instructions,
                        metadata=message.metadata,
                    )
                )
            else:
                rewritten.append(message)
        return rewritten

    return process


def _not_found(part: Any, known: Mapping[str, Any]) -> Any:
    if isinstance(part, RetryPromptPart | ToolReturnPart) and part.tool_name not in known:
        text = part.content if isinstance(part.content, str) else ""
        retried = isinstance(part, RetryPromptPart) or part.retry is not None
        if retried and (not text or text.startswith("Unknown tool name")):
            body = build_tool_not_found_response(part.tool_name, known)  # type: ignore[arg-type]
            return ToolReturnPart(
                tool_name=part.tool_name, tool_call_id=part.tool_call_id, content=body
            )
    return part


def hardened_agent(
    *,
    seam: PlannerSeam,
    model: PlannerModel,
    wiring: ToolWiring,
    parts: PlannerParts,
    unknown_tools: bool = True,
    instrument: bool = True,
) -> Agent[PlannerDeps, str | None]:
    """`pai_planner.planner_agent` with Pydantic AI's tool retries raised to the turn's
    own cap and, with `unknown_tools`, ADK's answer to a name that is no tool."""
    processors: list[Any] = [ProcessHistory(drop_unsigned_thoughts)]
    if instrument:
        processors.append(Instrumentation())
    if unknown_tools:
        processors.append(ProcessHistory(adk_tool_not_found(parts.tools)))
    return Agent(
        model,
        instructions=parts.instruction,
        tools=planner_tools(seam, parts, wiring),
        deps_type=PlannerDeps,
        output_type=OUTPUT_TYPE,
        model_settings=THINKING_SETTINGS,
        capabilities=processors,
        name=AGENT_NAME,
        retries=RETRIES if unknown_tools else 1,
    )


# --- The run -----------------------------------------------------------------------------


@dataclass
class HardRunner(FeedingRunner):
    """`FeedingRunner` whose run ends as ADK's does when Pydantic AI refuses a response,
    and whose tool responses carry the state their tools wrote.

    `swallow`: an `UnexpectedModelBehavior` (an empty stream, a response the provider's
    filter blocked) ends the run with no plan, as ADK ends a run on a response with no
    function call; the seam then refuses with `plan_not_ready`.

    `conversation`: the run's `gen_ai.conversation.id` is the ADK session's id, as on ADK's
    spans, in place of the uuid Pydantic AI makes up.

    `deltas`: each function-response event carries `state_delta` for every key the
    response's tools changed, so the key is in the stored session when the tool ends, as
    ADK's tool response events carry it. `FeedingRunner` and `mirror_changes` write them
    when the seam call ends, which a Stop never reaches.
    """

    swallow: bool = True
    deltas: bool = True
    conversation: bool = True
    seen: dict[str, Any] = field(default_factory=dict)

    async def events(self, brief: str) -> AsyncIterator[Event]:
        session = self.deps.session
        history = session.history()
        self.deps.ended = False
        self.seen = copy.deepcopy(self.deps.state)
        tail: list[ModelMessage] = []
        async with self.agent.iter(
            brief,
            message_history=history,
            deps=self.deps,
            usage_limits=_NO_REQUEST_LIMIT,
            conversation_id=session.id if self.conversation else None,
        ) as run:
            try:
                async for step in run:
                    if Agent.is_call_tools_node(step):
                        for event in call_events(step.model_response, self.tools):
                            yield event
                        continue
                    if not Agent.is_model_request_node(step):
                        continue
                    for event in self._responses(step.request):
                        yield event
                    for event in self.feed.drain():
                        yield event
                    if self.deps.ended:
                        tail = [step.request]
                        break
                    async with step.stream(run.ctx) as stream:
                        async for _event in stream:
                            pass
            except UnexpectedModelBehavior as exc:
                if not self.swallow:
                    raise
                logger.warning("planner response refused by Pydantic AI: %s", exc)
            finally:
                session.record.extend([*run.all_messages()[len(history) :], *tail])

    def _responses(self, request: ModelRequest) -> list[Event]:
        events = response_events(request, self.deps.state, self.tools)
        if not self.deltas or not events:
            return events
        state = self.deps.state
        changed = {
            key: copy.deepcopy(value)
            for key, value in state.items()
            if key not in self.seen or self.seen[key] != value
        }
        self.seen = copy.deepcopy(state)
        (event,) = events
        assert event.actions is not None
        event.actions = EventActions(state_delta={**event.actions.state_delta, **changed})
        return [event]


# --- The record a resume reads -----------------------------------------------------------


def record_from_events(ctx: Context) -> list[ModelMessage]:
    """The message record of a turn the ADK planner ran, read from the session's events as
    `planner.agent._session_steps` reads them. The tool bodies read only steps from a
    record (`pai_call.PaiCall`), so each step becomes the message that yields it."""
    record: list[ModelMessage] = []
    asked: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(_session_steps(ctx.session.events)):
        match step.kind:
            case "message":
                record.append(user_message(pai_node.user_text(ctx)))
            case "call":
                asked[step.call_id or ""] = dict(step.args or {})
                record.append(
                    ModelResponse(
                        parts=[ToolCallPart(PLANNER_ASK, dict(step.args or {}), step.call_id or "")]
                    )
                )
            case "answer":
                record.append(
                    ModelRequest(
                        parts=[ToolReturnPart(PLANNER_ASK, dict(step.body), step.call_id or "")]
                    )
                )
            case _:
                record.append(
                    ModelRequest(
                        parts=[ToolReturnPart(step.name or "", dict(step.body), f"adk-{index}")]
                    )
                )
    return record


@dataclass
class HardSeam(PaiPlannerSeam):
    """`PaiPlannerSeam` whose record of the turn is the session's events.

    `record_from="events"`: every run, whichever planner ran the runs before it, reads
    its earlier steps from the planner events in the session (`record_from_events`), which
    ADK and the Pydantic AI node both append. The node's own `planner:conversation` is
    written as before and left unread. `"stored"` is the node's behaviour: the stored record
    when there is one, the user's message alone when not.
    """

    record_from: str = "events"

    def _record(self, ctx: Context) -> list[ModelMessage]:
        if self.record_from == "stored":
            return super()._record(ctx)
        record = record_from_events(ctx) or [user_message(pai_node.user_text(ctx))]
        asked = load_pending_ask(ctx.session.state)
        if ctx.resume_inputs and asked is not None and asked.call_id in ctx.resume_inputs:
            answer = answer_message(asked.call_id, {"result": ctx.resume_inputs[asked.call_id]})
            if not _answered(record, asked.call_id):
                record.append(answer)
        return record


def _answered(record: Sequence[ModelMessage], call_id: str) -> bool:
    return any(
        isinstance(message, ModelRequest)
        and any(
            isinstance(part, ToolReturnPart)
            and part.tool_call_id == call_id
            and part.tool_name == PLANNER_ASK
            and isinstance(part.content, dict)
            and "result" in part.content
            for part in message.parts
        )
        for message in record
    )


# --- The planner -------------------------------------------------------------------------


def hard_planner(
    seam: PlannerSeam,
    *,
    client: Any,
    surface: Any,
    translate: Any,
    prefix_cache: Any = None,
    hold: bool = True,
    unknown_tools: bool = True,
    swallow: bool = True,
    deltas: bool = True,
    record_from: str = "events",
    conversation: bool = True,
    instrument: bool = True,
    **options: Any,
) -> HardSeam:
    """`pai_node.pai_planner` with every fix of this module; each keyword turns one off."""
    model_name = AgentSettings().planner_model_name
    planner_client = HoldingClient(
        client or genai_client(model_name), CallHooks(seam=seam), hold=hold
    )
    model = PlannerModel(
        google_model(model_name, planner_client.client),
        client=planner_client,
        prefix_cache=prefix_cache,
    )
    agent = hardened_agent(
        seam=seam,
        model=model,
        wiring=ToolWiring(surface=surface, translate=translate),
        parts=PlannerParts.of(seam),
        unknown_tools=unknown_tools,
        instrument=instrument,
    )

    def runner(*args: Any) -> HardRunner:
        return HardRunner(*args, swallow=swallow, deltas=deltas, conversation=conversation)

    return HardSeam(seam, agent, runner=runner, record_from=record_from, **options)  # type: ignore[arg-type]
