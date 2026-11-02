"""The `orchestrate` node on a Pydantic AI orchestrator.

`pai_orchestrate_node` takes the place of `harness.graph.orchestrate.orchestrate_node`
in the root workflow. It runs a Pydantic AI `Agent` with the ADK orchestrator's
instruction and tools, and leaves the rest of the turn to ADK: `route`, `ask`,
the planner, `approve`, `execute` and `plan_done` are dev's nodes.

- The agent's history is a Pydantic AI `Conversation`, kept in session state
  under `CONVERSATION_KEY` while the turn waits on a hand-off, and cleared when
  the agent answers. A `plan` or `ask_user` call that records its hand-off
  (`harness.orchestrator_tools._pend`, a result holding `HANDED_OFF`) raises
  `CallDeferred`, so the run ends with `DeferredToolRequests`. When `ask` or
  `plan_done` hands back, the node loads the conversation and answers that call
  with `DeferredToolResults`: the hand-back's `response()` becomes the call's
  result, as dev's `_call_result` makes it.
- The agent's events reach Studio as the ADK events dev's `LlmAgent` makes,
  authored `dfo_orchestrator` on its branch: a partial event per text or thought
  chunk, then the model call's whole content, then one event holding every tool
  result of that call with the state the tools wrote. `studio_translate` reads
  them unchanged.
- The tools are dev's ADK `FunctionTool`s, declared to Pydantic AI with the JSON
  schema ADK declares (`Tool.from_schema`) and run through `FunctionTool.run_async`
  with an ADK `Context` of their own, as ADK's tool caller builds one. A tool that
  raises is answered by the orchestrator's own `on_tool_error_callback`
  (`orchestrator.tool_failures.answering_tool_failures`).

`Variant` names the controls, each built to fail one pass line.
"""

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from google.adk import Context, Event
from google.adk.agents import LlmAgent
from google.adk.events import EventActions
from google.adk.tools import FunctionTool
from google.genai import types
from harness.graph.orchestrate import AgentHandBack
from harness.graph.state import (
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_STATUS_FAILURE_KEY,
    ORCHESTRATOR_STATUS_KEY,
)
from harness.orchestrator_tools import HANDED_OFF
from harness.turn_events import _turn_question
from orchestrator.agent import _DIGESTS, AGENT_NAME, SEEDED_STATE_KEY, TODAY_STATE_KEY, _utc_today
from pydantic_ai import (
    Agent,
    CallDeferred,
    Conversation,
    ConversationTypeAdapter,
    DeferredToolRequests,
    RunContext,
    Tool,
)
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import DeferredToolResults

# The orchestrator's paused `Conversation` as JSON; None once the agent answers.
# The `orchestrator:` prefix keeps it past `GraphState`'s validation.
CONVERSATION_KEY = "orchestrator:conversation"
# The JSON schema Pydantic AI is given for a declaration ADK sends with no parameters.
EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}
# The text of the leak `Variant.LEAK` writes into the session.
LEAK_SENTINEL = "ORCHESTRATOR-LEAKED-SENTINEL"


class Variant(StrEnum):
    LIVE = "live"
    # Sends no partial event: the reply reaches Studio whole, at the call's end.
    BUFFERED = "buffered"
    # Ignores the stored conversation on a hand-back and starts again from the
    # user's message.
    RESTART = "restart"
    # Answers the open call with its own record and sends the hand-back as the
    # user's next message.
    HAND_BACK_AS_MESSAGE = "hand_back_as_message"
    # Also writes `LEAK_SENTINEL` into the brief of the pending `plan` hand-off,
    # the state the planner reads.
    LEAK_STATE = "leak_state"
    # Also writes `LEAK_SENTINEL` into the session as the agent's own text on no
    # branch, as an event the node forgot to branch would be.
    LEAK_EVENT = "leak_event"


@dataclass(frozen=True)
class OrchestratorParts:
    """What the Pydantic AI agent takes from the ADK orchestrator: its instruction,
    its function tools and their declarations in ADK's order, its tool error
    callback and its thinking config."""

    instruction: str
    tools: tuple[FunctionTool, ...]
    declarations: tuple[types.FunctionDeclaration, ...]
    on_tool_error: Callable[..., dict[str, Any] | None] | None
    thinking: types.ThinkingConfig | None

    @classmethod
    def of(cls, agent: LlmAgent) -> "OrchestratorParts":
        assert isinstance(agent.instruction, str), "the orchestrator's instruction is static"
        tools = tuple(tool for tool in agent.tools if isinstance(tool, FunctionTool))
        declarations = []
        for tool in tools:
            declaration = tool._get_declaration()
            assert declaration is not None, f"{tool.name} declares nothing"
            declarations.append(declaration)
        (on_error, *_) = agent.canonical_on_tool_error_callbacks or [None]
        config = agent.generate_content_config
        return cls(
            instruction=agent.instruction,
            tools=tools,
            declarations=tuple(declarations),
            on_tool_error=on_error,
            thinking=config.thinking_config if config else None,
        )

    def model_settings(self) -> ModelSettings:
        """The thinking config the ADK agent sends, as Gemini model settings."""
        if self.thinking is None:
            return ModelSettings()
        thinking = self.thinking.model_dump(mode="json", exclude_none=True)
        return ModelSettings(google_thinking_config=thinking)  # ty: ignore[invalid-key]


@dataclass
class NodeDeps:
    """One node entry: the node's context, each tool call's own context, and the
    hand-off record of each call that deferred."""

    ctx: Context
    tool_contexts: dict[str, Context] = field(default_factory=dict)
    deferred: dict[str, dict[str, Any]] = field(default_factory=dict)


def _tool_body(
    tool: FunctionTool, on_error: Callable[..., dict[str, Any] | None] | None
) -> Callable[..., Any]:
    async def call(run: RunContext[NodeDeps], **args: Any) -> Any:
        call_id = run.tool_call_id
        assert call_id is not None
        # As `google.adk.flows.llm_flows.tools._caller._create_tool_context` builds it.
        context = Context(run.deps.ctx._invocation_context, function_call_id=call_id)
        run.deps.tool_contexts[call_id] = context
        try:
            result = await tool.run_async(args=args, tool_context=context)
        except Exception as error:
            answered = on_error(tool, args, context, error) if on_error is not None else None
            if answered is None:
                raise
            result = answered
        if isinstance(result, dict) and result.get(HANDED_OFF):
            run.deps.deferred[call_id] = result
            raise CallDeferred(metadata={"record": result})
        return result

    call.__name__ = tool.name
    return call


def build_agent(
    model: Model, parts: OrchestratorParts, *, answer_failures: bool = True
) -> Agent[NodeDeps, str | DeferredToolRequests]:
    """The orchestrator as a Pydantic AI agent. `answer_failures` off is O4's
    control: a tool that raises fails the run."""
    on_error = parts.on_tool_error if answer_failures else None
    tools = [
        Tool.from_schema(
            _tool_body(tool, on_error),
            name=declaration.name or tool.name,
            description=declaration.description,
            json_schema=(
                dict(declaration.parameters_json_schema)
                if declaration.parameters_json_schema is not None
                else EMPTY_PARAMETERS
            ),
            takes_ctx=True,
        )
        for tool, declaration in zip(parts.tools, parts.declarations, strict=True)
    ]
    return Agent(
        model,
        instructions=parts.instruction,
        tools=tools,
        deps_type=NodeDeps,
        output_type=[str, DeferredToolRequests],
        model_settings=parts.model_settings(),
        name=AGENT_NAME,
    )


# ---------------------------------------------------------------------------
# Where an entry starts
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    prompt: Sequence[str] | None
    conversation: Conversation
    results: DeferredToolResults | None


def first_prompt(ctx: Context, today: Callable[[], date]) -> list[str]:
    """The turn's first request, as `orchestrator.agent.prepend_digests` builds
    dev's contents: the date and digests, the user's message, and the seed's
    hand-back after it. Each is one part of one user prompt."""
    lines = []
    for key, label in _DIGESTS:
        digest = today().isoformat() if key == TODAY_STATE_KEY else ctx.state.get(key)
        if digest:
            lines.append(f"{label}: {digest}")
    question = _turn_question(ctx)  # ty: ignore[invalid-argument-type]
    text = "".join(part.text or "" for part in (question.parts or ())) if question else ""
    seeded = ctx.state.get(SEEDED_STATE_KEY)
    return [part for part in ("\n\n".join(lines), text, seeded) if part]


def _entry(
    ctx: Context, hand_back: AgentHandBack | None, variant: Variant, today: Callable[[], date]
) -> _Entry:
    """Continue the stored conversation when `hand_back` answers one of its calls;
    otherwise start the turn's conversation.

    Raises:
        RuntimeError: If no call of the agent's to the hand-back's tool waits on it.
    """
    answers = hand_back.answers if hand_back is not None else None
    if hand_back is None or answers is None or variant is Variant.RESTART:
        return _Entry(first_prompt(ctx, today), Conversation(conversation_id=ctx.session.id), None)
    stored = ctx.state.get(CONVERSATION_KEY)
    conversation = ConversationTypeAdapter.validate_json(stored) if stored else None
    requests = conversation.deferred_tool_requests if conversation else None
    call = next(
        (c for c in reversed(requests.calls) if c.tool_name == answers) if requests else (),
        None,
    )
    if conversation is None or requests is None or call is None:
        raise RuntimeError(f"no {answers} call of {AGENT_NAME} waits on this hand-back")
    if variant is Variant.HAND_BACK_AS_MESSAGE:
        record = requests.metadata.get(call.tool_call_id, {}).get("record", {})
        message = "\n".join([hand_back.text, json.dumps(hand_back.body)])
        return _Entry(
            [message], conversation, DeferredToolResults(calls={call.tool_call_id: record})
        )
    return _Entry(
        None, conversation, DeferredToolResults(calls={call.tool_call_id: hand_back.response()})
    )


# ---------------------------------------------------------------------------
# Pydantic AI events as the ADK events dev's agent makes
# ---------------------------------------------------------------------------


def _streamed_part(event: object) -> types.Part | None:
    """A text or thought chunk of a model call in flight, as a part."""
    if isinstance(event, PartStartEvent):
        if isinstance(event.part, TextPart) and event.part.content:
            return types.Part(text=event.part.content)
        if isinstance(event.part, ThinkingPart) and event.part.content:
            return types.Part(text=event.part.content, thought=True)
    if isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
            return types.Part(text=event.delta.content_delta)
        if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
            return types.Part(text=event.delta.content_delta, thought=True)
    return None


def adk_parts(response: ModelResponse) -> list[types.Part]:
    """A finished model call's thought, text and calls as ADK parts."""
    parts: list[types.Part] = []
    for part in response.parts:
        if isinstance(part, ThinkingPart):
            signature = part.signature.encode() if part.signature else None
            parts.append(types.Part(text=part.content, thought=True, thought_signature=signature))
        elif isinstance(part, TextPart):
            parts.append(types.Part(text=part.content))
        elif isinstance(part, ToolCallPart):
            call = types.FunctionCall(
                id=part.tool_call_id, name=part.tool_name, args=part.args_as_dict()
            )
            parts.append(types.Part(function_call=call))
    return parts


def _model_event(parts: list[types.Part], *, partial: bool = False) -> Event:
    return Event(
        content=types.Content(role="model", parts=parts), partial=partial, branch=AGENT_NAME
    )


def _body(part: object) -> dict[str, Any]:
    if isinstance(part, RetryPromptPart):
        return {"error": part.model_response()}
    content = getattr(part, "content", None)
    return content if isinstance(content, dict) else {"result": content}


def _responses_event(
    response: ModelResponse, returned: dict[str, dict[str, Any]], deps: NodeDeps
) -> Event | None:
    """Every result of `response`'s calls in one event, in call order, with the
    state each tool wrote, as ADK merges a model call's parallel responses."""
    parts: list[types.Part] = []
    actions = EventActions()
    for call in response.tool_calls:
        body = returned.get(call.tool_call_id) or deps.deferred.get(call.tool_call_id)
        if body is None:
            continue
        part = types.Part.from_function_response(name=call.tool_name, response=body)
        assert part.function_response is not None
        part.function_response.id = call.tool_call_id
        parts.append(part)
        context = deps.tool_contexts.get(call.tool_call_id)
        if context is not None:
            actions.state_delta.update(context.actions.state_delta)
            actions.skip_summarization = (
                actions.skip_summarization or context.actions.skip_summarization
            )
    if not parts:
        return None
    return Event(
        content=types.Content(role="user", parts=parts), actions=actions, branch=AGENT_NAME
    )


async def _relay(
    agent: Agent[NodeDeps, str | DeferredToolRequests],
    deps: NodeDeps,
    entry: _Entry,
    *,
    partials: bool,
) -> AsyncIterator[Event | AgentRunResult]:
    """The run's events as ADK events, as they happen, then its result. The run
    happens in this generator's task, so cancelling the node cancels it."""
    async with agent.iter(
        entry.prompt,
        deps=deps,
        conversation=entry.conversation,
        deferred_tool_results=entry.results,
    ) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        part = _streamed_part(event)
                        if part is not None and partials:
                            yield _model_event([part], partial=True)
            elif Agent.is_call_tools_node(node):
                response = node.model_response
                # A run continued with deferred results first answers the model
                # call the last entry ended on, already sent as an event.
                resumed = entry.results is not None and node.tool_call_results is not None
                if not resumed and (parts := adk_parts(response)):
                    yield _model_event(parts)
                returned: dict[str, dict[str, Any]] = {}
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, FunctionToolResultEvent):
                            returned[event.tool_call_id] = _body(event.part)
                answered = _responses_event(response, returned, deps)
                if answered is None:
                    continue
                if not resumed:
                    yield answered
                    continue
                # The hand-back, stored as dev's node stores it: authored `user`
                # on the agent's branch, which no agent's context reads as its own.
                author, deps.ctx.event_author = deps.ctx.event_author, "user"
                try:
                    yield answered
                finally:
                    deps.ctx.event_author = author
        assert run.result is not None
        yield run.result


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


def pai_orchestrate_node(
    *,
    orchestrator: LlmAgent,
    model: Model,
    variant: Variant = Variant.LIVE,
    answer_failures: bool = True,
    today: Callable[[], date] = _utc_today,
) -> Callable[[Context, Any], AsyncIterator[Event]]:
    """The `orchestrate` node, running the Pydantic AI version of `orchestrator`
    on `model`. Reached as dev's node is: from START with no input or a seed
    hand-back, and from `ask` and `plan_done` with the hand-back that answers the
    agent's `ask_user` or `plan` call."""
    agent = build_agent(model, OrchestratorParts.of(orchestrator), answer_failures=answer_failures)

    async def orchestrate(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        hand_back: AgentHandBack | None = None
        if node_input is None:
            yield Event(custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your question"})
        else:
            hand_back = AgentHandBack.model_validate(node_input)
            if hand_back.answers is None:
                ctx.state[SEEDED_STATE_KEY] = hand_back.text
            status: dict[str, Any] = {ORCHESTRATOR_STATUS_KEY: hand_back.status}
            if hand_back.failure is not None:
                status[ORCHESTRATOR_STATUS_FAILURE_KEY] = hand_back.failure
            yield Event(custom_metadata=status, branch="")
        entry = _entry(ctx, hand_back, variant, today)
        deps = NodeDeps(ctx=ctx)
        result: AgentRunResult | None = None
        author, ctx.event_author = ctx.event_author, AGENT_NAME
        try:
            async for item in _relay(agent, deps, entry, partials=variant is not Variant.BUFFERED):
                if isinstance(item, AgentRunResult):
                    result = item
                else:
                    yield item
            if variant is Variant.LEAK_EVENT:
                text = types.Part(text=LEAK_SENTINEL)
                yield Event(content=types.Content(role="model", parts=[text]), branch="")
        finally:
            ctx.event_author = author
        assert result is not None
        pending = ctx.state.get(ORCHESTRATOR_PENDING_KEY)
        if variant is Variant.LEAK_STATE and pending and pending.get("kind") == "plan":
            brief = f"{pending['brief']} {LEAK_SENTINEL}"
            ctx.state[ORCHESTRATOR_PENDING_KEY] = {**pending, "brief": brief}
        paused = isinstance(result.output, DeferredToolRequests)
        ctx.state[CONVERSATION_KEY] = (
            ConversationTypeAdapter.dump_json(result.conversation).decode() if paused else None
        )
        # `route` reads the pending action from state. The empty branch clears
        # the agent's branch for the nodes after this one, as dev's status event does.
        yield Event(output="", branch="")

    return orchestrate
