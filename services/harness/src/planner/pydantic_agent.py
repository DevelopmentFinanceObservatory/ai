"""The Pydantic AI planner: its agent (`planner_agent`) and its run (`run_planner`).

The agent serves each `planner.tool_spec.ToolSpec` with the declaration the
ADK planner shows for it (`function_declaration`), and a call runs the spec's
body over a `planner.pydantic_call.PydanticCall` (`run_tool`), with the
model's arguments as sent. `run_planner` drives one run with `Agent.iter`
from the user's message and the brief, as the ADK planner's `single_turn`
run starts from the turn's question and the brief (`planner.agent.
prepend_brief`), and yields each model response and each call's answer, with
the state the call wrote, as the run makes them. A tool that ends the run (`planner.tools.EndRun`) stops it
before the next model request.

What the model is sent equals the ADK planner's request for the same turn,
apart from how Pydantic AI shapes a request on the wire: the system
instruction as a `Content`, one `Tool` per declaration, an empty parameters
schema on a tool that takes none, an explicit `AUTO` tool config,
`response_modalities`, its own HTTP headers, and no `automatic_continuation`,
which google-genai reads as on. The model must be built with
`planner_profile`, which sends each declaration's schema as given and adds
nothing to the instruction.
"""

from __future__ import annotations

import base64
import copy
import inspect
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from google.genai import types
from opentelemetry import trace
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Status, StatusCode
from pydantic_ai import Agent, InstrumentationSettings, RunContext, Tool
from pydantic_ai.capabilities import Instrumentation, ProcessHistory
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.profiles.google import GoogleModelProfile, google_model_profile
from pydantic_ai.usage import RequestUsage, UsageLimits

from planner.call import RecordingState, Step
from planner.pydantic_call import PydanticCall, as_function_response
from planner.settings import AGENT_THINKING_CONFIG
from planner.tool_spec import ToolSpec, function_declaration, run_tool

logger = logging.getLogger(__name__)

# The schema Pydantic AI declares a tool with when its declaration has none:
# Pydantic AI always sends `parameters_json_schema`.
_NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}

# The turn budget bounds a run's model calls (`harness.turn_budget`), so Pydantic
# AI's own default cap of 50 requests is lifted.
_NO_REQUEST_LIMIT = UsageLimits(request_limit=None)

# Pydantic AI charges a call to a tool the planner lacks against the tool retry
# budget and ends the run once it is spent. The ADK planner answers every such
# call and goes on, so the budget is lifted and the turn budget's call cap ends
# a run that keeps making them. The planner's own tools never ask for a retry:
# their arguments are not validated.
_UNLIMITED_RETRIES = sys.maxsize

# Carried by the request holding the user's message. Pydantic AI merges
# consecutive requests into one content unless their `instructions` differ,
# and the brief's request carries the agent's instruction, so this keeps the
# user's message and the brief two contents, as the ADK planner sends them.
# Only the latest request's instructions reach the model, so the model never
# reads this text.
_QUESTION_INSTRUCTIONS = "The user's message, which opened the turn."

# The `error.type` of a tool call whose answer is an error, ADK's.
_TOOL_ERROR = "TOOL_ERROR"

type ToolAnswered = Callable[[str, dict[str, Any]], None]
type ToolFailed = Callable[[str, BaseException], dict[str, Any]]


def planner_profile(model_name: str) -> GoogleModelProfile:
    """Pydantic AI's profile for the Gemini `model_name`, with the declarations
    sent as given (no JSON schema transformer) and nothing added to the
    instruction about system messages mid-conversation (inline system prompts
    supported)."""
    return GoogleModelProfile(
        {
            **(google_model_profile(model_name) or {}),
            "json_schema_transformer": None,
            "supports_inline_system_prompts": True,
        }
    )


@dataclass
class _Call:
    """One tool call of the run: its tool, the view it writes through, its
    answer once it has one, and whether it ended the run."""

    name: str
    view: RecordingState
    answer: dict[str, Any] | None = None
    writes: dict[str, Any] | None = None
    ended: bool = False


@dataclass
class PlannerDeps:
    """One run's dependencies, which every tool call of the run reads.

    `state` is the session's state. Each call writes it through a
    `RecordingState` of its own, so the keys one call wrote are told apart
    from another's while every call reads the others' values. `earlier` is
    the steps of the session before the run. `tool_answered` is told each
    tool's answer and `tool_failed` each tool that raised. `ended` says
    whether a tool ended the run.
    """

    state: MutableMapping[str, Any]
    earlier: Sequence[Step]
    tool_answered: ToolAnswered
    tool_failed: ToolFailed
    ended: bool = False
    _calls: dict[str, _Call] = field(default_factory=dict, init=False)

    def end_run(self) -> None:
        """The run's `planner.tools.EndRun`: no model request follows the current calls."""
        self.ended = True

    def _written(self, view: RecordingState) -> dict[str, Any]:
        """The keys written through `view`, with a copy of their values. A key
        deleted since holds no value and is left out."""
        return {key: copy.deepcopy(self.state[key]) for key in view.written if key in self.state}


@dataclass(frozen=True)
class PlannerResponse:
    """One model response of the run, as soon as it is received and before its calls run.

    `parts` are its parts as google-genai parts: each function call with its
    id, text, and each thought with `thought` set, each with its thought
    signature. `usage` is the request's usage as Pydantic AI reads it.
    `block_reason` is Gemini's reason for blocking the prompt, for a response
    that is a block and holds no candidate, which the ADK planner records as
    an event with that `error_code` and no content. A response with no parts
    and no block reason is one the ADK planner records no event for.
    """

    parts: tuple[types.Part, ...]
    usage: RequestUsage
    block_reason: str | None = None


@dataclass(frozen=True)
class PlannerToolAnswer:
    """One call's answer, as soon as the call has answered.

    `writes` is every state key the call wrote, with its value when it
    answered, and `ended` whether the call ended the run
    (`planner.tools.EndRun`). The answers of one response come in the order
    their calls finish.
    """

    name: str
    call_id: str
    answer: dict[str, Any]
    writes: dict[str, Any]
    ended: bool = False


type PlannerItem = PlannerResponse | PlannerToolAnswer


@dataclass(frozen=True)
class PlannerAgent:
    """The Pydantic AI planner (`planner_agent`): the agent and its tools' names,
    in the order they are declared."""

    agent: Agent[PlannerDeps, str | None]
    tools: tuple[str, ...]


def tool_not_found(name: str, tools: Sequence[str]) -> dict[str, str]:
    """The answer to a call of `name`, a tool the planner lacks, naming the tools
    it can call in the order they are declared: the ADK planner's answer, word
    for word."""
    available = ", ".join(tools) or "none"
    return {
        "error": (
            f"Invoking `{name}()` failed as no tool with that name is available. "
            f"The tools you can call are: {available}. You could retry, but it is "
            "IMPORTANT that you only call a tool from that list."
        )
    }


def planner_agent(
    specs: Sequence[ToolSpec],
    *,
    instruction: str,
    model: Model,
    name: str | None = None,
    include_content: bool = True,
) -> PlannerAgent:
    """The planner as a Pydantic AI agent named `name` on `model`, with
    `instruction` and one tool per spec.

    Each tool is declared as `function_declaration(spec)` states it, its
    schema as it stands, and runs the spec's body over the model's arguments
    as sent (`_tool_function`). Every tool is declared not strict, which keeps
    the request's function calling mode `AUTO`, the mode the ADK planner's
    request leaves Gemini on. A response with no tool call ends the run, an
    empty one included, as it ends the ADK planner's. Thinking is
    `planner.settings.AGENT_THINKING_CONFIG`.

    A call to a tool the planner lacks is answered with `tool_not_found`, as
    the ADK planner answers it, and the run goes on however many such calls
    the model makes (`_UNLIMITED_RETRIES`).

    The agent traces through the global tracer provider: a span per run
    (`invoke_agent <name>`), per model call (`chat <model>`) and per tool call
    (`execute_tool <tool>`), with the OpenTelemetry GenAI attributes. With
    `include_content` the spans carry the model's messages, the instruction
    and each call's arguments and answer. The request parameters are left
    off: the tool declarations are on each model call's span already.
    """
    tools = tuple(spec.name for spec in specs)
    agent = Agent(
        model,
        instructions=instruction,
        tools=[_tool(spec) for spec in specs],
        deps_type=PlannerDeps,
        output_type=str | None,
        model_settings=GoogleModelSettings(
            google_thinking_config=AGENT_THINKING_CONFIG.model_dump(mode="json", exclude_none=True)
        ),
        retries={"tools": _UNLIMITED_RETRIES},
        capabilities=[
            ProcessHistory(_answering_unknown_tools(tools)),
            Instrumentation(
                InstrumentationSettings(
                    include_content=include_content, include_model_request_parameters=False
                )
            ),
        ],
        name=name,
    )
    return PlannerAgent(agent=agent, tools=tools)


def _answering_unknown_tools(
    tools: Sequence[str],
) -> Callable[[list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """A history processor that answers each call to a tool not in `tools` with
    `tool_not_found`, in place of the retry Pydantic AI answers it with. It runs
    before every model request, so the model reads the answer, and the run's
    messages keep it, so a later call's steps read it too."""

    async def process(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            replace(message, parts=[_answered_if_unknown(part, tools) for part in message.parts])
            if isinstance(message, ModelRequest)
            else message
            for message in messages
        ]

    return process


def _answered_if_unknown(part: ModelRequestPart, tools: Sequence[str]) -> ModelRequestPart:
    """`part`, or `tool_not_found`'s answer in its place when it answers a call
    to a tool not in `tools`."""
    if not isinstance(part, ToolReturnPart) or part.tool_name in tools:
        return part
    return ToolReturnPart(
        tool_name=part.tool_name,
        content=tool_not_found(part.tool_name, tools),
        tool_call_id=part.tool_call_id,
    )


def _tool(spec: ToolSpec) -> Tool[PlannerDeps]:
    declaration = function_declaration(spec)
    schema = declaration.parameters_json_schema
    tool = Tool.from_schema(
        _tool_function(spec),
        name=spec.name,
        description=declaration.description,
        json_schema=dict(schema) if schema is not None else _NO_PARAMETERS,
        takes_ctx=True,
    )
    tool.strict = False
    return tool


def _tool_function(spec: ToolSpec) -> Callable[..., Any]:
    """The function Pydantic AI calls for `spec`, with the model's arguments unvalidated.

    It runs the spec's body (`run_tool`) over the call's `PydanticCall`, whose
    state is a view of the call's own. An answer that is no dict is wrapped
    under `result`, as ADK wraps one. A body that raises is answered with
    `tool_failed`'s answer, as the ADK planner's tool error callback answers
    it, and a cancellation propagates. An answer of the body's holding an
    `error` key, the refusal of a call missing a required argument
    (`planner.tool_spec.missing_arguments`), marks the call's span failed, as
    the ADK planner marks its tool's (`_mark_tool_error`). Either answer goes
    to `tool_answered` and, unless that raises, is kept with the call's writes
    for the run to yield.
    """

    async def call(ctx: RunContext[PlannerDeps], **raw: Any) -> dict[str, Any]:
        deps = ctx.deps
        record = _Call(name=spec.name, view=RecordingState(deps.state))
        if ctx.tool_call_id is not None:
            deps._calls[ctx.tool_call_id] = record

        def end_run() -> None:
            record.ended = True
            deps.end_run()

        planner_call = PydanticCall(
            state=record.view,
            call_id=ctx.tool_call_id,
            earlier=deps.earlier,
            messages=tuple(ctx.messages),
        )
        try:
            answer = run_tool(spec, raw, call=planner_call, end_run=end_run)
            if inspect.isawaitable(answer):
                answer = await answer
            if isinstance(answer, dict) and answer.get("error"):
                _mark_tool_error(trace.get_current_span())
        except Exception as exc:
            answer = deps.tool_failed(spec.name, exc)
        response = as_function_response(answer)
        writes = deps._written(record.view)
        deps.tool_answered(spec.name, response)
        record.writes, record.answer = writes, response
        return response

    call.__name__ = spec.name
    return call


def _mark_tool_error(span: trace.Span) -> None:
    """Mark `span`, a tool call's, failed with `error.type` `TOOL_ERROR`, as ADK
    marks a tool's whose answer holds an `error` key."""
    span.set_attribute(ERROR_TYPE, _TOOL_ERROR)
    span.set_status(Status(StatusCode.ERROR, _TOOL_ERROR))


async def run_planner(
    planner: PlannerAgent,
    deps: PlannerDeps,
    *,
    question: str | None,
    brief: str,
    conversation_id: str | None = None,
) -> AsyncIterator[PlannerItem]:
    """Run the planner once on `brief`, after the user's `question`, yielding as it goes.

    The run's first request holds `question`, then `brief`, as two user
    contents, or `brief` alone for a turn with no question, as the ADK
    planner sends it (`planner.agent.prepend_brief`). Each model response is
    yielded as a `PlannerResponse` once received, before its calls run, and
    each call's answer as a `PlannerToolAnswer` once that call has answered,
    a call to a tool the planner lacks included (`tool_not_found`). The run stops before its next
    model request once a tool ended it, and ends on a response with no tool
    call.

    A response Pydantic AI refuses (`UnexpectedModelBehavior`: a stream with
    no chunk, a blocked prompt) ends the run as a response with no tool call
    ends the ADK planner's: the items yielded so far stand and nothing is
    raised. A blocked prompt's response is yielded first, with its
    `block_reason`. Any other exception is raised once the answers of the
    calls that answered and were not yet yielded are, in the order the calls
    began, as the ADK planner keeps a response's finished calls when another
    of its calls fails. A cancellation yields nothing more.

    `deps` holds the session's state, the earlier steps and the callbacks
    (`PlannerDeps`). `conversation_id` is the conversation the run's spans
    name (`gen_ai.conversation.id`); None leaves Pydantic AI to make one up.
    """
    history = (
        [ModelRequest(parts=[UserPromptPart(question)], instructions=_QUESTION_INSTRUCTIONS)]
        if question is not None
        else []
    )
    async with planner.agent.iter(
        brief,
        message_history=history,
        deps=deps,
        usage_limits=_NO_REQUEST_LIMIT,
        conversation_id=conversation_id,
    ) as run:
        try:
            async for node in run:
                if Agent.is_model_request_node(node):
                    if deps.ended:
                        return
                    async with node.stream(run.ctx) as stream:
                        async for _event in stream:
                            pass
                        response = stream.response
                    yield _planner_response(response)
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as events:
                        async for event in events:
                            if not isinstance(event, FunctionToolResultEvent):
                                continue
                            answered = _answered(deps, event, planner.tools)
                            if answered is not None:
                                yield answered
        except UnexpectedModelBehavior as exc:
            logger.warning("planner run ended on a response Pydantic AI refused: %s", exc)
        except Exception:
            for call_id, call in list(deps._calls.items()):
                if call.answer is not None:
                    del deps._calls[call_id]
                    yield _answer_of(call_id, call)
            raise


def _planner_response(response: ModelResponse) -> PlannerResponse:
    block_reason = (response.provider_details or {}).get("block_reason")
    return PlannerResponse(
        parts=response_parts(response),
        usage=response.usage,
        block_reason=str(block_reason) if block_reason else None,
    )


def _answered(
    deps: PlannerDeps, event: FunctionToolResultEvent, tools: Sequence[str]
) -> PlannerToolAnswer | None:
    """The answer of the call `event` reports: a planner tool's, taken out of
    what the run has not yielded, or `tool_not_found`'s for a tool the planner
    lacks, which writes nothing."""
    name = event.part.tool_name
    if name not in tools:
        return PlannerToolAnswer(
            name=name, call_id=event.tool_call_id, answer=tool_not_found(name, tools), writes={}
        )
    call = deps._calls.get(event.tool_call_id)
    if call is None or call.answer is None or call.writes is None:
        return None
    del deps._calls[event.tool_call_id]
    return _answer_of(event.tool_call_id, call)


def _answer_of(call_id: str, call: _Call) -> PlannerToolAnswer:
    """The answer of `call`, which has answered."""
    assert call.answer is not None and call.writes is not None
    return PlannerToolAnswer(call.name, call_id, call.answer, call.writes, call.ended)


def response_parts(response: ModelResponse) -> tuple[types.Part, ...]:
    """`response`'s parts as the google-genai parts Gemini sent for them.

    A thought signature is restored to the part that carried it, which
    Pydantic AI keeps in the part's `provider_details`.

    Raises:
        ValueError: A part is none of a function call, text or a thought.
    """
    parts: list[types.Part] = []
    for part in response.parts:
        details = part.provider_details or {}
        encoded = details.get("thought_signature")
        signature = base64.b64decode(encoded) if encoded else None
        match part:
            case ToolCallPart():
                call = types.FunctionCall(
                    id=part.tool_call_id, name=part.tool_name, args=part.args_as_dict()
                )
                parts.append(types.Part(function_call=call, thought_signature=signature))
            case TextPart():
                parts.append(types.Part(text=part.content, thought_signature=signature))
            case ThinkingPart():
                parts.append(
                    types.Part(text=part.content, thought=True, thought_signature=signature)
                )
            case _:
                raise ValueError(f"the planner's response holds a {part.part_kind} part")
    return tuple(parts)
