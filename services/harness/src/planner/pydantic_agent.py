"""The Pydantic AI planner: its agent (`planner_agent`) and its run (`run_planner`).

The agent serves each `planner.tool_spec.ToolSpec` with the declaration the
ADK planner shows for it (`function_declaration`), and a call runs the spec's
body over a `planner.session.SessionCall` (`run_tool`), with the
model's arguments as sent. `run_planner` drives one run with `Agent.iter`
from the user's message and the brief, as the ADK planner's `single_turn`
run starts from the turn's question and the brief
(`planner.agent.prepend_brief`), and yields each model response as it
arrives and then its calls' answers, with the state they wrote, once every
call has answered. A tool that ends the run (`planner.tools.EndRun`) stops it
before the next model request.

Pydantic AI shapes a request its own way: the system instruction as a
`Content`, one `Tool` per declaration, an empty parameters schema on a tool
that takes none, an explicit `AUTO` tool config and `response_modalities`.
The planner's model (`harness.planner_model`) puts each request in the ADK
planner's form before it is sent, so the model is sent the ADK planner's
request for the same turn but for each framework's HTTP headers. The model
must be built with `PLANNER_PROFILE`, which sends each declaration's schema
as given and adds nothing to the instruction.
"""

from __future__ import annotations

import base64
import copy
import inspect
import logging
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from google.genai import types
from opentelemetry import trace
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Status, StatusCode
from pydantic_ai import Agent, InstrumentationSettings, RunContext, Tool
from pydantic_ai.capabilities import Instrumentation, ProcessHistory
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.profiles.google import GoogleModelProfile
from pydantic_ai.usage import UsageLimits

from planner.call import RecordingState
from planner.session import SessionCall, SessionEvents
from planner.settings import AGENT_THINKING_CONFIG
from planner.tool_spec import ToolSpec, function_declaration, run_tool

logger = logging.getLogger(__name__)

# The schema Pydantic AI declares a tool with when its declaration has none.
# Pydantic AI always sends `parameters_json_schema`.
_NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}

# The turn budget bounds a run's model calls (`harness.turn_budget`), so Pydantic
# AI's own default cap of 50 requests is lifted.
_NO_REQUEST_LIMIT = UsageLimits(request_limit=None)

# Pydantic AI charges a call to a tool the planner lacks against the tool retry
# budget and ends the run once it is spent. The ADK planner answers every such
# call and goes on, so the budget is lifted and the turn budget's call cap ends
# a run that keeps making them. The planner's own tools take their arguments
# unvalidated, so none asks for a retry.
_UNLIMITED_RETRIES = sys.maxsize

# Carried by the request holding the user's message. Pydantic AI merges
# consecutive requests into one content unless their `instructions` differ,
# and the brief's request carries the agent's instruction, so this keeps the
# user's message and the brief two contents, as the ADK planner sends them.
# Only the latest request's instructions reach the model, so this text is not
# sent.
_QUESTION_INSTRUCTIONS = "The user's message, which opened the turn."

# The `error.type` of a tool call whose answer is an error, ADK's.
_TOOL_ERROR = "TOOL_ERROR"

type ToolAnswered = Callable[[str, dict[str, Any]], None]
type ToolFailed = Callable[[str, BaseException], dict[str, Any]]


# The planner's changes to Gemini's profile, which `GoogleModel(profile=...)`
# merges over the provider's own for the model. Declarations are sent as given
# (no JSON schema transformer), and nothing is added to the instruction about
# system messages mid-conversation (inline system prompts supported).
PLANNER_PROFILE = GoogleModelProfile(
    json_schema_transformer=None, supports_inline_system_prompts=True
)


@dataclass
class PlannerDeps:
    """One run's dependencies, which every tool call of the run reads.

    `state` is the session's state. The calls of one model response write it
    through one `RecordingState`, which `run_planner` replaces before each
    model request, so the keys a response's calls wrote are told apart from
    the keys of the responses before it. `events` is the session's events,
    the live list each call reads its steps from
    (`planner.session.SessionCall`). `tool_answered` is told each tool's
    answer and `tool_failed` each tool that raised. `ended` says whether a
    tool ended the run.
    """

    state: MutableMapping[str, Any]
    events: SessionEvents
    tool_answered: ToolAnswered
    tool_failed: ToolFailed
    ended: bool = False
    _response_view: RecordingState = field(init=False)

    def __post_init__(self) -> None:
        self._response_view = RecordingState(self.state)

    def end_run(self) -> None:
        """Record that a tool ended the run, so no model request follows the current calls."""
        self.ended = True


@dataclass(frozen=True)
class PlannerResponse:
    """One model response of the run, as soon as it is received and before its calls run.

    `parts` are its parts as google-genai parts: each function call with its
    id, text, and each thought with `thought` set, each with its thought
    signature. What the API reported on the response (usage, finish reason, a
    blocked prompt) is the stream's metadata, which `harness.planner_model`
    reads as `ResponseMetadata`.
    """

    parts: tuple[types.Part, ...]


@dataclass(frozen=True)
class PlannerAnswers:
    """The answers of one model response's calls, once every call has answered.

    `answers` holds each call's answer as a function response, with the call's
    id and tool, in the order of the calls in the response. `state` is the
    session's value, after the last call answered, of every key the
    response's calls wrote. A key deleted since holds no value and is left out.
    """

    answers: tuple[types.FunctionResponse, ...]
    state: dict[str, Any]


type PlannerItem = PlannerResponse | PlannerAnswers


@dataclass(frozen=True)
class PydanticPlanner:
    """The Pydantic AI planner (`planner_agent`): the agent, its tools' names,
    in the order they are declared, and whether its spans carry message
    content (`include_content`)."""

    agent: Agent[PlannerDeps, str | None]
    tools: tuple[str, ...]
    include_content: bool


def tool_not_found(name: str, tools: Sequence[str]) -> dict[str, str]:
    """The answer to a call of `name`, a tool the planner lacks, naming the tools
    it can call in the order they are declared. It is the ADK planner's answer,
    word for word."""
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
    model: Model | None = None,
    name: str | None = None,
    include_content: bool = True,
) -> PydanticPlanner:
    """The planner as a Pydantic AI agent named `name` on `model`, with
    `instruction` and one tool per spec. With no `model` each run names its
    own (`run_planner`), so one agent serves concurrent runs, each on a model
    of its own.

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
    off, since the tool declarations are on each model call's span.
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
    return PydanticPlanner(agent=agent, tools=tools, include_content=include_content)


def _answering_unknown_tools(
    tools: Sequence[str],
) -> Callable[[list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """A history processor that answers each call to a tool not in `tools` with
    `tool_not_found`, in place of the retry Pydantic AI answers it with. It runs
    before every model request, so the model reads the answer."""

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

    It runs the spec's body (`run_tool`) over the call's `SessionCall`, whose
    state is the view the response's calls write through. An answer that is
    no dict is wrapped under `result`, as ADK wraps one. A body that raises is
    answered with `tool_failed`'s answer, as the ADK planner's tool error
    callback answers it, and a cancellation propagates. An answer of the
    body's holding an `error` key, the refusal of a call missing a required
    argument (`planner.tool_spec.missing_arguments`), marks the call's span
    failed, as the ADK planner marks its tool's (`_mark_tool_error`). Either
    answer goes to `tool_answered` as the call finishes, and what that raises
    fails the run.
    """

    async def call(ctx: RunContext[PlannerDeps], **raw: Any) -> dict[str, Any]:
        deps = ctx.deps
        planner_call = SessionCall(deps.events, deps._response_view, ctx.tool_call_id)
        try:
            answer = run_tool(spec, raw, call=planner_call, end_run=deps.end_run)
            if inspect.isawaitable(answer):
                answer = await answer
            if isinstance(answer, dict) and answer.get("error"):
                _mark_tool_error(trace.get_current_span())
        except Exception as exc:
            answer = deps.tool_failed(spec.name, exc)
        response = as_function_response(answer)
        deps.tool_answered(spec.name, response)
        return response

    call.__name__ = spec.name
    return call


def _mark_tool_error(span: trace.Span) -> None:
    """Mark `span`, a tool call's, failed with `error.type` `TOOL_ERROR`, as ADK
    marks a tool's whose answer holds an `error` key."""
    span.set_attribute(ERROR_TYPE, _TOOL_ERROR)
    span.set_status(Status(StatusCode.ERROR, _TOOL_ERROR))


async def run_planner(
    planner: PydanticPlanner,
    deps: PlannerDeps,
    *,
    question: types.Content | str | None,
    brief: str,
    model: Model | None = None,
    conversation_id: str | None = None,
) -> AsyncGenerator[PlannerItem]:
    """Run the planner once on `brief`, after the user's `question`, yielding as it goes.

    The run's first request holds `question`, then `brief`, as two user
    contents, or `brief` alone for a turn with no question, as the ADK
    planner sends it (`planner.agent.prepend_brief`). A question given as a
    `Content` is sent with its parts as the user sent them
    (`_question_prompt`). Each model response is yielded as a
    `PlannerResponse` once received, before its calls run. Once every call
    has answered, their answers are yielded as one `PlannerAnswers`, before
    the next model request: the tool returns Pydantic AI collects into that
    request (`ModelRequestNode.request`), in call order, with
    `tool_not_found`'s answer for a call to a tool the planner lacks. The run
    stops after yielding them once a tool ended it, and ends on a response
    with no tool call.

    A response Pydantic AI refuses (`UnexpectedModelBehavior`, as for a stream
    with no chunk or a blocked prompt) ends the run, as a response with no tool
    call ends the ADK planner's, and raises nothing. A blocked prompt's response is
    yielded first, with no parts. Any other exception is raised. A run that
    ends either way while a response's calls run yields none of their
    answers. Closing the iterator closes the run (`Agent.iter`) and any
    stream it holds open.

    `deps` holds the session's state and events and the callbacks
    (`PlannerDeps`). `model` is the model the run calls, or None for the
    agent's own. `conversation_id` is the conversation the run's spans name
    (`gen_ai.conversation.id`). None leaves Pydantic AI to make one up.
    """
    history = (
        [
            ModelRequest(
                parts=[UserPromptPart(_question_prompt(question))],
                instructions=_QUESTION_INSTRUCTIONS,
            )
        ]
        if question is not None
        else []
    )
    async with planner.agent.iter(
        brief,
        model=model,
        message_history=history,
        deps=deps,
        usage_limits=_NO_REQUEST_LIMIT,
        conversation_id=conversation_id,
    ) as run:
        try:
            async for node in run:
                if not Agent.is_model_request_node(node):
                    continue
                returns = [part for part in node.request.parts if isinstance(part, ToolReturnPart)]
                if returns:
                    yield _answers(returns, deps, planner.tools)
                if deps.ended:
                    return
                deps._response_view = RecordingState(deps.state)
                async with node.stream(run.ctx) as stream:
                    async for _event in stream:
                        pass
                    response = stream.response
                yield _planner_response(response)
        except UnexpectedModelBehavior as exc:
            logger.warning("planner run ended on a response Pydantic AI refused: %s", exc)


class UnsentQuestionPartError(ValueError):
    """The turn's question holds a part the Pydantic AI planner does not send
    (`_question_prompt`)."""


def _question_prompt(question: types.Content | str) -> str | list[UserContent]:
    """`question`'s parts as the user prompt Pydantic AI sends as one content of
    the same parts: one text part as its text, several as a list of them.

    Studio sends one text part (`harness.studio`). An inline file is sent as
    `BinaryContent`, its bytes and media type unchanged. Any other part, a
    file by URI or a function part among them, is refused. The ADK planner
    sends every part of the question as it is.

    Raises:
        UnsentQuestionPartError: A part is neither text nor an inline file,
            the message naming the part's fields, or is an inline file with
            no bytes.
    """
    if isinstance(question, str):
        return question
    prompt: list[UserContent] = []
    for part in question.parts or ():
        if part.text is not None:
            prompt.append(part.text)
        elif part.inline_data is not None:
            if part.inline_data.data is None:
                raise UnsentQuestionPartError(
                    "the question holds an inline_data part with no bytes, so the "
                    "Pydantic AI planner has no file to send for it"
                )
            media_type = part.inline_data.mime_type or "application/octet-stream"
            prompt.append(BinaryContent(data=part.inline_data.data, media_type=media_type))
        else:
            received = sorted(part.model_dump(exclude_none=True))
            raise UnsentQuestionPartError(
                f"the question holds a part with {received}; the Pydantic AI planner "
                "takes text and inline data (inline_data) parts only"
            )
    if len(prompt) == 1 and isinstance(prompt[0], str):
        return prompt[0]
    return prompt


def _planner_response(response: ModelResponse) -> PlannerResponse:
    return PlannerResponse(parts=response_parts(response))


def _answers(
    returns: Sequence[ToolReturnPart], deps: PlannerDeps, tools: Sequence[str]
) -> PlannerAnswers:
    """The answers `returns`, a response's tool returns in call order, stand for:
    a planner tool's own answer, or `tool_not_found`'s for a tool the planner
    lacks, with the state the response's calls wrote."""
    view = deps._response_view
    return PlannerAnswers(
        answers=tuple(
            types.FunctionResponse(
                id=part.tool_call_id,
                name=part.tool_name,
                response=as_function_response(part.content)
                if part.tool_name in tools
                else tool_not_found(part.tool_name, tools),
            )
            for part in returns
        ),
        state={key: copy.deepcopy(deps.state[key]) for key in view.written if key in deps.state},
    )


def as_function_response(value: object) -> dict[str, Any]:
    """`value` as a function response carries it: a dict as it is, anything else
    under `result`, as ADK wraps a tool's answer that is no dict."""
    if isinstance(value, dict):
        # A function response's keys are the JSON object's, so strings.
        return cast("dict[str, Any]", value)
    return {"result": value}


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
